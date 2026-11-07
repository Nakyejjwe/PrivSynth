"""Building blocks shared by every evaluator.

Every evaluator answers one question about a synthetic dataset, either
"how *useful* is it?" (utility) or "how much does it still reveal about
real people?" (risk). They all follow the same interface::

    result = evaluator.evaluate(real, synthetic, holdout)

* ``real``      - the records the synthesizer was trained on
* ``synthetic`` - the records it generated
* ``holdout``   - real records the synthesizer never saw (optional)

The holdout set is what makes the risk numbers honest. A synthetic record
that is close to a real person is only a problem if it is closer than an
*unrelated* real person would be; the holdout supplies those unrelated
people as a baseline.

:class:`Evaluator` is abstract. :meth:`Evaluator.evaluate` is a template
method: it checks the inputs once, here, and each subclass only writes
:meth:`Evaluator._evaluate`. This is why the pipeline, the experiment
runner and (in Week 4) the dashboard can run any list of evaluators
without knowing which ones they are (polymorphism).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.exceptions import EvaluationError, SchemaError
from privsynth.generation.encoding import TableEncoder


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of one evaluator on one synthetic dataset (immutable).

    Attributes:
        evaluator: Display name of the evaluator, e.g. ``"Marginal similarity"``.
        key: Short identifier used to prefix metric names, e.g. ``"marginal"``.
        kind: ``"utility"`` or ``"risk"``.
        metrics: Metric name -> value. Every value is a plain float so the
            results can go straight into a table, CSV or chart.
        lower_is_better: Names of metrics where a smaller value is better.
        details: Optional per-column or per-record breakdown.
        notes: Human-readable remarks, e.g. a fallback that was used.
    """

    evaluator: str
    key: str
    kind: str
    metrics: Mapping[str, float]
    lower_is_better: frozenset = frozenset()
    details: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)
    notes: tuple[str, ...] = ()

    def __getitem__(self, metric: str) -> float:
        return self.metrics[metric]

    def to_dict(self, prefix: bool = True) -> dict[str, float]:
        """Metrics as a flat dict; keys become ``"key.metric"`` when ``prefix`` is True."""
        if not prefix:
            return dict(self.metrics)
        return {f"{self.key}.{name}": value for name, value in self.metrics.items()}

    def __str__(self) -> str:
        lines = [f"{self.evaluator} ({self.kind})"]
        for name, value in self.metrics.items():
            arrow = "lower is better" if name in self.lower_is_better else ""
            lines.append(f"  {name:<28} {_format(value):>10}  {arrow}".rstrip())
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


class Evaluator(ABC):
    """Abstract base class for utility and privacy-risk evaluators.

    Subclasses set the class attributes and implement :meth:`_evaluate`.
    """

    #: Display name shown in reports.
    name: str = "evaluator"
    #: Short identifier used as the metric prefix in tables.
    key: str = "evaluator"
    #: ``"utility"`` or ``"risk"``.
    kind: str = "utility"
    #: True if the evaluator cannot run without a holdout set.
    requires_holdout: bool = False

    def evaluate(
        self,
        real: Dataset,
        synthetic: Dataset,
        holdout: Optional[Dataset] = None,
    ) -> EvaluationResult:
        """Compare ``synthetic`` with ``real`` (and ``holdout`` if given).

        Raises:
            SchemaError: If an input is not a Dataset or the columns differ.
            EvaluationError: If a holdout set is required but missing.
        """
        _check_dataset(real, "real")
        _check_dataset(synthetic, "synthetic")
        synthetic = _align_columns(real, synthetic, "synthetic")
        if holdout is not None:
            _check_dataset(holdout, "holdout")
            holdout = _align_columns(real, holdout, "holdout")
        elif self.requires_holdout:
            raise EvaluationError(
                f"{self.name} needs a holdout set of real records the synthesizer never saw."
            )
        return self._evaluate(real, synthetic, holdout)

    @abstractmethod
    def _evaluate(
        self, real: Dataset, synthetic: Dataset, holdout: Optional[Dataset]
    ) -> EvaluationResult:
        """Compute the metrics (inputs are already validated and aligned)."""

    # ------------------------------------------------------------------ #
    def _result(
        self,
        metrics: Mapping[str, float],
        lower_is_better: Iterable[str] = (),
        details: Optional[pd.DataFrame] = None,
        notes: Sequence[str] = (),
    ) -> EvaluationResult:
        """Package metrics into an :class:`EvaluationResult` for this evaluator."""
        return EvaluationResult(
            evaluator=self.name,
            key=self.key,
            kind=self.kind,
            metrics={name: float(value) for name, value in metrics.items()},
            lower_is_better=frozenset(lower_is_better),
            details=details,
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #
def encode_together(
    datasets: Sequence[Dataset],
    n_bins: int = 16,
    bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
    max_exact_integers: int = 32,
) -> tuple[TableEncoder, list[pd.DataFrame]]:
    """Encode several datasets with **one** encoder fitted on all of them.

    Comparing histograms only makes sense if "bin 3 of age" means the same
    range in the real and the synthetic data, so the domain is learned from
    the union. (Evaluation is done by the data owner, not released, so this
    does not spend privacy budget.)

    Args:
        datasets: Datasets with the same columns.
        n_bins: Bins for numeric columns.
        bin_edges: Explicit bin edges per numeric column.
        max_exact_integers: Whole-number columns with at most this many
            possible values keep one code per value instead of being binned.

    Returns:
        The fitted encoder and one DataFrame of integer codes per dataset.
    """
    combined = pd.concat([d.data for d in datasets], ignore_index=True)
    encoder = TableEncoder(n_bins=n_bins, bin_edges=_widen_edges(combined, bin_edges),
                           max_exact_integers=max_exact_integers)
    encoder.fit(Dataset(combined, name="combined"))
    codes = [encoder.encode(Dataset(d.data, name=d.name)) for d in datasets]
    return encoder, codes


def _widen_edges(
    frame: pd.DataFrame, bin_edges: Optional[Mapping[str, Sequence[float]]]
) -> dict[str, list[float]]:
    """Copy ``bin_edges``, making sure every value falls inside the outer edges."""
    edges = {col: list(map(float, e)) for col, e in (bin_edges or {}).items() if col in frame.columns}
    for col, e in edges.items():
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            e[0] = min(e[0], float(values.min()))
            e[-1] = max(e[-1], float(values.max()) + 1)
    return edges


class MixedFeatureSpace:
    """Turn mixed numeric/categorical records into points for distance calculations.

    Numeric columns are scaled to ``[0, 1]`` using the range seen in
    :meth:`fit`; categorical columns are one-hot encoded and multiplied by
    ``1/sqrt(2)``. With that factor, two records that differ in one
    category are exactly distance 1 apart, the same as two records at
    opposite ends of a numeric range, so no column dominates.
    """

    def __init__(self) -> None:
        self._numeric: list[str] = []
        self._ranges: dict[str, tuple[float, float]] = {}
        self._categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame, numeric: Iterable[str]) -> "MixedFeatureSpace":
        self._numeric = [c for c in frame.columns if c in set(numeric)]
        self._ranges = {}
        for col in self._numeric:
            values = frame[col].to_numpy(dtype=float)
            self._ranges[col] = (float(values.min()), float(values.max()))
        self._categories = {
            col: sorted(frame[col].astype(str).unique())
            for col in frame.columns
            if col not in self._numeric
        }
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a float array with one row per record."""
        if not self._ranges and not self._categories:
            raise EvaluationError("MixedFeatureSpace must be fitted before transform().")
        parts = []
        for col in self._numeric:
            low, high = self._ranges[col]
            span = high - low if high > low else 1.0
            parts.append(((frame[col].to_numpy(dtype=float) - low) / span)[:, None])
        for col, cats in self._categories.items():
            index = pd.Index(cats)
            positions = index.get_indexer(frame[col].astype(str))
            onehot = np.zeros((len(frame), len(cats)))
            known = positions >= 0          # an unseen category stays all-zero
            onehot[np.flatnonzero(known), positions[known]] = 1.0 / math.sqrt(2.0)
            parts.append(onehot)
        return np.hstack(parts).astype(np.float64)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two histograms of the same shape.

    ``TVD = 0.5 * sum |p_i - q_i|`` after normalising both to sum to 1.
    0 means identical distributions; 1 means they never overlap.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError("Histograms must have the same shape.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Histogram counts cannot be negative.")
    sp, sq = p.sum(), q.sum()
    if sp == 0 and sq == 0:
        return 0.0
    if sp == 0 or sq == 0:
        return 1.0
    return 0.5 * float(np.abs(p / sp - q / sq).sum())


def _check_dataset(obj: object, role: str) -> None:
    if not isinstance(obj, Dataset):
        raise SchemaError(f"The {role} data must be a Dataset, got {type(obj).__name__}.")


def _align_columns(real: Dataset, other: Dataset, role: str) -> Dataset:
    """Return ``other`` with its columns in the same order as ``real``."""
    if set(other.columns) != set(real.columns):
        missing = sorted(set(real.columns) - set(other.columns))
        extra = sorted(set(other.columns) - set(real.columns))
        raise SchemaError(f"The {role} data has different columns (missing {missing}, extra {extra}).")
    if other.columns == real.columns:
        return other
    return other.select(real.columns)


def _format(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.4f}"
