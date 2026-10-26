"""The abstract :class:`Synthesizer` that every generator inherits from.

Every generator follows the same two-step life cycle::

    synth = SomeSynthesizer(epsilon=1.0)
    synth.fit(real_dataset)          # looks at the real data, spends the budget
    fake = synth.sample(1000)        # free: uses only the noisy statistics

``fit`` and ``sample`` are *template methods*: the shared work (input
checks, encoding, budget creation, decoding the output back into a
:class:`Dataset`) lives here, while each subclass fills in ``_fit`` (which
noisy statistics to measure) and ``_sample`` (how to draw records from
them). This is the abstraction and polymorphism the proposal describes:
the pipeline and dashboard call ``fit``/``sample`` on any synthesizer.

Why sampling is free: by the *post-processing* property of differential
privacy [Dwork & Roth, 2014, Prop 2.1], anything computed only from DP
outputs is still DP with the same epsilon. Once ``fit`` is done, the real
data is never touched again, so we can draw as many records as we like.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.exceptions import NotFittedError, SchemaError
from privsynth.generation.encoding import TableEncoder
from privsynth.privacy.budget import PrivacyBudget, _check_epsilon
from privsynth.privacy.mechanisms import (
    NOISE_MECHANISMS,
    RandomState,
    _as_rng,
    make_noise_mechanism,
)


class Synthesizer(ABC):
    """Abstract base class for differentially private synthetic data generators.

    Args:
        epsilon: Total privacy budget for :meth:`fit`. Smaller = more private.
        noise: ``"laplace"`` (pure epsilon-DP) or ``"gaussian"`` ((epsilon, delta)-DP).
        delta: Total delta. Defaults to 0 for Laplace and 1e-5 for Gaussian.
        n_bins: Bins for numeric columns (see :class:`TableEncoder`).
        bin_edges: Explicit bin edges per numeric column.
        categories: Explicit category lists per categorical column.
        random_state: Seed for reproducible results.
    """

    #: Display name, set by each subclass.
    name: str = "synthesizer"

    def __init__(
        self,
        epsilon: float = 1.0,
        noise: str = "laplace",
        delta: Optional[float] = None,
        n_bins: int = 16,
        bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
        categories: Optional[Mapping[str, Sequence]] = None,
        random_state: RandomState = None,
    ) -> None:
        _check_epsilon(epsilon)
        noise = noise.lower().strip()
        if noise not in NOISE_MECHANISMS:
            raise ValueError(f"Unknown noise {noise!r}; choose from {sorted(NOISE_MECHANISMS)}.")
        if delta is None:
            delta = 1e-5 if noise == "gaussian" else 0.0
        if noise == "gaussian" and delta <= 0:
            raise ValueError("Gaussian noise needs delta > 0.")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.noise = noise
        self._encoder = TableEncoder(n_bins=n_bins, bin_edges=bin_edges, categories=categories)
        self._rng = _as_rng(random_state)
        self._budget: Optional[PrivacyBudget] = None
        self._schema: Optional[dict] = None
        self._noisy_total: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public interface (template methods)
    # ------------------------------------------------------------------ #
    def fit(self, dataset: Dataset) -> "Synthesizer":
        """Learn noisy statistics from ``dataset``, spending the privacy budget.

        Each call starts a fresh budget of ``epsilon``; refitting on the
        same data therefore costs another ``epsilon`` in reality, which the
        caller (later the pipeline) must account for.

        Returns:
            ``self``, so calls can be chained: ``synth.fit(d).sample(100)``.

        Raises:
            SchemaError: If ``dataset`` is not a :class:`Dataset`.
        """
        if not isinstance(dataset, Dataset):
            raise SchemaError(f"fit() expects a Dataset, got {type(dataset).__name__}.")
        self._budget = PrivacyBudget(self.epsilon, self.delta)
        self._encoder.fit(dataset)
        codes = self._encoder.encode(dataset)
        self._noisy_total = None
        self._fit(codes, n_records=dataset.n_rows)
        # Keep only the schema, never the real records.
        self._schema = {
            "name": dataset.name,
            "columns": dataset.columns,
            "dtypes": dataset.data.dtypes.to_dict(),
            "quasi_identifiers": dataset.quasi_identifiers or None,
            "sensitive_attribute": dataset.sensitive_attribute,
        }
        return self

    def sample(self, n: Optional[int] = None) -> Dataset:
        """Generate ``n`` synthetic records as a :class:`Dataset`.

        If ``n`` is omitted, the size is the *noisy* record count measured
        during :meth:`fit`, so the true size is not revealed either.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
            ValueError: If ``n`` is not a positive integer.
        """
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before sampling.")
        if n is None:
            n = max(1, int(round(self._noisy_total or 1)))
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        codes = self._sample(int(n))
        frame = self._encoder.decode(codes, rng=self._rng)
        frame = self._restore_dtypes(frame)
        return Dataset(
            frame,
            name=f"{self._schema['name']} - {self.name} (eps={self.epsilon:g})",
            quasi_identifiers=self._schema["quasi_identifiers"],
            sensitive_attribute=self._schema["sensitive_attribute"],
        )

    def fit_sample(self, dataset: Dataset, n: Optional[int] = None) -> Dataset:
        """Convenience: :meth:`fit` then :meth:`sample`."""
        return self.fit(dataset).sample(n)

    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._schema is not None

    @property
    def budget(self) -> PrivacyBudget:
        """The budget used by the last :meth:`fit` (its ledger shows every charge)."""
        if self._budget is None:
            raise NotFittedError("No budget yet: the synthesizer has not been fitted.")
        return self._budget

    @property
    def encoder(self) -> TableEncoder:
        return self._encoder

    # ------------------------------------------------------------------ #
    # Steps that subclasses implement
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit(self, codes: pd.DataFrame, n_records: int) -> None:
        """Measure noisy statistics from the encoded data, using ``self._measure``."""

    @abstractmethod
    def _sample(self, n: int) -> pd.DataFrame:
        """Return ``n`` rows of integer codes drawn from the noisy statistics."""

    # ------------------------------------------------------------------ #
    # Helpers shared by subclasses
    # ------------------------------------------------------------------ #
    def _measure(self, counts: np.ndarray, epsilon: float, delta: float, label: str) -> np.ndarray:
        """Release a histogram of counts with DP noise, charged to the budget.

        A histogram has sensitivity 1 (one record changes one cell by 1),
        in both L1 and L2. Negative noisy counts are set to 0
        (post-processing, so still private). The first table measured also
        gives a free noisy estimate of the number of records.
        """
        mechanism = make_noise_mechanism(
            self.noise, epsilon=epsilon, delta=delta, sensitivity=1.0, random_state=self._rng
        )
        noisy = mechanism.add_noise(counts.astype(float), budget=self._budget, label=label)
        if self._noisy_total is None:
            self._noisy_total = float(noisy.sum())
        return np.clip(noisy, 0.0, None)

    @staticmethod
    def _normalise(counts: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Turn non-negative counts into probabilities (uniform where all are 0)."""
        counts = np.asarray(counts, dtype=float)
        totals = counts.sum(axis=axis, keepdims=axis is not None)
        size = counts.size if axis is None else counts.shape[axis]
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.where(totals > 0, counts / np.where(totals > 0, totals, 1), 1.0 / size)
        return probs

    def _restore_dtypes(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Give synthetic columns the same types as the real ones where possible."""
        for col, dtype in self._schema["dtypes"].items():
            try:
                frame[col] = frame[col].astype(dtype)
            except (TypeError, ValueError):
                pass        # keep the decoded type if the cast is impossible
        return frame

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "not fitted"
        return f"{type(self).__name__}(epsilon={self.epsilon:g}, noise={self.noise!r}, {state})"
