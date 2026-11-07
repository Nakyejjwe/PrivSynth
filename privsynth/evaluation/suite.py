"""Run several evaluators together and collect their results.

:class:`EvaluationSuite` holds a list of :class:`Evaluator` objects
(composition) and runs them all on the same data. Because every evaluator
shares the ``evaluate`` interface, the suite never needs to know which
ones it holds. The results come back as one :class:`EvaluationReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.evaluation.base import EvaluationResult, Evaluator
from privsynth.evaluation.disclosure import AttributeInferenceRisk, DistanceToClosestRecord
from privsynth.evaluation.utility import (
    AssociationSimilarity,
    MachineLearningUtility,
    MarginalSimilarity,
)
from privsynth.exceptions import PrivSynthError


@dataclass
class EvaluationReport:
    """Results of an :class:`EvaluationSuite` run.

    Attributes:
        results: One :class:`EvaluationResult` per evaluator that ran.
        skipped: Evaluator name -> reason, for evaluators that could not run.
    """

    results: list[EvaluationResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> EvaluationResult:
        """Look up a result by its key (``"ml"``) or display name."""
        for result in self.results:
            if key in (result.key, result.evaluator):
                return result
        raise KeyError(f"No result for {key!r}; available: {[r.key for r in self.results]}")

    def __contains__(self, key: str) -> bool:
        return any(key in (r.key, r.evaluator) for r in self.results)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def utility(self) -> list[EvaluationResult]:
        return [r for r in self.results if r.kind == "utility"]

    @property
    def risk(self) -> list[EvaluationResult]:
        return [r for r in self.results if r.kind == "risk"]

    def flat(self) -> dict[str, float]:
        """Every metric in one dict, keyed ``"evaluator_key.metric"``."""
        out: dict[str, float] = {}
        for result in self.results:
            out.update(result.to_dict(prefix=True))
        return out

    def to_frame(self) -> pd.DataFrame:
        """One row per metric: evaluator, kind, metric, value, direction."""
        rows = [
            {
                "evaluator": r.evaluator,
                "kind": r.kind,
                "metric": name,
                "value": value,
                "lower_is_better": name in r.lower_is_better,
            }
            for r in self.results
            for name, value in r.metrics.items()
        ]
        return pd.DataFrame(rows, columns=["evaluator", "kind", "metric", "value", "lower_is_better"])

    def __str__(self) -> str:
        parts = [str(r) for r in self.results]
        parts.extend(f"{name}: skipped ({reason})" for name, reason in self.skipped.items())
        return "\n\n".join(parts) if parts else "(no evaluations)"


class EvaluationSuite:
    """A collection of evaluators run together.

    Args:
        evaluators: The evaluators to run, in order.
        raise_errors: If True (default), an evaluator's error stops the run.
            If False, it is recorded in :attr:`EvaluationReport.skipped` and
            the other evaluators still run, which suits the dashboard.
    """

    def __init__(self, evaluators: Sequence[Evaluator], raise_errors: bool = True) -> None:
        evaluators = list(evaluators)
        bad = [type(e).__name__ for e in evaluators if not isinstance(e, Evaluator)]
        if bad:
            raise TypeError(f"Not evaluators: {bad}")
        if not evaluators:
            raise ValueError("An EvaluationSuite needs at least one evaluator.")
        self._evaluators = evaluators
        self.raise_errors = raise_errors

    @property
    def evaluators(self) -> list[Evaluator]:
        return list(self._evaluators)

    def run(self, real: Dataset, synthetic: Dataset, holdout: Optional[Dataset] = None) -> EvaluationReport:
        """Run every evaluator and return their results.

        Evaluators that need a holdout are skipped (not failed) when
        ``holdout`` is None, whatever ``raise_errors`` says.
        """
        report = EvaluationReport()
        for evaluator in self._evaluators:
            if evaluator.requires_holdout and holdout is None:
                report.skipped[evaluator.name] = "needs a holdout set"
                continue
            try:
                report.results.append(evaluator.evaluate(real, synthetic, holdout))
            except PrivSynthError as err:
                if self.raise_errors:
                    raise
                report.skipped[evaluator.name] = str(err)
        return report

    def __repr__(self) -> str:
        names = ", ".join(type(e).__name__ for e in self._evaluators)
        return f"EvaluationSuite([{names}])"


def default_evaluators(
    bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
    n_bins: int = 16,
    target: Optional[str] = None,
    max_records: int = 2000,
    random_state: int = 0,
) -> list[Evaluator]:
    """The standard set of three utility and two risk evaluators."""
    return [
        MarginalSimilarity(n_bins=n_bins, bin_edges=bin_edges),
        AssociationSimilarity(n_bins=n_bins, bin_edges=bin_edges),
        MachineLearningUtility(target=target, random_state=random_state),
        DistanceToClosestRecord(max_records=max_records, random_state=random_state),
        AttributeInferenceRisk(max_records=max_records, n_bins=n_bins, bin_edges=bin_edges,
                               random_state=random_state),
    ]
