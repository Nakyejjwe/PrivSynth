"""The :class:`PrivSynthPipeline`: one object that runs the whole workflow.

The proposal describes four modules connected by a pipeline object. This
is it. A single :meth:`PrivSynthPipeline.run` call does::

    real data --split--> train + holdout
    train     --KAnonymityChecker--> re-identification risk of the original
    train     --Synthesizer.fit/sample--> synthetic data   (spends epsilon)
    synthetic --EvaluationSuite--> utility and remaining privacy risk

The pipeline *has* a synthesizer, a k-anonymity checker and an evaluation
suite (composition) and only talks to them through their shared
interfaces, so any generator or list of evaluators can be plugged in.
The Week 4 dashboard drives everything through this class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from privsynth.data.dataset import Dataset
from privsynth.evaluation.base import Evaluator
from privsynth.evaluation.suite import EvaluationReport, EvaluationSuite, default_evaluators
from privsynth.exceptions import SchemaError
from privsynth.generation import SYNTHESIZERS
from privsynth.generation.base import Synthesizer
from privsynth.privacy.budget import PrivacyBudget
from privsynth.privacy.risk import KAnonymityChecker, RiskReport


@dataclass
class PipelineResult:
    """Everything one pipeline run produced.

    Attributes:
        train: Real records the synthesizer learned from.
        holdout: Real records kept back for evaluation (None if not split).
        synthetic: The generated records.
        original_risk: k-anonymity report for the real training data.
        evaluation: Utility and privacy-risk results for the synthetic data.
        synthesizer: The fitted synthesizer (its budget ledger, network, ...).
        timings: Seconds spent on each stage.
    """

    train: Dataset
    holdout: Optional[Dataset]
    synthetic: Dataset
    original_risk: RiskReport
    evaluation: EvaluationReport
    synthesizer: Synthesizer
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def budget(self) -> PrivacyBudget:
        """The privacy budget spent by the synthesizer."""
        return self.synthesizer.budget

    def summary(self) -> dict:
        """Settings and every headline number in one flat dict (one table row)."""
        row = {
            "generator": self.synthesizer.name,
            "noise": self.synthesizer.noise,
            "epsilon": self.synthesizer.epsilon,
            "delta": self.synthesizer.delta,
            "epsilon_spent": self.budget.spent_epsilon,
            "n_train": self.train.n_rows,
            "n_holdout": self.holdout.n_rows if self.holdout is not None else 0,
            "n_synthetic": self.synthetic.n_rows,
            "original.k_achieved": self.original_risk.k_achieved,
            "original.pct_unique": self.original_risk.pct_unique,
        }
        row.update(self.evaluation.flat())
        row.update({f"seconds.{stage}": secs for stage, secs in self.timings.items()})
        return row

    def __str__(self) -> str:
        holdout = f"{self.holdout.n_rows:,}" if self.holdout is not None else "none"
        header = (
            f"PrivSynth run: {self.synthesizer.name}, {self.synthesizer.noise} noise, "
            f"epsilon={self.synthesizer.epsilon:g} (spent {self.budget.spent_epsilon:.4f})\n"
            f"  train {self.train.n_rows:,} | holdout {holdout} | synthetic {self.synthetic.n_rows:,}\n"
        )
        return f"{header}\nOriginal data:\n{self.original_risk}\n\n{self.evaluation}"


class PrivSynthPipeline:
    """Split, check risk, generate and evaluate in one call.

    Args:
        synthesizer: Any :class:`Synthesizer` (it is refitted on each run).
        evaluators: Evaluators to run; defaults to :func:`default_evaluators`
            using the synthesizer's own bin edges.
        k: Target k for the k-anonymity check of the original data.
        holdout_fraction: Share of real records kept back for evaluation.
            ``0`` trains on everything and skips holdout-based evaluators.
        split_seed: Seed for the train/holdout split.
        raise_errors: Passed to :class:`EvaluationSuite`.

    Example:
        >>> pipeline = PrivSynthPipeline.from_settings("bayesian", epsilon=1.0, random_state=0)
        >>> result = pipeline.run(dataset)
        >>> print(result)
        >>> result.synthetic.data.to_csv("synthetic.csv", index=False)
    """

    def __init__(
        self,
        synthesizer: Synthesizer,
        evaluators: Optional[Sequence[Evaluator]] = None,
        k: int = 5,
        holdout_fraction: float = 0.2,
        split_seed: Optional[int] = 0,
        raise_errors: bool = True,
    ) -> None:
        if not isinstance(synthesizer, Synthesizer):
            raise TypeError(f"synthesizer must be a Synthesizer, got {type(synthesizer).__name__}.")
        if isinstance(holdout_fraction, bool) or not 0 <= holdout_fraction < 1:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction!r}.")
        self.synthesizer = synthesizer
        if evaluators is None:
            evaluators = default_evaluators(bin_edges=synthesizer.encoder.bin_edges)
        self.suite = EvaluationSuite(evaluators, raise_errors=raise_errors)
        self.checker = KAnonymityChecker(k)
        self.holdout_fraction = float(holdout_fraction)
        self.split_seed = split_seed

    @classmethod
    def from_settings(
        cls,
        generator: str = "bayesian",
        epsilon: float = 1.0,
        noise: str = "laplace",
        delta: Optional[float] = None,
        bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
        random_state: Optional[int] = None,
        **pipeline_options,
    ) -> "PrivSynthPipeline":
        """Build a pipeline from plain settings (what dashboard widgets provide).

        Args:
            generator: A key of :data:`privsynth.generation.SYNTHESIZERS`
                (``"independent"`` or ``"bayesian"``).
            epsilon, noise, delta, bin_edges, random_state: Synthesizer settings.
            **pipeline_options: Passed to :class:`PrivSynthPipeline`.

        Raises:
            ValueError: If ``generator`` is unknown.
        """
        key = generator.lower().strip()
        if key not in SYNTHESIZERS:
            raise ValueError(f"Unknown generator {generator!r}; choose from {sorted(SYNTHESIZERS)}.")
        synthesizer = SYNTHESIZERS[key](
            epsilon=epsilon, noise=noise, delta=delta, bin_edges=bin_edges, random_state=random_state
        )
        return cls(synthesizer, **pipeline_options)

    # ------------------------------------------------------------------ #
    def run(self, dataset: Dataset, n: Optional[int] = None) -> PipelineResult:
        """Run the whole workflow on ``dataset``.

        Args:
            dataset: The real data, with quasi-identifiers set.
            n: Synthetic records to generate; defaults to the training size.

        Raises:
            SchemaError: If ``dataset`` is not a Dataset or has no quasi-identifiers.
        """
        if not isinstance(dataset, Dataset):
            raise SchemaError(f"run() expects a Dataset, got {type(dataset).__name__}.")
        if self.holdout_fraction > 0:
            train, holdout = dataset.split(self.holdout_fraction, random_state=self.split_seed)
        else:
            train, holdout = dataset, None
        return self.run_on_split(train, holdout, n=n)

    def run_on_split(self, train: Dataset, holdout: Optional[Dataset], n: Optional[int] = None) -> PipelineResult:
        """Run on a train/holdout split made elsewhere (used by experiments).

        Reusing one split for many runs makes their results directly comparable.
        """
        if not isinstance(train, Dataset):
            raise SchemaError(f"train must be a Dataset, got {type(train).__name__}.")
        timings: dict[str, float] = {}

        start = time.perf_counter()
        original_risk = self.checker.evaluate(train)
        timings["risk_check"] = time.perf_counter() - start

        start = time.perf_counter()
        self.synthesizer.fit(train)
        synthetic = self.synthesizer.sample(n if n is not None else train.n_rows)
        timings["generate"] = time.perf_counter() - start

        start = time.perf_counter()
        evaluation = self.suite.run(train, synthetic, holdout)
        timings["evaluate"] = time.perf_counter() - start

        return PipelineResult(
            train=train,
            holdout=holdout,
            synthetic=synthetic,
            original_risk=original_risk,
            evaluation=evaluation,
            synthesizer=self.synthesizer,
            timings=timings,
        )

    def __repr__(self) -> str:
        return (f"PrivSynthPipeline({self.synthesizer!r}, {self.suite!r}, k={self.checker.k}, "
                f"holdout_fraction={self.holdout_fraction:g})")
