"""The privacy settings a dashboard user chooses, with plain-language presets.

Most people will not know what "epsilon = 0.5" means, so the dashboard
offers named :data:`PRIVACY_LEVELS` and explains each one. The chosen
values are collected in a :class:`PrivacySettings` object, which checks
them and builds the :class:`~privsynth.pipeline.PrivSynthPipeline`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from privsynth.evaluation.suite import default_evaluators
from privsynth.generation import SYNTHESIZERS
from privsynth.pipeline import PrivSynthPipeline
from privsynth.privacy.mechanisms import NOISE_MECHANISMS


@dataclass(frozen=True)
class PrivacyLevel:
    """A named epsilon with an explanation for non-specialists."""

    name: str
    epsilon: float
    explanation: str


#: Presets from most private to most useful. The epsilons match the
#: values tested in the Week 3 experiments, so their effects are known.
PRIVACY_LEVELS: tuple[PrivacyLevel, ...] = (
    PrivacyLevel("Very strong privacy", 0.1,
                 "Heavy noise. Safe to share widely, but the data keeps only rough patterns."),
    PrivacyLevel("Strong privacy", 0.5,
                 "A common choice for public release; most simple patterns survive."),
    PrivacyLevel("Balanced", 1.0,
                 "A standard middle ground between protecting people and keeping detail."),
    PrivacyLevel("More detail", 2.0,
                 "Less noise; suited to sharing with trusted partners under an agreement."),
    PrivacyLevel("Most detail", 5.0,
                 "Light noise; keeps relationships best. Use only for internal or controlled sharing."),
)

GENERATOR_INFO = {
    "bayesian": ("Bayesian network (recommended)",
                 "Learns which columns depend on each other, so relationships such as "
                 "education and income survive."),
    "independent": ("Independent columns (baseline)",
                    "Copies each column's distribution separately. Fast, but every link "
                    "between columns is lost."),
}

NOISE_INFO = {
    "laplace": ("Laplace (pure ε-DP, recommended)",
                "The strictest guarantee, with no chance of failure. Did better in our experiments."),
    "gaussian": ("Gaussian ((ε, δ)-DP)",
                 "Allows a tiny probability δ that the guarantee fails; adds more noise for small tables."),
}


def level_for(epsilon: float) -> Optional[PrivacyLevel]:
    """The preset with this epsilon, or None for a custom value."""
    return next((lvl for lvl in PRIVACY_LEVELS if math.isclose(lvl.epsilon, epsilon)), None)


@dataclass(frozen=True)
class PrivacySettings:
    """Everything the user chooses before generating synthetic data.

    Attributes:
        generator: ``"bayesian"`` or ``"independent"``.
        noise: ``"laplace"`` or ``"gaussian"``.
        epsilon: Total privacy budget (smaller = more private).
        delta: Gaussian failure probability; None uses the default (1e-5).
        k: Target k for the k-anonymity check of the original data.
        holdout_fraction: Share of real records kept back for evaluation.
        n_synthetic: Synthetic records to generate; None = as many as trained on.
        seed: Random seed, so a run can be repeated exactly.

    Call :meth:`validate` (or :meth:`build_pipeline`, which calls it) to
    check the values.
    """

    generator: str = "bayesian"
    noise: str = "laplace"
    epsilon: float = 1.0
    delta: Optional[float] = None
    k: int = 5
    holdout_fraction: float = 0.2
    n_synthetic: Optional[int] = None
    seed: int = 0

    def validate(self) -> None:
        """Check every value, raising ValueError with a readable message."""
        if self.generator not in SYNTHESIZERS:
            raise ValueError(f"Unknown generator '{self.generator}'.")
        if self.noise not in NOISE_MECHANISMS:
            raise ValueError(f"Unknown noise type '{self.noise}'.")
        if not isinstance(self.epsilon, (int, float)) or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("The privacy budget ε must be a positive number.")
        if self.epsilon > 20:
            raise ValueError("ε above 20 gives almost no privacy; choose 20 or less.")
        if self.noise == "gaussian" and self.delta is not None and not 0 < self.delta < 1:
            raise ValueError("δ must be between 0 and 1 (typically 1e-5 or smaller).")
        if not isinstance(self.k, int) or self.k < 2:
            raise ValueError("k must be a whole number of at least 2.")
        if not 0 < self.holdout_fraction < 0.5:
            raise ValueError("The holdout share must be between 0 and 50%.")
        if self.n_synthetic is not None and (not isinstance(self.n_synthetic, int) or self.n_synthetic < 1):
            raise ValueError("The number of synthetic records must be a positive whole number.")

    def build_pipeline(self, bin_edges: Optional[Mapping[str, Sequence[float]]] = None) -> PrivSynthPipeline:
        """A pipeline with these settings that records evaluator errors instead of stopping.

        Raises:
            ValueError: If a setting is invalid.
        """
        self.validate()
        pipeline = PrivSynthPipeline.from_settings(
            self.generator,
            epsilon=self.epsilon,
            noise=self.noise,
            delta=self.delta,
            bin_edges=bin_edges,
            random_state=self.seed,
            evaluators=default_evaluators(bin_edges=bin_edges, random_state=self.seed),
            k=self.k,
            holdout_fraction=self.holdout_fraction,
            split_seed=self.seed,
            raise_errors=False,
        )
        return pipeline

    @property
    def level(self) -> Optional[PrivacyLevel]:
        return level_for(self.epsilon)

    def describe(self) -> str:
        """One line for reports, e.g. ``"Bayesian network, Laplace noise, ε = 1 (Balanced)"``."""
        level = f" ({self.level.name})" if self.level else ""
        gen = GENERATOR_INFO[self.generator][0].split(" (")[0]
        noise = self.noise.capitalize()
        delta = f", δ = {self.delta if self.delta is not None else 1e-5:g}" if self.noise == "gaussian" else ""
        return f"{gen}, {noise} noise, ε = {self.epsilon:g}{delta}{level}"

    def to_dict(self) -> dict:
        return asdict(self)
