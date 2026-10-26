"""Differential privacy mechanisms.

A mechanism is *epsilon-differentially private* if adding or removing any
one person's record changes the probability of every output by at most a
factor ``e^epsilon`` [Dwork & Roth, 2014]. ``(epsilon, delta)``-DP allows
that guarantee to fail with small probability ``delta``.

Class hierarchy (the inheritance / polymorphism part of the design)::

    PrivacyMechanism (abstract)      epsilon, delta, charge(budget)
    |-- NoiseMechanism (abstract)    add_noise(values) using scale + _sample_noise
    |   |-- LaplaceMechanism         pure epsilon-DP, noise scale = L1 sensitivity / epsilon
    |   `-- GaussianMechanism        (epsilon, delta)-DP, noise scale from L2 sensitivity
    `-- ExponentialMechanism         privately *chooses* the best of several options

The synthesizers only ever call ``add_noise``; they do not know whether
Laplace or Gaussian noise is being added. Swapping one for the other is a
one-word change (see :func:`make_noise_mechanism`).

*Sensitivity* is how much one person can change the true answer. For a
histogram of counts, adding or removing one record changes exactly one
cell by 1, so both the L1 and the L2 sensitivity are 1.
"""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
from scipy import optimize, stats

from privsynth.privacy.budget import PrivacyBudget, _check_delta, _check_epsilon

RandomState = Union[None, int, np.random.Generator]


def _as_rng(random_state: RandomState) -> np.random.Generator:
    """Turn ``None``, a seed or a Generator into a numpy Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class PrivacyMechanism(ABC):
    """Abstract base class for every differentially private mechanism.

    Args:
        epsilon: Privacy loss of one use of the mechanism.
        delta: Failure probability (0 for pure DP).
        random_state: Seed or numpy Generator, for reproducible experiments.
    """

    #: Short name used in reports and the dashboard; set by each subclass.
    name: str = "mechanism"

    def __init__(self, epsilon: float, delta: float = 0.0, random_state: RandomState = None) -> None:
        _check_epsilon(epsilon)
        _check_delta(delta)
        self._epsilon = float(epsilon)
        self._delta = float(delta)
        self._rng = _as_rng(random_state)

    @property
    def epsilon(self) -> float:
        """Epsilon spent each time the mechanism is used (read-only)."""
        return self._epsilon

    @property
    def delta(self) -> float:
        """Delta spent each time the mechanism is used (read-only)."""
        return self._delta

    def charge(self, budget: Optional[PrivacyBudget], label: Optional[str] = None) -> None:
        """Spend this mechanism's cost from ``budget`` (no-op if ``budget`` is None).

        Called *before* touching the data, so an over-budget request fails
        without releasing anything.
        """
        if budget is not None:
            budget.spend(self._epsilon, self._delta, label=label or self.name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(epsilon={self._epsilon:.4g}, delta={self._delta:.3g})"


class NoiseMechanism(PrivacyMechanism):
    """Abstract base for mechanisms that release ``true value + random noise``.

    Subclasses decide the noise *distribution* (:meth:`_sample_noise`) and
    its *size* (:attr:`scale`); this class supplies the shared workflow in
    :meth:`add_noise` (a template method).

    Args:
        epsilon: Privacy loss per release.
        sensitivity: Most one record can change the true answer (L1 for
            Laplace, L2 for Gaussian). 1 for counts and histograms.
        delta: Failure probability.
        random_state: Seed or numpy Generator.
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
        delta: float = 0.0,
        random_state: RandomState = None,
    ) -> None:
        super().__init__(epsilon, delta, random_state)
        if isinstance(sensitivity, bool) or not isinstance(sensitivity, numbers.Real) or not sensitivity > 0:
            raise ValueError(f"sensitivity must be a positive number, got {sensitivity!r}.")
        self._sensitivity = float(sensitivity)

    @property
    def sensitivity(self) -> float:
        """How much one record can change the true answer."""
        return self._sensitivity

    @property
    @abstractmethod
    def scale(self) -> float:
        """The noise scale parameter (Laplace ``b`` or Gaussian ``sigma``)."""

    @property
    @abstractmethod
    def variance(self) -> float:
        """Variance of the added noise: the expected squared error per value."""

    @property
    def std(self) -> float:
        """Standard deviation of the added noise, in the data's own units."""
        return math.sqrt(self.variance)

    @abstractmethod
    def _sample_noise(self, size: tuple[int, ...]) -> np.ndarray:
        """Draw noise of the given shape from the mechanism's distribution."""

    def add_noise(
        self,
        values: Union[float, Sequence[float], np.ndarray],
        budget: Optional[PrivacyBudget] = None,
        label: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """Return ``values`` plus noise, charging ``budget`` first if given.

        Noise is added to every element independently. This is one release:
        it costs ``(epsilon, delta)`` once **provided** ``sensitivity`` was set
        for the whole vector (as it is for a histogram).

        Args:
            values: A number or array of true answers.
            budget: Budget to charge. If it cannot pay, nothing is released.
            label: Description recorded in the budget ledger.

        Returns:
            A float for scalar input, otherwise a float array of the same shape.

        Raises:
            PrivacyBudgetExceededError: If ``budget`` has too little left.
            ValueError: If ``values`` contains NaN or infinity.
        """
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("Cannot add noise to NaN or infinite values.")
        self.charge(budget, label)
        noisy = array + self._sample_noise(array.shape)
        return float(noisy) if noisy.ndim == 0 else noisy


class LaplaceMechanism(NoiseMechanism):
    """Pure epsilon-DP via Laplace noise with scale ``b = sensitivity / epsilon``.

    [Dwork & Roth, 2014, Thm 3.6]. ``sensitivity`` is the L1 sensitivity.
    Halving epsilon doubles the typical error.
    """

    name = "laplace"

    def __init__(self, epsilon: float, sensitivity: float = 1.0, random_state: RandomState = None) -> None:
        super().__init__(epsilon, sensitivity=sensitivity, delta=0.0, random_state=random_state)

    @property
    def scale(self) -> float:
        return self._sensitivity / self._epsilon

    @property
    def variance(self) -> float:
        return 2.0 * self.scale**2

    def _sample_noise(self, size: tuple[int, ...]) -> np.ndarray:
        return self._rng.laplace(0.0, self.scale, size=size)


class GaussianMechanism(NoiseMechanism):
    """(epsilon, delta)-DP via Gaussian noise with standard deviation ``sigma``.

    ``sensitivity`` is the **L2** sensitivity. Two calibrations are offered:

    * **classic** [Dwork & Roth, 2014, Thm A.1]:
      ``sigma = sensitivity * sqrt(2 ln(1.25 / delta)) / epsilon``.
      Only proven for ``epsilon < 1``.
    * **analytic** (default) [Balle & Wang, 2018, Thm 8]: the smallest sigma
      that satisfies the exact (epsilon, delta) condition, found
      numerically. Valid for any epsilon and never noisier than classic.

    Args:
        epsilon: Privacy loss per release.
        delta: Failure probability; must be greater than 0 (typically 1e-5).
        sensitivity: L2 sensitivity of the query.
        analytic: Use the analytic calibration (True) or the classic one.
        random_state: Seed or numpy Generator.

    Raises:
        ValueError: If ``delta`` is 0, or classic calibration is used with
            ``epsilon >= 1``.
    """

    name = "gaussian"

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        analytic: bool = True,
        random_state: RandomState = None,
    ) -> None:
        super().__init__(epsilon, sensitivity=sensitivity, delta=delta, random_state=random_state)
        if self._delta <= 0:
            raise ValueError("The Gaussian mechanism needs delta > 0 (try 1e-5).")
        if not analytic and self._epsilon >= 1:
            raise ValueError(
                "The classic Gaussian calibration is only valid for epsilon < 1; "
                "use analytic=True for larger epsilon."
            )
        self.analytic = analytic
        self._sigma = (
            self.analytic_sigma(self._epsilon, self._delta, self._sensitivity)
            if analytic
            else self.classic_sigma(self._epsilon, self._delta, self._sensitivity)
        )

    @staticmethod
    def classic_sigma(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
        """Noise level from the textbook formula (valid for epsilon < 1)."""
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    @staticmethod
    def privacy_loss_delta(sigma: float, epsilon: float, sensitivity: float = 1.0) -> float:
        """The exact delta achieved by noise ``sigma`` at a given epsilon.

        ``delta(sigma) = Phi(D/(2s) - e*s/D) - e^e * Phi(-D/(2s) - e*s/D)``
        where ``D`` is the sensitivity [Balle & Wang, 2018, Thm 8]. Computed
        in log space so large epsilon does not overflow.
        """
        a = sensitivity / (2.0 * sigma)
        b = epsilon * sigma / sensitivity
        first = stats.norm.cdf(a - b)
        second = math.exp(epsilon + stats.norm.logcdf(-a - b))
        return float(first - second)

    @classmethod
    def analytic_sigma(cls, epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
        """Smallest sigma whose exact delta is at most ``delta`` (bisection)."""
        def gap(sigma: float) -> float:
            return cls.privacy_loss_delta(sigma, epsilon, sensitivity) - delta

        low = 1e-6 * sensitivity
        high = sensitivity
        while gap(high) > 0:      # widen until the bracket contains the root
            high *= 2.0
        return float(optimize.brentq(gap, low, high, xtol=1e-12, rtol=1e-10))

    @property
    def scale(self) -> float:
        return self._sigma

    @property
    def variance(self) -> float:
        return self._sigma**2

    def _sample_noise(self, size: tuple[int, ...]) -> np.ndarray:
        return self._rng.normal(0.0, self._sigma, size=size)

    def __repr__(self) -> str:
        kind = "analytic" if self.analytic else "classic"
        return (
            f"GaussianMechanism(epsilon={self._epsilon:.4g}, delta={self._delta:.3g}, "
            f"sigma={self._sigma:.4g}, {kind})"
        )


class ExponentialMechanism(PrivacyMechanism):
    """Privately pick one option from a list, favouring high scores.

    Option ``i`` is chosen with probability proportional to
    ``exp(epsilon * score_i / (2 * sensitivity))`` [Dwork & Roth, 2014,
    Thm 3.10]. Used by the Bayesian network synthesizer to choose which
    columns depend on which, without revealing the exact dependencies.

    Args:
        epsilon: Privacy loss of one selection.
        sensitivity: Most one record can change any option's score.
        random_state: Seed or numpy Generator.
    """

    name = "exponential"

    def __init__(self, epsilon: float, sensitivity: float = 1.0, random_state: RandomState = None) -> None:
        super().__init__(epsilon, delta=0.0, random_state=random_state)
        if isinstance(sensitivity, bool) or not isinstance(sensitivity, numbers.Real) or not sensitivity > 0:
            raise ValueError(f"sensitivity must be a positive number, got {sensitivity!r}.")
        self._sensitivity = float(sensitivity)

    @property
    def sensitivity(self) -> float:
        """Most one record can change any option's score."""
        return self._sensitivity

    def probabilities(self, scores: Sequence[float]) -> np.ndarray:
        """Selection probability for each score (sums to 1)."""
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("scores must be a non-empty 1-D sequence.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must be finite.")
        logits = self._epsilon * scores / (2.0 * self._sensitivity)
        logits -= logits.max()          # subtract the max so exp() cannot overflow
        weights = np.exp(logits)
        return weights / weights.sum()

    def select(
        self,
        scores: Sequence[float],
        budget: Optional[PrivacyBudget] = None,
        label: Optional[str] = None,
    ) -> int:
        """Return the index of the chosen option, charging ``budget`` first."""
        probs = self.probabilities(scores)
        self.charge(budget, label)
        return int(self._rng.choice(len(probs), p=probs))


#: Noise mechanisms the synthesizers (and later the dashboard) can choose by name.
NOISE_MECHANISMS: dict[str, type[NoiseMechanism]] = {
    LaplaceMechanism.name: LaplaceMechanism,
    GaussianMechanism.name: GaussianMechanism,
}


def make_noise_mechanism(
    kind: str,
    epsilon: float,
    delta: float = 0.0,
    sensitivity: float = 1.0,
    random_state: RandomState = None,
) -> NoiseMechanism:
    """Build a noise mechanism by name (``"laplace"`` or ``"gaussian"``).

    A small factory so callers can switch mechanism with a string, which is
    what a dashboard drop-down provides.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    key = kind.lower().strip()
    if key == LaplaceMechanism.name:
        return LaplaceMechanism(epsilon, sensitivity=sensitivity, random_state=random_state)
    if key == GaussianMechanism.name:
        return GaussianMechanism(epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
    raise ValueError(f"Unknown noise mechanism {kind!r}; choose from {sorted(NOISE_MECHANISMS)}.")
