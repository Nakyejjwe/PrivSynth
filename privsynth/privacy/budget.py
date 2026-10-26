"""The privacy budget: how much epsilon (and delta) may still be spent.

Every time a differentially private algorithm looks at the real data it
"spends" some of the privacy budget. By **sequential composition**
[Dwork & Roth, 2014, Thm 3.16], running mechanisms with costs
``(e1, d1), (e2, d2), ...`` on the same data is ``(sum e, sum d)``-DP.
So the total guarantee of a whole synthesizer is simply the sum of what
its individual steps spent.

:class:`PrivacyBudget` enforces that sum. Its balance is private: the only
way to change it is :meth:`PrivacyBudget.spend`, which refuses any request
that would overspend and records every successful charge in a read-only
ledger. This is the encapsulation example from the proposal.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from privsynth.exceptions import PrivacyBudgetExceededError

# Splitting a budget into equal parts (e.g. 1.0 / 3 three times) produces
# floating-point sums such as 1.0000000000000002. This tolerance stops such
# rounding noise from being treated as overspending.
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BudgetEntry:
    """One charge against a :class:`PrivacyBudget` (immutable).

    Attributes:
        label: What the budget was spent on, e.g. ``"marginal: age"``.
        epsilon: Epsilon spent by this step.
        delta: Delta spent by this step (0 for pure-DP mechanisms).
    """

    label: str
    epsilon: float
    delta: float = 0.0


class PrivacyBudget:
    """Track and limit the privacy loss of a sequence of DP mechanisms.

    Args:
        epsilon: Total epsilon allowed. Smaller means more private.
        delta: Total delta allowed (probability that the epsilon guarantee
            fails). Only Gaussian noise needs it; use ``0`` for pure DP.

    Raises:
        ValueError: If ``epsilon`` is not positive or ``delta`` is not in [0, 1).

    Example:
        >>> budget = PrivacyBudget(epsilon=1.0)
        >>> budget.spend(0.4, label="count of >50K")
        >>> budget.remaining_epsilon
        0.6
    """

    def __init__(self, epsilon: float, delta: float = 0.0) -> None:
        _check_epsilon(epsilon)
        _check_delta(delta)
        self._total_epsilon = float(epsilon)
        self._total_delta = float(delta)
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._ledger: list[BudgetEntry] = []

    # ------------------------------------------------------------------ #
    # Read-only view of the balance
    # ------------------------------------------------------------------ #
    @property
    def total_epsilon(self) -> float:
        """Epsilon available when the budget was created."""
        return self._total_epsilon

    @property
    def total_delta(self) -> float:
        """Delta available when the budget was created."""
        return self._total_delta

    @property
    def spent_epsilon(self) -> float:
        """Epsilon used so far."""
        return self._spent_epsilon

    @property
    def spent_delta(self) -> float:
        """Delta used so far."""
        return self._spent_delta

    @property
    def remaining_epsilon(self) -> float:
        """Epsilon still available (never negative)."""
        return max(0.0, self._total_epsilon - self._spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Delta still available (never negative)."""
        return max(0.0, self._total_delta - self._spent_delta)

    @property
    def ledger(self) -> tuple[BudgetEntry, ...]:
        """Every charge so far, oldest first. A tuple, so it cannot be edited."""
        return tuple(self._ledger)

    @property
    def is_exhausted(self) -> bool:
        """True when (almost) no epsilon remains."""
        return self.remaining_epsilon <= _TOLERANCE

    # ------------------------------------------------------------------ #
    # The only way to change the balance
    # ------------------------------------------------------------------ #
    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Return True if ``(epsilon, delta)`` fits in what remains."""
        return (
            self._spent_epsilon + epsilon <= self._total_epsilon + _TOLERANCE
            and self._spent_delta + delta <= self._total_delta + _TOLERANCE
        )

    def spend(self, epsilon: float, delta: float = 0.0, label: str = "unnamed query") -> None:
        """Charge ``(epsilon, delta)`` to the budget.

        The charge is all-or-nothing: if it does not fit, nothing is spent
        and the ledger is unchanged.

        Raises:
            ValueError: If ``epsilon`` is not positive or ``delta`` is invalid.
            PrivacyBudgetExceededError: If the charge would overspend.
        """
        _check_epsilon(epsilon)
        _check_delta(delta)
        if not self.can_spend(epsilon, delta):
            raise PrivacyBudgetExceededError(
                f"Cannot spend epsilon={epsilon:.4g}, delta={delta:.3g} on '{label}': "
                f"only epsilon={self.remaining_epsilon:.4g}, "
                f"delta={self.remaining_delta:.3g} remain."
            )
        self._spent_epsilon += float(epsilon)
        self._spent_delta += float(delta)
        self._ledger.append(BudgetEntry(label=label, epsilon=float(epsilon), delta=float(delta)))

    def split(self, parts: int) -> tuple[float, float]:
        """Return the ``(epsilon, delta)`` share for each of ``parts`` equal steps.

        Uses what *remains*, so it can be called part-way through a run.
        """
        if not isinstance(parts, int) or parts < 1:
            raise ValueError(f"parts must be a positive integer, got {parts!r}.")
        return self.remaining_epsilon / parts, self.remaining_delta / parts

    def summary(self) -> str:
        """A short multi-line description of the ledger (for reports)."""
        lines = [
            f"Privacy budget: spent epsilon={self._spent_epsilon:.4f} of {self._total_epsilon:.4f}"
            + (f", delta={self._spent_delta:.2e} of {self._total_delta:.2e}" if self._total_delta else "")
        ]
        for entry in self._ledger:
            extra = f", delta={entry.delta:.2e}" if entry.delta else ""
            lines.append(f"  - {entry.label}: epsilon={entry.epsilon:.4f}{extra}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PrivacyBudget(epsilon={self._total_epsilon}, delta={self._total_delta}, "
            f"spent_epsilon={self._spent_epsilon:.4g}, charges={len(self._ledger)})"
        )


def _check_epsilon(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be a positive finite number, got {epsilon!r}.")


def _check_delta(delta: float) -> None:
    if isinstance(delta, bool) or not isinstance(delta, numbers.Real) or not 0 <= delta < 1:
        raise ValueError(f"delta must be in [0, 1), got {delta!r}.")
