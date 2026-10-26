"""Tests for PrivacyBudget (encapsulation of the epsilon/delta balance)."""

import pytest

from privsynth import PrivacyBudget, PrivacyBudgetExceededError


def test_spend_reduces_remaining_and_records_ledger():
    budget = PrivacyBudget(epsilon=1.0)
    budget.spend(0.4, label="first")
    budget.spend(0.1, label="second")
    assert budget.spent_epsilon == pytest.approx(0.5)
    assert budget.remaining_epsilon == pytest.approx(0.5)
    assert [e.label for e in budget.ledger] == ["first", "second"]


def test_overspending_raises_and_changes_nothing():
    budget = PrivacyBudget(epsilon=1.0)
    budget.spend(0.8)
    with pytest.raises(PrivacyBudgetExceededError):
        budget.spend(0.3)
    assert budget.spent_epsilon == pytest.approx(0.8)
    assert len(budget.ledger) == 1


def test_delta_is_tracked_separately():
    budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
    budget.spend(0.1, delta=1e-5)
    with pytest.raises(PrivacyBudgetExceededError):
        budget.spend(0.1, delta=1e-6)          # epsilon fits, delta does not


def test_equal_splits_use_the_whole_budget_despite_rounding():
    budget = PrivacyBudget(epsilon=1.0)
    for _ in range(3):
        budget.spend(1.0 / 3)                  # sums to 1.0000000000000002
    assert budget.is_exhausted


def test_split_divides_what_remains():
    budget = PrivacyBudget(epsilon=1.0, delta=1e-4)
    budget.spend(0.5, delta=0.5e-4)
    assert budget.split(5) == pytest.approx((0.1, 1e-5))


def test_ledger_and_balance_cannot_be_edited_from_outside():
    budget = PrivacyBudget(epsilon=1.0)
    budget.spend(0.2)
    with pytest.raises(AttributeError):
        budget.remaining_epsilon = 5.0
    assert isinstance(budget.ledger, tuple)
    with pytest.raises(Exception):
        budget.ledger[0].epsilon = 0.0        # entries are frozen dataclasses


@pytest.mark.parametrize("epsilon", [0, -1, float("inf"), "1", True])
def test_invalid_epsilon_rejected(epsilon):
    with pytest.raises(ValueError):
        PrivacyBudget(epsilon=epsilon)


@pytest.mark.parametrize("delta", [-0.1, 1.0])
def test_invalid_delta_rejected(delta):
    with pytest.raises(ValueError):
        PrivacyBudget(epsilon=1.0, delta=delta)


def test_summary_lists_each_charge():
    budget = PrivacyBudget(epsilon=2.0)
    budget.spend(0.5, label="count of >50K")
    assert "count of >50K" in budget.summary()
