"""Tests for the Laplace, Gaussian and Exponential mechanisms."""

import numpy as np
import pytest

from privsynth import (
    ExponentialMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyBudget,
    PrivacyBudgetExceededError,
)
from privsynth.privacy import NoiseMechanism, make_noise_mechanism


# --- Laplace ------------------------------------------------------------- #
def test_laplace_scale_is_sensitivity_over_epsilon():
    assert LaplaceMechanism(epsilon=0.5, sensitivity=2.0).scale == pytest.approx(4.0)


def test_laplace_noise_is_unbiased_with_expected_variance():
    mech = LaplaceMechanism(epsilon=1.0, random_state=0)
    noisy = mech.add_noise(np.full(200_000, 100.0))
    assert noisy.mean() == pytest.approx(100.0, abs=0.02)
    assert noisy.var() == pytest.approx(mech.variance, rel=0.03)   # 2 b^2 = 2


def test_smaller_epsilon_means_more_noise():
    assert LaplaceMechanism(0.1).std > LaplaceMechanism(1.0).std


def test_scalar_in_scalar_out():
    assert isinstance(LaplaceMechanism(1.0, random_state=1).add_noise(5), float)


def test_add_noise_charges_budget_before_release():
    budget = PrivacyBudget(epsilon=1.0)
    mech = LaplaceMechanism(epsilon=0.6, random_state=0)
    mech.add_noise([1, 2, 3], budget=budget, label="counts")
    assert budget.spent_epsilon == pytest.approx(0.6)
    with pytest.raises(PrivacyBudgetExceededError):
        mech.add_noise([1, 2, 3], budget=budget)
    assert budget.ledger[-1].label == "counts"


def test_nan_values_rejected():
    with pytest.raises(ValueError):
        LaplaceMechanism(1.0).add_noise([1.0, np.nan])


def test_seeded_mechanisms_are_reproducible():
    a = LaplaceMechanism(1.0, random_state=42).add_noise(np.zeros(5))
    b = LaplaceMechanism(1.0, random_state=42).add_noise(np.zeros(5))
    assert np.array_equal(a, b)


# --- Gaussian ------------------------------------------------------------ #
def test_classic_gaussian_matches_textbook_formula():
    mech = GaussianMechanism(epsilon=0.5, delta=1e-5, analytic=False)
    assert mech.scale == pytest.approx(np.sqrt(2 * np.log(1.25 / 1e-5)) / 0.5)


def test_classic_gaussian_refuses_epsilon_of_one_or_more():
    with pytest.raises(ValueError):
        GaussianMechanism(epsilon=1.0, analytic=False)


def test_gaussian_needs_positive_delta():
    with pytest.raises(ValueError):
        GaussianMechanism(epsilon=0.5, delta=0.0)


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 0.9])
def test_analytic_gaussian_is_tighter_than_classic(epsilon):
    analytic = GaussianMechanism(epsilon, delta=1e-5).scale
    classic = GaussianMechanism.classic_sigma(epsilon, 1e-5)
    assert analytic < classic


@pytest.mark.parametrize("epsilon", [0.1, 1.0, 5.0])
def test_analytic_sigma_achieves_target_delta(epsilon):
    mech = GaussianMechanism(epsilon, delta=1e-6)
    achieved = GaussianMechanism.privacy_loss_delta(mech.scale, epsilon)
    assert achieved == pytest.approx(1e-6, rel=1e-4)


def test_gaussian_charges_delta():
    budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
    GaussianMechanism(0.5, delta=1e-5, random_state=0).add_noise([10.0], budget=budget)
    assert budget.spent_delta == pytest.approx(1e-5)


# --- Polymorphism ------------------------------------------------------- #
def test_noise_mechanisms_are_interchangeable():
    """Callers use add_noise without knowing which subclass they hold."""
    mechanisms: list[NoiseMechanism] = [
        LaplaceMechanism(1.0, random_state=0),
        GaussianMechanism(1.0, delta=1e-5, random_state=0),
    ]
    counts = np.array([5.0, 10.0, 15.0])
    for mech in mechanisms:
        noisy = mech.add_noise(counts)
        assert noisy.shape == counts.shape
        assert mech.variance > 0


def test_factory_builds_by_name():
    assert isinstance(make_noise_mechanism("Laplace", 1.0), LaplaceMechanism)
    assert isinstance(make_noise_mechanism("gaussian", 1.0, delta=1e-5), GaussianMechanism)
    with pytest.raises(ValueError):
        make_noise_mechanism("uniform", 1.0)


def test_noise_mechanism_is_abstract():
    with pytest.raises(TypeError):
        NoiseMechanism(1.0)


# --- Exponential --------------------------------------------------------- #
def test_exponential_equal_scores_give_equal_chances():
    probs = ExponentialMechanism(1.0).probabilities([3.0, 3.0, 3.0, 3.0])
    assert np.allclose(probs, 0.25)


def test_exponential_prefers_high_scores():
    probs = ExponentialMechanism(1.0).probabilities([0.0, 1.0, 2.0])
    assert probs[2] > probs[1] > probs[0]
    # ratio between neighbours is exp(eps * 1 / 2)
    assert probs[1] / probs[0] == pytest.approx(np.exp(0.5))


def test_exponential_with_large_epsilon_picks_the_best():
    mech = ExponentialMechanism(1000.0, random_state=0)
    assert all(mech.select([1.0, 5.0, 2.0]) == 1 for _ in range(20))


def test_exponential_handles_huge_scores_without_overflow():
    probs = ExponentialMechanism(10.0).probabilities([1e6, 1e6 - 1])
    assert np.isfinite(probs).all() and probs.sum() == pytest.approx(1.0)


def test_exponential_select_charges_budget():
    budget = PrivacyBudget(epsilon=0.5)
    ExponentialMechanism(0.5, random_state=0).select([1, 2], budget=budget)
    assert budget.is_exhausted
