"""Tests for the encoder and the two synthetic data generators."""

import numpy as np
import pandas as pd
import pytest

from privsynth import (
    BayesianNetworkSynthesizer,
    Dataset,
    IndependentMarginalSynthesizer,
    NotFittedError,
    SchemaError,
    Synthesizer,
)
from privsynth.generation import SYNTHESIZERS, TableEncoder
from privsynth.generation.bayesian import mutual_information

GENERATORS = [IndependentMarginalSynthesizer, BayesianNetworkSynthesizer]


@pytest.fixture
def linked_dataset() -> Dataset:
    """2,000 records where ``degree`` completely determines ``pay``.

    Also an independent ``region`` column and a numeric ``age``.
    """
    rng = np.random.default_rng(7)
    n = 2000
    degree = rng.choice(["none", "diploma", "degree"], size=n, p=[0.5, 0.3, 0.2])
    pay = np.select([degree == "none", degree == "diploma"], ["low", "mid"], "high")
    frame = pd.DataFrame({
        "age": rng.integers(18, 70, size=n),
        "degree": degree,
        "pay": pay,
        "region": rng.choice(["Central", "Eastern", "Northern", "Western"], size=n),
    })
    return Dataset(frame, name="linked", quasi_identifiers=["age", "region"], sensitive_attribute="pay")


# --- Encoder ------------------------------------------------------------- #
def test_encoder_round_trips_categories(small_dataset):
    enc = TableEncoder().fit(small_dataset)
    decoded = enc.decode(enc.encode(small_dataset), rng=np.random.default_rng(0))
    for col in ["sex", "city", "income"]:
        assert list(decoded[col]) == list(small_dataset.data[col])


def test_small_integer_range_is_reproduced_exactly(small_dataset):
    enc = TableEncoder().fit(small_dataset)       # age is 30..55: one bin per year
    decoded = enc.decode(enc.encode(small_dataset), rng=np.random.default_rng(0))
    assert list(decoded["age"]) == list(small_dataset.data["age"])


def test_custom_bin_edges_keep_values_inside_their_bin():
    data = Dataset(pd.DataFrame({"gain": [0, 0, 0, 500, 2500, 9000]}), quasi_identifiers=["gain"])
    enc = TableEncoder(bin_edges={"gain": [0, 1, 1000, 10000]}).fit(data)
    codes = enc.encode(data)
    assert list(codes["gain"]) == [0, 0, 0, 1, 2, 2]
    decoded = enc.decode(codes, rng=np.random.default_rng(0))
    assert list(decoded["gain"][:3]) == [0, 0, 0]      # zeros stay zero
    assert decoded["gain"][3] < 1000 and decoded["gain"][5] >= 1000


def test_bad_bin_edges_rejected():
    with pytest.raises(ValueError):
        TableEncoder(bin_edges={"age": [10, 5, 20]})


def test_unknown_category_rejected(small_dataset):
    enc = TableEncoder(categories={"sex": ["F"]}).fit(small_dataset)
    with pytest.raises(SchemaError):
        enc.encode(small_dataset)


def test_encoder_must_be_fitted():
    with pytest.raises(NotFittedError):
        TableEncoder().columns


# --- Both generators (polymorphism) ------------------------------------- #
@pytest.mark.parametrize("cls", GENERATORS)
def test_output_has_same_columns_and_schema(cls, linked_dataset):
    synthetic = cls(epsilon=1.0, random_state=0).fit_sample(linked_dataset, n=500)
    assert isinstance(synthetic, Dataset)
    assert synthetic.columns == linked_dataset.columns
    assert synthetic.n_rows == 500
    assert synthetic.quasi_identifiers == linked_dataset.quasi_identifiers
    assert synthetic.sensitive_attribute == "pay"
    assert pd.api.types.is_integer_dtype(synthetic.data["age"])
    assert synthetic.data["age"].between(18, 69).all()


@pytest.mark.parametrize("cls", GENERATORS)
@pytest.mark.parametrize("noise", ["laplace", "gaussian"])
def test_budget_is_used_exactly_and_never_exceeded(cls, noise, linked_dataset):
    synth = cls(epsilon=0.8, noise=noise, random_state=0).fit(linked_dataset)
    assert synth.budget.spent_epsilon == pytest.approx(0.8)
    assert synth.budget.spent_delta == pytest.approx(synth.delta)
    assert len(synth.budget.ledger) > 0


@pytest.mark.parametrize("cls", GENERATORS)
def test_sampling_before_fit_raises(cls):
    with pytest.raises(NotFittedError):
        cls().sample(10)


@pytest.mark.parametrize("cls", GENERATORS)
def test_same_seed_gives_same_data(cls, linked_dataset):
    a = cls(epsilon=1.0, random_state=3).fit_sample(linked_dataset, n=100).data
    b = cls(epsilon=1.0, random_state=3).fit_sample(linked_dataset, n=100).data
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cls", GENERATORS)
def test_default_size_is_noisy_count_close_to_real(cls, linked_dataset):
    synthetic = cls(epsilon=5.0, random_state=0).fit(linked_dataset).sample()
    assert abs(synthetic.n_rows - linked_dataset.n_rows) < 50


@pytest.mark.parametrize("cls", GENERATORS)
def test_large_epsilon_reproduces_marginals(cls, linked_dataset):
    synthetic = cls(epsilon=50.0, random_state=0).fit_sample(linked_dataset, n=5000)
    real = linked_dataset.data["degree"].value_counts(normalize=True)
    fake = synthetic.data["degree"].value_counts(normalize=True)
    assert (real - fake.reindex(real.index, fill_value=0)).abs().sum() < 0.05


@pytest.mark.parametrize("cls", GENERATORS)
def test_invalid_settings_rejected(cls):
    with pytest.raises(ValueError):
        cls(epsilon=0)
    with pytest.raises(ValueError):
        cls(noise="uniform")
    with pytest.raises(ValueError):
        cls(noise="gaussian", delta=0.0)


@pytest.mark.parametrize("cls", GENERATORS)
def test_fit_needs_a_dataset(cls, linked_dataset):
    with pytest.raises(SchemaError):
        cls().fit(linked_dataset.data)


@pytest.mark.parametrize("cls", GENERATORS)
def test_single_column_dataset_works(cls):
    data = Dataset(pd.DataFrame({"sex": ["F", "M"] * 50}), quasi_identifiers=["sex"])
    synth = cls(epsilon=1.0, random_state=0)
    assert synth.fit_sample(data, n=20).n_rows == 20
    assert synth.budget.spent_epsilon == pytest.approx(1.0)


def test_registry_lists_both_generators():
    assert set(SYNTHESIZERS) == {"independent", "bayesian"}
    assert all(issubclass(cls, Synthesizer) for cls in SYNTHESIZERS.values())


# --- Generator 2 is better than generator 1 ----------------------------- #
def _agreement(dataset: Dataset) -> float:
    """Share of records where pay matches what degree implies."""
    frame = dataset.data
    expected = frame["degree"].map({"none": "low", "diploma": "mid", "degree": "high"})
    return float((frame["pay"] == expected).mean())


def test_bayesian_network_keeps_relationships_independent_loses_them(linked_dataset):
    independent = IndependentMarginalSynthesizer(epsilon=5.0, random_state=0)
    bayesian = BayesianNetworkSynthesizer(epsilon=5.0, random_state=0)
    ind = _agreement(independent.fit_sample(linked_dataset, n=3000))
    bay = _agreement(bayesian.fit_sample(linked_dataset, n=3000))
    assert ind < 0.5           # about 0.5*0.5 + 0.3*0.3 + 0.2*0.2 = 0.38 by chance
    assert bay > 0.9
    assert ("pay", "degree") in bayesian.network or ("degree", "pay") in bayesian.network


def test_network_covers_every_column_once(linked_dataset):
    synth = BayesianNetworkSynthesizer(epsilon=1.0, random_state=1).fit(linked_dataset)
    children = [child for child, _ in synth.network]
    assert sorted(children) == sorted(linked_dataset.columns)
    assert synth.network[0][1] is None                   # the root has no parent
    seen = set()
    for child, parent in synth.network:                  # parents come before children
        assert parent is None or parent in seen
        seen.add(child)


def test_mutual_information_basic_cases():
    x = np.array([0, 1] * 50)
    assert mutual_information(x, x, 2, 2) == pytest.approx(1.0)            # 1 bit
    y = np.array([0, 0, 1, 1] * 25)
    assert mutual_information(x, y, 2, 2) == pytest.approx(0.0, abs=1e-12)


def test_structure_fraction_validated():
    with pytest.raises(ValueError):
        BayesianNetworkSynthesizer(structure_fraction=1.0)
