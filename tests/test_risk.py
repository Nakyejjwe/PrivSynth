"""Tests for the k-anonymity checker and generaliser."""

import pandas as pd
import pytest

from privsynth import Dataset, Generalizer, KAnonymityChecker, SchemaError


def test_class_sizes_and_headline_numbers(small_dataset):
    report = KAnonymityChecker(k=2).evaluate(small_dataset)
    assert report.record_class_sizes.tolist() == [3, 3, 3, 2, 2, 2, 2, 1]
    assert report.k_achieved == 1
    assert report.n_classes == 4
    assert report.n_unique == 1
    assert report.n_at_risk == 1
    assert report.max_risk == 1.0
    assert not report.is_k_anonymous


def test_average_risk_is_mean_of_inverse_class_size(small_dataset):
    report = KAnonymityChecker(k=2).evaluate(small_dataset)
    expected = (3 * (1 / 3) + 4 * (1 / 2) + 1 * 1) / 8
    assert report.average_risk == pytest.approx(expected)
    assert report.record_risks.iloc[-1] == 1.0


def test_l_diversity(small_dataset):
    # The (30, M) and (40, F) classes contain only '<=50K'
    assert KAnonymityChecker().evaluate(small_dataset).l_diversity == 1


def test_fewer_quasi_identifiers_means_lower_risk(small_dataset):
    report = KAnonymityChecker(k=2).evaluate(small_dataset, quasi_identifiers=["sex"])
    assert report.k_achieved == 3
    assert report.is_k_anonymous


def test_fully_identical_records_are_k_anonymous():
    ds = Dataset(pd.DataFrame({"age": [30] * 6, "sex": ["F"] * 6}), quasi_identifiers=["age", "sex"])
    report = KAnonymityChecker(k=5).evaluate(ds)
    assert report.is_k_anonymous and report.average_risk == pytest.approx(1 / 6)


def test_missing_values_form_their_own_class():
    frame = pd.DataFrame({"age": [30, 30, None], "sex": ["F", "F", "F"]})
    ds = Dataset(frame, quasi_identifiers=["age", "sex"])
    assert KAnonymityChecker().evaluate(ds).record_class_sizes.tolist() == [2, 2, 1]


def test_risky_records(small_dataset):
    checker = KAnonymityChecker(k=3)
    risky = checker.risky_records(small_dataset)
    assert len(risky) == 5
    assert risky.iloc[0]["age"] == 55 and risky.iloc[0]["risk"] == 1.0


@pytest.mark.parametrize("bad_k", [0, -1, 2.5, "5", True])
def test_invalid_k_rejected(bad_k):
    with pytest.raises(ValueError):
        KAnonymityChecker(k=bad_k)


def test_checker_needs_quasi_identifiers(small_frame):
    with pytest.raises(SchemaError):
        KAnonymityChecker().evaluate(Dataset(small_frame))
    with pytest.raises(SchemaError):
        KAnonymityChecker().evaluate(Dataset(small_frame), quasi_identifiers=["district"])


def test_report_dict_and_text(small_dataset):
    report = KAnonymityChecker(k=2).evaluate(small_dataset)
    d = report.to_dict()
    assert d["unique_records"] == 1 and d["pct_unique"] == 12.5 and d["k_anonymous"] is False
    assert "FAILS" in str(report)


# --- Generalizer ---------------------------------------------------------
def test_age_binning_raises_k(small_dataset):
    generalised = Generalizer(bin_widths={"age": 30}).generalize(small_dataset)
    assert sorted(generalised.data["age"].unique()) == ["30-59"]
    assert small_dataset.data["age"].iloc[0] == 30          # original untouched
    report = KAnonymityChecker(k=3).evaluate(generalised)
    assert report.k_achieved == 3 and report.is_k_anonymous


def test_rare_categories_merged(small_dataset):
    ds = small_dataset.with_data(small_dataset.data)
    ds.quasi_identifiers = ["city"]
    out = Generalizer(rare_threshold=2).generalize(ds)
    assert out.data["city"].tolist().count("Other") == 2   # Mbarara and Jinja


def test_generalizer_validation(small_dataset):
    with pytest.raises(ValueError):
        Generalizer(bin_widths={"age": 0})
    with pytest.raises(ValueError):
        Generalizer(rare_threshold=-1)
    with pytest.raises(SchemaError):
        Generalizer(bin_widths={"sex": 5}).generalize(small_dataset)
    with pytest.raises(SchemaError):
        Generalizer(bin_widths={"district": 5}).generalize(small_dataset)
