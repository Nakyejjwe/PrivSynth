"""Tests for the utility and privacy-risk evaluators and the evaluation suite."""

import numpy as np
import pandas as pd
import pytest

from privsynth import (
    BayesianNetworkSynthesizer,
    Dataset,
    EvaluationError,
    IndependentMarginalSynthesizer,
    SchemaError,
)
from privsynth.evaluation import (
    AssociationSimilarity,
    AttributeInferenceRisk,
    DistanceToClosestRecord,
    EvaluationResult,
    EvaluationSuite,
    Evaluator,
    MachineLearningUtility,
    MarginalSimilarity,
    MixedFeatureSpace,
    cramers_v,
    default_evaluators,
    total_variation_distance,
)


def _shuffled(dataset: Dataset, seed: int = 0) -> Dataset:
    """Shuffle every column independently: keeps 1-way marginals, destroys relationships."""
    rng = np.random.default_rng(seed)
    frame = dataset.data
    for col in frame.columns:
        frame[col] = frame[col].to_numpy()[rng.permutation(len(frame))]
    return dataset.with_data(frame, name="shuffled")


@pytest.fixture
def income_data():
    """3,000 people where income depends strongly on education and hours.

    Every person has a unique ``person_id``-like ``code`` quasi-identifier
    combination with ``age`` and ``sex``, so copying records is detectable.
    Returns (train, holdout).
    """
    rng = np.random.default_rng(11)
    n = 3000
    education = rng.choice(["primary", "secondary", "university"], size=n, p=[0.4, 0.4, 0.2])
    hours = rng.integers(10, 70, size=n)
    score = (education == "university") * 2.5 + (education == "secondary") * 1.0 + (hours - 40) / 10
    high = rng.random(n) < 1 / (1 + np.exp(-(score - 1.5)))
    frame = pd.DataFrame({
        "age": rng.integers(18, 80, size=n),
        "sex": rng.choice(["F", "M"], size=n),
        "district": rng.choice([f"D{i}" for i in range(30)], size=n),
        "education": education,
        "hours": hours,
        "income": np.where(high, "high", "low"),
    })
    data = Dataset(frame, name="income", quasi_identifiers=["age", "sex", "district"],
                   sensitive_attribute="income")
    return data.split(0.3, random_state=0)


# --- Helpers ----------------------------------------------------------------- #
def test_tvd_basic_cases():
    assert total_variation_distance([1, 1], [5, 5]) == 0.0          # same proportions
    assert total_variation_distance([1, 0], [0, 1]) == 1.0          # no overlap
    assert total_variation_distance([3, 1], [1, 1]) == pytest.approx(0.25)
    assert total_variation_distance([0, 0], [0, 0]) == 0.0
    assert total_variation_distance([0, 0], [1, 0]) == 1.0


def test_tvd_rejects_bad_input():
    with pytest.raises(ValueError):
        total_variation_distance([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        total_variation_distance([-1, 2], [1, 2])


def test_cramers_v_extremes():
    a = np.array([0, 1, 2] * 100)
    assert cramers_v(a, a, 3, 3) == pytest.approx(1.0)                 # one decides the other
    b = np.tile(np.repeat([0, 1], 3), 50)                              # balanced against a
    assert cramers_v(a, b, 3, 2) == pytest.approx(0.0, abs=1e-12)
    assert cramers_v(np.zeros(10, int), np.arange(10) % 2, 1, 2) == 0.0  # constant column


def test_mixed_feature_space_distances():
    frame = pd.DataFrame({"age": [20, 60, 20], "sex": ["F", "F", "M"]})
    space = MixedFeatureSpace().fit(frame, numeric=["age"])
    x = space.transform(frame)
    assert np.linalg.norm(x[0] - x[1]) == pytest.approx(1.0)    # opposite ends of the age range
    assert np.linalg.norm(x[0] - x[2]) == pytest.approx(1.0)    # one category differs


# --- Input checks shared by every evaluator (template method) -------------- #
ALL = [MarginalSimilarity, AssociationSimilarity, MachineLearningUtility,
       DistanceToClosestRecord, AttributeInferenceRisk]


@pytest.mark.parametrize("cls", ALL)
def test_every_evaluator_rejects_non_datasets(cls, income_data):
    train, holdout = income_data
    with pytest.raises(SchemaError):
        cls().evaluate(train.data, train, holdout)


@pytest.mark.parametrize("cls", ALL)
def test_every_evaluator_rejects_different_columns(cls, income_data):
    train, holdout = income_data
    with pytest.raises(SchemaError):
        cls().evaluate(train, train.select(["age", "sex", "income"]), holdout)


@pytest.mark.parametrize("cls", [MachineLearningUtility, DistanceToClosestRecord, AttributeInferenceRisk])
def test_holdout_evaluators_need_a_holdout(cls, income_data):
    train, _ = income_data
    assert cls.requires_holdout
    with pytest.raises(EvaluationError):
        cls().evaluate(train, train)


@pytest.mark.parametrize("cls", ALL)
def test_every_evaluator_returns_a_result(cls, income_data):
    train, holdout = income_data
    result = cls().evaluate(train, _shuffled(train), holdout)
    assert isinstance(result, EvaluationResult)
    assert result.kind in {"utility", "risk"}
    assert result.metrics and all(isinstance(v, float) for v in result.metrics.values())
    assert set(result.lower_is_better) <= set(result.metrics)
    assert str(result).startswith(cls.name)


def test_column_order_does_not_matter(income_data):
    train, holdout = income_data
    reordered = train.select(list(reversed(train.columns)))
    result = MarginalSimilarity().evaluate(train, reordered, holdout)
    assert result["tvd_2way_max"] == 0.0


def test_evaluator_is_abstract():
    with pytest.raises(TypeError):
        Evaluator()


# --- Utility --------------------------------------------------------------- #
def test_marginals_identical_data_scores_zero(linked_dataset):
    result = MarginalSimilarity().evaluate(linked_dataset, linked_dataset)
    assert result["tvd_1way_mean"] == 0.0
    assert result["tvd_2way_mean"] == 0.0


def test_marginals_detect_lost_relationships(linked_dataset):
    result = MarginalSimilarity().evaluate(linked_dataset, _shuffled(linked_dataset))
    assert result["tvd_1way_max"] == 0.0               # each column unchanged
    pairs = result.details.set_index("columns")["tvd"]
    assert pairs["degree x pay"] > 0.4                  # their link is gone
    assert pairs["age x region"] < 0.15                 # these were unrelated anyway


def test_bayesian_generator_keeps_pairs_better_than_independent(linked_dataset):
    ind = IndependentMarginalSynthesizer(epsilon=5.0, random_state=0).fit_sample(linked_dataset)
    bay = BayesianNetworkSynthesizer(epsilon=5.0, random_state=0).fit_sample(linked_dataset)
    evaluator = MarginalSimilarity()
    assert (evaluator.evaluate(linked_dataset, bay)["tvd_2way_mean"]
            < evaluator.evaluate(linked_dataset, ind)["tvd_2way_mean"])


def test_association_similarity(linked_dataset):
    same = AssociationSimilarity().evaluate(linked_dataset, linked_dataset)
    assert same["assoc_diff_max"] == 0.0
    broken = AssociationSimilarity().evaluate(linked_dataset, _shuffled(linked_dataset))
    row = broken.details.query("column_a == 'degree' and column_b == 'pay'").iloc[0]
    assert row["real_v"] == pytest.approx(1.0)
    assert row["synthetic_v"] < 0.1
    matrix = AssociationSimilarity.matrix(broken.details)
    assert matrix.loc["pay", "degree"] == pytest.approx(1.0)
    assert list(matrix.columns) == linked_dataset.columns


def test_ml_utility_real_data_matches_itself(income_data):
    train, holdout = income_data
    result = MachineLearningUtility().evaluate(train, train, holdout)
    assert result["auc_synthetic"] == pytest.approx(result["auc_real"])
    assert result["utility_ratio"] == pytest.approx(1.0)
    assert result["auc_real"] > 0.75                    # the signal is learnable
    assert result["accuracy_real"] > result["accuracy_majority"]


def test_ml_utility_shuffled_data_is_useless(income_data):
    train, holdout = income_data
    result = MachineLearningUtility(model="random_forest").evaluate(train, _shuffled(train), holdout)
    assert result["auc_synthetic"] == pytest.approx(0.5, abs=0.1)


def test_ml_utility_single_class_synthetic_data_is_handled(income_data):
    train, holdout = income_data
    frame = train.data
    frame["income"] = "low"
    result = MachineLearningUtility().evaluate(train, train.with_data(frame), holdout)
    assert result["auc_synthetic"] == 0.5
    assert any("only one value" in note for note in result.notes)


def test_ml_utility_multiclass_uses_f1(linked_dataset):
    train, holdout = linked_dataset.split(0.3, random_state=0)
    result = MachineLearningUtility().evaluate(train, train, holdout)
    assert "auc_real" not in result.metrics
    assert result["f1_real"] == pytest.approx(1.0)      # degree decides pay exactly


def test_ml_utility_needs_a_target(income_data):
    train, holdout = income_data
    no_target = train.with_data(train.data)
    no_target.sensitive_attribute = None
    with pytest.raises(EvaluationError):
        MachineLearningUtility().evaluate(no_target, no_target, holdout)
    with pytest.raises(SchemaError):
        MachineLearningUtility(target="salary").evaluate(train, train, holdout)
    with pytest.raises(ValueError):
        MachineLearningUtility(model="svm")


# --- Risk ------------------------------------------------------------------ #
def test_dcr_flags_copied_records(income_data):
    train, holdout = income_data
    result = DistanceToClosestRecord().evaluate(train, train, holdout)
    assert result["exact_match_synthetic"] == 1.0
    assert result["dcr_synthetic_median"] == 0.0
    # The equal-sized training sample holds 900 of the 2,100 copied records, so
    # about 900/2100 queries find themselves and the rest split 50/50: ~0.71.
    assert result["closer_to_train_share"] > 0.6
    assert result["exact_match_holdout"] < 0.05


def test_dcr_fresh_people_look_like_holdout(income_data):
    train, holdout = income_data
    fresh, rest = holdout.split(0.5, random_state=1)    # real people the "generator" never saw
    result = DistanceToClosestRecord().evaluate(train, fresh, rest)
    assert result["exact_match_synthetic"] == 0.0
    assert 0.3 < result["closer_to_train_share"] < 0.7
    assert result["dcr_ratio"] == pytest.approx(1.0, abs=0.35)


def test_dcr_validates_settings():
    with pytest.raises(ValueError):
        DistanceToClosestRecord(max_records=0)


def test_inference_attack_succeeds_on_copied_records(income_data):
    train, holdout = income_data
    result = AttributeInferenceRisk().evaluate(train, train, holdout)
    # Most members are unique on (age, sex, district), so a copy reveals their income.
    assert result["accuracy_members"] > 0.85
    assert result["accuracy_non_members"] < 0.8
    assert result["inference_risk"] > 0.7
    assert result["match_rate_members"] == 1.0


def test_inference_attack_learns_nothing_from_shuffled_data(income_data):
    train, holdout = income_data
    result = AttributeInferenceRisk().evaluate(train, _shuffled(train), holdout)
    assert result["inference_risk"] < 0.1


def test_inference_with_single_quasi_identifier(income_data):
    train, holdout = income_data
    result = AttributeInferenceRisk(quasi_identifiers=["sex"]).evaluate(train, train, holdout)
    assert result["match_rate_members"] == 1.0
    assert 0 <= result["inference_risk"] <= 1


def test_inference_needs_schema(income_data):
    train, holdout = income_data
    with pytest.raises(EvaluationError):
        AttributeInferenceRisk(quasi_identifiers=["income"]).evaluate(train, train, holdout)
    with pytest.raises(SchemaError):
        AttributeInferenceRisk(sensitive_attribute="salary").evaluate(train, train, holdout)


def test_dp_synthesizer_has_low_risk(income_data):
    train, holdout = income_data
    synthetic = BayesianNetworkSynthesizer(epsilon=1.0, random_state=0).fit_sample(train)
    suite = EvaluationSuite([DistanceToClosestRecord(), AttributeInferenceRisk()])
    report = suite.run(train, synthetic, holdout)
    assert report["dcr"]["exact_match_synthetic"] < 0.01
    assert report["inference"]["inference_risk"] < 0.15


# --- Suite and report (composition, polymorphism) -------------------------- #
def test_suite_runs_default_evaluators(income_data):
    train, holdout = income_data
    report = EvaluationSuite(default_evaluators()).run(train, _shuffled(train), holdout)
    assert [r.key for r in report] == ["marginal", "association", "ml", "dcr", "inference"]
    assert len(report.utility) == 3 and len(report.risk) == 2
    flat = report.flat()
    assert "ml.auc_synthetic" in flat and "inference.inference_risk" in flat
    frame = report.to_frame()
    assert set(frame["kind"]) == {"utility", "risk"}
    assert len(frame) == len(flat)
    assert "ml" in report and "nope" not in report
    with pytest.raises(KeyError):
        report["nope"]


def test_suite_skips_holdout_evaluators_without_holdout(linked_dataset):
    report = EvaluationSuite(default_evaluators()).run(linked_dataset, linked_dataset)
    assert [r.key for r in report] == ["marginal", "association"]
    assert set(report.skipped) == {MachineLearningUtility.name, DistanceToClosestRecord.name,
                                   AttributeInferenceRisk.name}
    assert "skipped" in str(report)


def test_suite_can_record_errors_instead_of_raising(income_data):
    train, holdout = income_data
    bad = MachineLearningUtility(target="salary")
    with pytest.raises(SchemaError):
        EvaluationSuite([bad]).run(train, train, holdout)
    report = EvaluationSuite([bad, MarginalSimilarity()], raise_errors=False).run(train, train, holdout)
    assert [r.key for r in report] == ["marginal"]
    assert "salary" in report.skipped[bad.name]


def test_suite_validates_members():
    with pytest.raises(TypeError):
        EvaluationSuite([MarginalSimilarity(), "not an evaluator"])
    with pytest.raises(ValueError):
        EvaluationSuite([])


def test_inference_lower_bound_separates_leaks_from_noise(income_data):
    train, holdout = income_data
    leak = AttributeInferenceRisk().evaluate(train, train, holdout)
    assert 0 < leak["inference_risk_lower"] < leak["inference_risk"]
    noise = AttributeInferenceRisk().evaluate(train, _shuffled(train), holdout)
    assert noise["inference_risk_lower"] == 0.0
    assert "inference_risk_lower" in noise.lower_is_better
