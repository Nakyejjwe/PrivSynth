"""Tests for the dashboard logic (privsynth.dashboard) and the Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from privsynth import DataLoadError, Dataset, KAnonymityChecker
from privsynth.dashboard import (
    PRIVACY_LEVELS,
    PrivacySettings,
    SampleDataSource,
    UploadedCSVSource,
    Verdict,
    assess_disclosure,
    assess_original_risk,
    assess_utility,
    level_for,
    overall_recommendation,
    sensitive_candidates,
    suggest_schema,
)
from privsynth.dashboard import charts
from privsynth.dashboard.sources import MAX_UPLOAD_MB, DataSource
from privsynth.dashboard.verdicts import METRIC_GUIDE
from privsynth.evaluation import EvaluationReport, EvaluationResult, EvaluationSuite, default_evaluators

ROOT = Path(__file__).resolve().parents[1]


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


@pytest.fixture
def survey():
    """A made-up survey with direct identifiers, demographics and an outcome."""
    rng = np.random.default_rng(3)
    n = 400
    return pd.DataFrame({
        "PatientID": [f"P{i:05d}" for i in range(n)],
        "full_name": [f"Person {i}" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "Sex": rng.choice(["F", "M"], n),
        "district": rng.choice(["Kampala", "Gulu", "Mbarara", "Jinja"], n),
        "household_size": rng.integers(1, 9, n),
        "diagnosis": rng.choice(["malaria", "typhoid", "none"], n),
    })


# --- Schema suggestions -------------------------------------------------------- #
def test_suggest_schema_finds_roles(survey):
    s = suggest_schema(survey)
    assert s.direct_identifiers == ["PatientID", "full_name"]
    assert s.quasi_identifiers == ["age", "Sex", "district"]
    assert s.sensitive_attribute == "diagnosis"
    assert set(s.reasons) >= {"PatientID", "age", "diagnosis"}


def test_unique_text_column_is_a_direct_identifier(survey):
    frame = survey.rename(columns={"full_name": "x1"})
    s = suggest_schema(frame)
    assert "x1" in s.direct_identifiers
    assert "different values" in s.reasons["x1"]


def test_numeric_unique_column_is_not_flagged():
    frame = pd.DataFrame({"weight": np.arange(100) * 1.5, "group": ["a", "b"] * 50})
    assert suggest_schema(frame).direct_identifiers == []


def test_sensitive_falls_back_to_last_low_cardinality_column():
    frame = pd.DataFrame({"x": range(50), "colour": ["r", "g"] * 25, "flag": ["y", "n", "m", "y", "n"] * 10})
    assert suggest_schema(frame).sensitive_attribute == "flag"


def test_no_sensitive_candidate():
    frame = pd.DataFrame({"a": range(30), "b": range(30, 60)})
    assert suggest_schema(frame).sensitive_attribute is None


def test_sensitive_candidates(survey):
    assert sensitive_candidates(survey) == ["Sex", "district", "household_size", "diagnosis"]


# --- Data sources (polymorphism) ---------------------------------------------- #
def test_sample_source_loads_adult():
    source = SampleDataSource()
    assert isinstance(source, DataSource)
    data = source.load()
    assert data.sensitive_attribute == "income"
    assert "age" in data.quasi_identifiers
    assert "capital_gain" in source.bin_edges
    assert source.description


def test_sample_source_falls_back_to_the_sample(tmp_path):
    (tmp_path / "sample_adult.csv").write_bytes((ROOT / "data" / "sample_adult.csv").read_bytes())
    source = SampleDataSource(tmp_path)
    assert not source.is_full
    assert source.load().n_rows > 1000


def test_upload_applies_suggestions(survey):
    source = UploadedCSVSource(_csv_bytes(survey), "clinic.csv")
    data = source.load()
    assert data.name == "clinic"
    assert data.quasi_identifiers == ["age", "Sex", "district"]
    assert data.sensitive_attribute == "diagnosis"
    assert source.rows_dropped == 0
    assert "clinic.csv" in source.description


def test_upload_reports_dropped_rows(survey):
    frame = survey.astype({"district": object})
    frame.loc[:9, "district"] = "?"
    frame.loc[10:14, "diagnosis"] = None
    source = UploadedCSVSource(_csv_bytes(frame))
    assert source.load().n_rows == len(frame) - 15
    assert source.rows_dropped == 15


@pytest.mark.parametrize("content", [b"", b"just one column\n1\n2\n"])
def test_upload_rejects_unusable_files(content):
    with pytest.raises(DataLoadError):
        UploadedCSVSource(content).load()


def test_upload_size_limit():
    with pytest.raises(DataLoadError, match="limit"):
        UploadedCSVSource(b"x" * int((MAX_UPLOAD_MB + 1) * 1e6))


# --- Settings -------------------------------------------------------------------- #
def test_privacy_levels_are_ordered_and_findable():
    epsilons = [lvl.epsilon for lvl in PRIVACY_LEVELS]
    assert epsilons == sorted(epsilons)
    assert level_for(1.0).name == "Balanced"
    assert level_for(0.7) is None


def test_settings_describe():
    assert PrivacySettings().describe() == "Bayesian network, Laplace noise, ε = 1 (Balanced)"
    text = PrivacySettings(generator="independent", noise="gaussian", epsilon=0.7).describe()
    assert text == "Independent columns, Gaussian noise, ε = 0.7, δ = 1e-05"


@pytest.mark.parametrize("bad", [
    {"generator": "gan"}, {"noise": "uniform"}, {"epsilon": 0}, {"epsilon": float("nan")},
    {"epsilon": 50}, {"noise": "gaussian", "delta": 1.5}, {"k": 1}, {"k": 2.5},
    {"holdout_fraction": 0}, {"holdout_fraction": 0.6}, {"n_synthetic": 0},
])
def test_settings_validation(bad):
    with pytest.raises(ValueError):
        PrivacySettings(**bad).validate()


def test_settings_build_a_working_pipeline(linked_dataset):
    settings = PrivacySettings(generator="independent", epsilon=2.0, k=3, seed=1)
    result = settings.build_pipeline().run(linked_dataset)
    assert result.budget.spent_epsilon == pytest.approx(2.0)
    assert result.original_risk.k_target == 3
    assert result.holdout.n_rows == 400
    assert settings.to_dict()["epsilon"] == 2.0


def test_pipeline_records_evaluator_errors_instead_of_crashing(linked_dataset):
    no_sensitive = Dataset(linked_dataset.data, quasi_identifiers=["age", "region"])
    result = PrivacySettings(seed=0).build_pipeline().run(no_sensitive)
    assert "Machine-learning utility" in result.evaluation.skipped
    assert "marginal" in result.evaluation


# --- Verdicts -------------------------------------------------------------------- #
def _report(**evaluators) -> EvaluationReport:
    return EvaluationReport(results=[
        EvaluationResult(evaluator=key, key=key, kind="x", metrics=metrics) for key, metrics in evaluators.items()
    ])


def test_verdict_rejects_unknown_level():
    with pytest.raises(ValueError):
        Verdict("purple", "x", "y")
    assert str(Verdict("good", "Fine", "All good.")).startswith("✅ Fine")


@pytest.mark.parametrize("group_sizes, level", [
    ([1] * 5 + [10] * 20, "critical"),      # 5 of 205 people unique (2.4%)
    ([4] + [10] * 99, "warning"),           # 4 of ~1000 at risk, nobody unique
    ([10] * 20, "good"),                    # everyone in a group of 10
])
def test_original_risk_verdicts(group_sizes, level):
    zone = np.repeat(np.arange(len(group_sizes)), group_sizes)
    data = Dataset(pd.DataFrame({"zone": zone.astype(str), "x": np.arange(len(zone))}), quasi_identifiers=["zone"])
    verdict = assess_original_risk(KAnonymityChecker(5).evaluate(data))
    assert verdict.level == level
    if level == "good":
        assert "k-anonymous (k = 10)" == verdict.label


@pytest.mark.parametrize("auc_synthetic, level", [(0.88, "good"), (0.82, "warning"), (0.6, "critical")])
def test_utility_verdict_levels(auc_synthetic, level):
    # Real AUC 0.9 = 0.4 above chance; 0.88 keeps 95% of it, 0.82 keeps 80%, 0.6 keeps 25%.
    report = _report(ml={"auc_real": 0.9, "auc_synthetic": auc_synthetic, "utility_ratio": auc_synthetic / 0.9})
    assert assess_utility(report).level == level


def test_utility_judges_gain_above_chance():
    # 0.54 vs 0.63 is 86% of the AUC but only 31% of the gain over a coin toss.
    report = _report(ml={"auc_real": 0.63, "auc_synthetic": 0.54, "utility_ratio": 0.86})
    verdict = assess_utility(report)
    assert verdict.level == "critical" and "31%" in verdict.message


def test_utility_ignores_unlearnable_targets():
    report = _report(ml={"auc_real": 0.52, "auc_synthetic": 0.5, "utility_ratio": 0.96},
                     marginal={"tvd_2way_mean": 0.04})
    verdict = assess_utility(report)
    assert verdict.level == "good" and "not used" in verdict.message


def test_utility_advice_depends_on_generator():
    report = _report(ml={"auc_real": 0.9, "auc_synthetic": 0.45, "utility_ratio": 0.5})
    assert "Bayesian network" in assess_utility(report, generator="independent").message
    assert "larger ε" in assess_utility(report, generator="bayesian").message


def test_utility_falls_back_to_marginals_and_unknown():
    assert assess_utility(_report(marginal={"tvd_2way_mean": 0.03})).level == "good"
    assert assess_utility(_report(marginal={"tvd_2way_mean": 0.3})).level == "critical"
    assert assess_utility(_report()).level == "unknown"


def _dcr(copies=0.0, chance=0.0, closer=0.5):
    return {"exact_match_synthetic": copies, "exact_match_holdout": chance, "closer_to_train_share": closer}


@pytest.mark.parametrize("dcr, lower, level", [
    (_dcr(), 0.0, "good"),
    (_dcr(closer=0.57), 0.0, "warning"),
    (_dcr(), 0.05, "warning"),
    (_dcr(closer=0.62), 0.0, "critical"),
    (_dcr(copies=0.2, chance=0.01), 0.0, "critical"),
    (_dcr(copies=0.03, chance=0.025), 0.0, "good"),      # copies no more common than chance
    (_dcr(), 0.12, "critical"),
])
def test_disclosure_verdict_levels(dcr, lower, level):
    report = _report(dcr=dcr, inference={"inference_risk": max(lower, 0.08), "inference_risk_lower": lower})
    assert assess_disclosure(report).level == level


def test_disclosure_uses_the_lower_bound_not_the_noisy_estimate():
    report = _report(dcr=_dcr(), inference={"inference_risk": 0.09, "inference_risk_lower": 0.0})
    verdict = assess_disclosure(report)
    assert verdict.level == "good"
    assert "within chance" in verdict.message


def test_disclosure_unknown_without_risk_evaluators():
    assert assess_disclosure(_report(marginal={"tvd_2way_mean": 0.1})).level == "unknown"


def test_overall_recommendation():
    good, warn, bad = (Verdict(level, "x", "y") for level in ("good", "warning", "critical"))
    unknown = Verdict("unknown", "x", "y")
    assert overall_recommendation(good, good, 1).startswith("Safe to share and useful")
    assert overall_recommendation(good, bad, 1).startswith("Not safe")
    assert "Bayesian" in overall_recommendation(bad, good, 1, "independent")
    assert overall_recommendation(warn, warn, 2).startswith("Usable")
    assert overall_recommendation(unknown, good, 1) is None


def test_every_default_metric_is_explained(income_like):
    train, holdout = income_like
    report = EvaluationSuite(default_evaluators()).run(train, train, holdout)
    missing = [m for r in report for m in r.metrics if m not in METRIC_GUIDE]
    assert missing == []


@pytest.fixture
def income_like(linked_dataset):
    frame = linked_dataset.data
    frame["high"] = np.where(frame["pay"] == "high", "yes", "no")
    data = Dataset(frame.drop(columns="pay"), quasi_identifiers=["age", "region"], sensitive_attribute="high")
    return data.split(0.3, random_state=0)


# --- Charts ---------------------------------------------------------------------- #
def test_group_size_chart(linked_dataset):
    report = KAnonymityChecker(5).evaluate(linked_dataset, ["age", "region"])
    fig = charts.group_size_chart(report)
    assert isinstance(fig, Figure)
    heights = [bar.get_height() for bar in fig.axes[0].patches]
    assert sum(heights) == linked_dataset.n_rows


@pytest.mark.parametrize("column", ["age", "region", "pay"])
def test_distribution_chart(linked_dataset, column):
    other = linked_dataset.sample(500, random_state=0)
    fig = charts.distribution_chart(linked_dataset, other, column)
    shares = [bar.get_height() for bar in fig.axes[0].patches]
    assert sum(shares) == pytest.approx(2.0)           # two series, each sums to 100%


def test_distribution_chart_many_categories_and_bad_column():
    frame = pd.DataFrame({"c": [f"v{i % 30}" for i in range(300)], "x": range(300)})
    data = Dataset(frame)
    fig = charts.distribution_chart(data, data, "c")
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert len(labels) == charts.MAX_CATEGORIES + 1 and labels[-1] == "Other"
    with pytest.raises(KeyError):
        charts.distribution_chart(data, data, "nope")


def test_association_chart(linked_dataset):
    from privsynth.evaluation import AssociationSimilarity
    details = AssociationSimilarity().evaluate(linked_dataset, linked_dataset).details
    fig = charts.association_chart(details, columns=linked_dataset.columns)
    assert len(fig.axes) == 3                           # two heatmaps and a colour bar


# --- The Streamlit app, run headless ---------------------------------------------- #
def test_app_runs_and_generates():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    app.run()
    assert not app.exception
    assert any("re-identification" in box.value.lower() for box in app.error) or app.success
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any("Done" in box.value for box in app.sidebar.success)
    headings = [h.value for h in app.subheader]
    assert "Column by column" in headings and "All measurements" in headings
