"""Tests for Dataset.split, PrivSynthPipeline and the ExperimentRunner."""

import math

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from privsynth import (
    BayesianNetworkSynthesizer,
    ExperimentResults,
    ExperimentRunner,
    IndependentMarginalSynthesizer,
    PipelineResult,
    PrivSynthPipeline,
    SchemaError,
)
from privsynth.evaluation import AttributeInferenceRisk, MarginalSimilarity
from privsynth.experiments import REAL_DATA


# --- Dataset.split ------------------------------------------------------------ #
def test_split_is_disjoint_and_keeps_schema(linked_dataset):
    train, holdout = linked_dataset.split(0.25, random_state=0)
    assert train.n_rows == 1500 and holdout.n_rows == 500
    assert train.quasi_identifiers == linked_dataset.quasi_identifiers
    assert holdout.sensitive_attribute == "pay"
    together = pd.concat([train.data, holdout.data]).sort_values(list(train.columns))
    original = linked_dataset.data.sort_values(list(train.columns))
    pd.testing.assert_frame_equal(together.reset_index(drop=True), original.reset_index(drop=True))


def test_split_is_reproducible(linked_dataset):
    a, _ = linked_dataset.split(0.2, random_state=4)
    b, _ = linked_dataset.split(0.2, random_state=4)
    pd.testing.assert_frame_equal(a.data, b.data)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5, True])
def test_split_rejects_bad_fractions(fraction, linked_dataset):
    with pytest.raises(ValueError):
        linked_dataset.split(fraction)


def test_split_needs_enough_records(small_dataset):
    with pytest.raises(ValueError):
        small_dataset.split(0.01)          # rounds to an empty holdout


# --- Pipeline ------------------------------------------------------------------ #
def test_pipeline_runs_end_to_end(linked_dataset):
    pipeline = PrivSynthPipeline.from_settings("bayesian", epsilon=2.0, random_state=0)
    result = pipeline.run(linked_dataset)
    assert isinstance(result, PipelineResult)
    assert result.train.n_rows == 1600 and result.holdout.n_rows == 400
    assert result.synthetic.n_rows == result.train.n_rows
    assert result.synthetic.columns == linked_dataset.columns
    assert result.budget.spent_epsilon == pytest.approx(2.0)
    assert [r.key for r in result.evaluation] == ["marginal", "association", "ml", "dcr", "inference"]
    assert result.original_risk.n_records == result.train.n_rows

    summary = result.summary()
    assert summary["generator"] == "Bayesian network"
    assert summary["epsilon_spent"] == pytest.approx(2.0)
    assert "ml.f1_synthetic" in summary and "seconds.generate" in summary
    assert "PrivSynth run" in str(result) and "k-anonymity" in str(result)


def test_pipeline_accepts_any_synthesizer_and_custom_evaluators(linked_dataset):
    pipeline = PrivSynthPipeline(
        IndependentMarginalSynthesizer(epsilon=1.0, random_state=0),
        evaluators=[MarginalSimilarity()],
        k=3,
    )
    result = pipeline.run(linked_dataset, n=250)
    assert result.synthetic.n_rows == 250
    assert [r.key for r in result.evaluation] == ["marginal"]
    assert result.original_risk.k_target == 3


def test_pipeline_without_holdout_skips_holdout_evaluators(linked_dataset):
    pipeline = PrivSynthPipeline(BayesianNetworkSynthesizer(random_state=0), holdout_fraction=0)
    result = pipeline.run(linked_dataset)
    assert result.holdout is None
    assert result.train.n_rows == linked_dataset.n_rows
    assert AttributeInferenceRisk.name in result.evaluation.skipped
    assert result.summary()["n_holdout"] == 0


def test_pipeline_validates_inputs(linked_dataset):
    with pytest.raises(TypeError):
        PrivSynthPipeline("bayesian")
    with pytest.raises(ValueError):
        PrivSynthPipeline(BayesianNetworkSynthesizer(), holdout_fraction=1.0)
    with pytest.raises(ValueError):
        PrivSynthPipeline.from_settings("gan")
    with pytest.raises(SchemaError):
        PrivSynthPipeline.from_settings("independent").run(linked_dataset.data)


def test_pipeline_with_gaussian_noise(linked_dataset):
    pipeline = PrivSynthPipeline.from_settings("independent", epsilon=1.0, noise="gaussian", random_state=0)
    result = pipeline.run(linked_dataset)
    assert result.budget.spent_delta == pytest.approx(1e-5)


# --- Experiments --------------------------------------------------------------- #
@pytest.fixture
def small_results(linked_dataset):
    runner = ExperimentRunner(
        linked_dataset,
        epsilons=[0.5, 5.0],
        seeds=[0, 1],
        evaluators=lambda: [MarginalSimilarity(), AttributeInferenceRisk(max_records=300)],
    )
    messages = []
    results = runner.run(progress=messages.append)
    assert len(messages) == len(runner.configs()) + 1        # plus the reference row
    return results


def test_runner_grid(linked_dataset):
    runner = ExperimentRunner(linked_dataset, epsilons=[1, 2], noises=["laplace", "gaussian"], seeds=[0, 1, 2])
    assert len(runner.configs()) == 2 * 2 * 2 * 3
    assert runner.train.n_rows + runner.holdout.n_rows == linked_dataset.n_rows


def test_results_table(small_results):
    runs = small_results.runs
    assert len(small_results) == 2 * 2 * 2 + 1
    base = small_results.baseline
    assert base["generator"] == REAL_DATA and math.isinf(base["epsilon"])
    assert base["marginal.tvd_2way_mean"] == 0.0
    assert "marginal.tvd_2way_mean" in small_results.metrics
    assert set(runs["generator"]) == {REAL_DATA, "Independent marginals", "Bayesian network"}


def test_more_epsilon_means_more_utility(small_results):
    table = small_results.summary(["marginal.tvd_2way_mean"])
    bayes = table.loc[("Bayesian network", "laplace")]["marginal.tvd_2way_mean"]
    assert bayes.loc[5.0] < bayes.loc[0.5]


def test_summary_with_std_and_bad_metric(small_results):
    table = small_results.summary(["marginal.tvd_1way_mean"], std=True)
    assert list(table.columns) == ["marginal.tvd_1way_mean (mean)", "marginal.tvd_1way_mean (std)"]
    with pytest.raises(KeyError):
        small_results.summary(["nope"])


def test_results_csv_round_trip(small_results, tmp_path):
    path = tmp_path / "runs.csv"
    small_results.to_csv(path)
    loaded = ExperimentResults.from_csv(path)
    pd.testing.assert_frame_equal(loaded.runs, small_results.runs, check_dtype=False)
    assert math.isinf(loaded.baseline["epsilon"])


def test_results_plot(small_results):
    ax = small_results.plot("marginal.tvd_2way_mean", title="2-way TVD", reference=0.1, reference_label="ten")
    labels = ax.get_legend_handles_labels()[1]
    assert {"Independent marginals", "Bayesian network", REAL_DATA, "ten"} <= set(labels)
    with pytest.raises(KeyError):
        small_results.plot("nope")


def test_results_need_setting_columns():
    with pytest.raises(ValueError):
        ExperimentResults(pd.DataFrame({"epsilon": [1.0]}))


@pytest.mark.parametrize("kwargs", [
    {"epsilons": []}, {"epsilons": [0]}, {"epsilons": [float("inf")]},
    {"generators": ["gan"]}, {"noises": ["uniform"]}, {"seeds": []},
])
def test_runner_validates_settings(kwargs, linked_dataset):
    with pytest.raises(ValueError):
        ExperimentRunner(linked_dataset, **kwargs)


def test_runner_needs_a_dataset(linked_dataset):
    with pytest.raises(TypeError):
        ExperimentRunner(linked_dataset.data)
