"""Week 3 experiments: the privacy/utility trade-off.

1. One full pipeline run (Bayesian network, epsilon = 1), printed in detail.
2. The experiment grid: both generators x Laplace/Gaussian noise x
   epsilon in {0.1, 0.5, 1, 2, 5} x 3 seeds, all on one fixed 80/20
   train/holdout split, plus a reference row where the real training data
   itself is "released".
3. Summary tables, a CSV of every run, and figures for the report.

Usage:
    python scripts/week3_experiments.py                # data/adult.data if present
    python scripts/week3_experiments.py path/to/file   # any Adult-format file
    python scripts/week3_experiments.py --figures-only # redraw charts from week3_runs.csv

Outputs (in results/):
    week3_runs.csv        every run, every metric
    week3_summary.csv     mean and std over seeds
    figures/*.png         charts for the report
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from privsynth import (
    AdultDataLoader,
    ExperimentResults,
    ExperimentRunner,
    PrivSynthError,
    PrivSynthPipeline,
)
from privsynth.experiments import REAL_DATA

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2]
OPTIONS = {"bin_edges": AdultDataLoader.BIN_EDGES}

#: Headline metrics: (column, short label, which direction is better)
HEADLINE = [
    ("marginal.tvd_1way_mean", "1-way TVD", "lower"),
    ("marginal.tvd_2way_mean", "2-way TVD", "lower"),
    ("association.assoc_diff_mean", "Assoc. diff", "lower"),
    ("ml.auc_synthetic", "AUC (TSTR)", "higher"),
    ("ml.utility_ratio", "AUC ratio", "higher"),
    ("dcr.exact_match_synthetic", "Exact copies", "lower"),
    ("dcr.closer_to_train_share", "Closer-to-train", "0.5"),
    ("inference.accuracy_members", "Attack acc.", "lower"),
    ("inference.inference_risk", "Inference risk", "lower"),
]


def main() -> None:
    if "--figures-only" in sys.argv:          # redraw charts from a previous run
        FIGURES.mkdir(parents=True, exist_ok=True)
        make_figures(ExperimentResults.from_csv(RESULTS / "week3_runs.csv"))
        print(f"Redrew figures in {FIGURES}")
        return
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    elif (ROOT / "data" / "adult.data").exists():
        path = ROOT / "data" / "adult.data"
    else:
        path = ROOT / "data" / "sample_adult.csv"

    try:
        dataset = AdultDataLoader().load(path)
    except PrivSynthError as err:
        sys.exit(f"Error: {err}")
    print(dataset, "\n")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # 1. One pipeline run, in detail -------------------------------------
    print("== 1. One pipeline run ==")
    pipeline = PrivSynthPipeline.from_settings("bayesian", epsilon=1.0, random_state=0, **OPTIONS)
    result = pipeline.run(dataset)
    print(result)
    print("\nTimings: " + ", ".join(f"{k} {v:.1f}s" for k, v in result.timings.items()), "\n")

    # 2. The experiment grid ----------------------------------------------
    runner = ExperimentRunner(
        dataset, epsilons=EPSILONS, noises=["laplace", "gaussian"], seeds=SEEDS,
        synthesizer_options=OPTIONS,
    )
    total = len(runner.configs())
    print(f"== 2. Experiment grid: {total} runs + 1 reference "
          f"(train {runner.train.n_rows:,}, holdout {runner.holdout.n_rows:,}) ==")
    start = time.perf_counter()
    results = runner.run(progress=lambda msg: print("  " + msg, flush=True))
    print(f"Finished in {time.perf_counter() - start:.0f}s\n")
    results.to_csv(RESULTS / "week3_runs.csv")
    columns = [m for m, _, _ in HEADLINE]
    results.summary(columns, std=True).to_csv(RESULTS / "week3_summary.csv")

    # 3. Tables -------------------------------------------------------------
    labels = {m: label for m, label, _ in HEADLINE}
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("== 3. Mean over 3 seeds (Laplace noise) ==")
    print("Better: " + ", ".join(f"{label} {d}" for _, label, d in HEADLINE), "\n")
    table = results.summary(columns).rename(columns=labels)
    laplace = table.xs("laplace", level="noise")
    print(laplace.round(3).to_string(), "\n")

    print("Reference: the real training data released as-is")
    base = results.baseline[columns].rename(labels)
    print(base.astype(float).round(3).to_string(), "\n")

    print("== 4. Laplace vs Gaussian noise (mean 2-way TVD and AUC) ==")
    compare = results.summary(["marginal.tvd_2way_mean", "ml.auc_synthetic"]).unstack("noise")
    compare.columns = [f"{labels[m]} ({n})" for m, n in compare.columns]
    print(compare.round(3).to_string(), "\n")

    print("== 5. Spread across seeds (std of AUC, Laplace) ==")
    spread = results.summary(["ml.auc_synthetic"], std=True).xs("laplace", level="noise")
    print(spread.round(3).to_string(), "\n")

    # 4. Figures ------------------------------------------------------------
    make_figures(results)
    print(f"Saved results/week3_runs.csv, results/week3_summary.csv and {len(list(FIGURES.glob('*.png')))} "
          "figures in results/figures/")


def make_figures(results) -> None:
    """One figure per question, plus a 2x2 overview for the report."""
    panels = [
        ("marginal.tvd_2way_mean", "Column pairs kept (2-way TVD)", "TVD (lower is better)", None, None),
        ("ml.auc_synthetic", "Prediction quality (train synthetic, test real)", "AUC (higher is better)",
         0.5, "coin toss"),
        ("inference.inference_risk", "Attribute inference risk", "risk (lower is better)", None, None),
        ("dcr.closer_to_train_share", "Closer to training than to new people", "share (0.5 = no leak)",
         0.5, "no leak"),
    ]
    names = ["fig1_two_way_tvd", "fig2_ml_utility", "fig3_inference_risk", "fig4_closer_to_train"]
    for (metric, title, ylabel, ref, ref_label), name in zip(panels, names):
        fig, ax = plt.subplots(figsize=(6.4, 4), dpi=200)
        results.plot(metric, ax=ax, title=title, ylabel=ylabel, reference=ref, reference_label=ref_label)
        fig.patch.set_facecolor("#fcfcfb")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=200)
    for ax, (metric, title, ylabel, ref, ref_label) in zip(axes.ravel(), panels):
        results.plot(metric, ax=ax, title=title, ylabel=ylabel, reference=ref,
                     reference_label=ref_label, legend=False)
    # One shared legend for the four series; reference lines are labelled on the charts.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    series = [(h, l) for h, l in zip(handles, labels) if l != REAL_DATA]
    fig.legend(*zip(*series), loc="upper left", bbox_to_anchor=(0.01, 0.96), ncol=4,
               frameon=False, fontsize=9, labelcolor="#52514e")
    fig.suptitle("PrivSynth: privacy/utility trade-off on UCI Adult (mean of 3 seeds, band = ±1 std)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(FIGURES / "fig0_tradeoff_overview.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
