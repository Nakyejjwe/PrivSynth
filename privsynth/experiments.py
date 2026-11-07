"""Experiments: measure the privacy/utility trade-off across settings.

:class:`ExperimentRunner` runs the :class:`~privsynth.pipeline.PrivSynthPipeline`
over a grid of settings (generator x noise x epsilon x seed) on **one fixed
train/holdout split**, so every run is judged on the same real people. It
can also add a *reference row*: the real training data "released" as if it
were synthetic. That shows the utility ceiling and the risk of sharing the
raw data, which is exactly what PrivSynth is meant to avoid.

Results come back as an :class:`ExperimentResults` object: a tidy table
(one row per run) with methods to summarise, save and plot it.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.evaluation.base import Evaluator
from privsynth.evaluation.suite import EvaluationSuite, default_evaluators
from privsynth.generation import SYNTHESIZERS
from privsynth.pipeline import PrivSynthPipeline
from privsynth.privacy.mechanisms import NOISE_MECHANISMS

#: Label used for the "release the real data" reference row.
REAL_DATA = "Real data (no protection)"

SETTINGS = ["generator", "noise", "epsilon", "seed"]


@dataclass(frozen=True)
class ExperimentConfig:
    """One cell of the experiment grid."""

    generator: str
    noise: str
    epsilon: float
    seed: int


class ExperimentRunner:
    """Run the pipeline over a grid of privacy settings.

    Args:
        dataset: The real data.
        epsilons: Privacy budgets to try.
        generators: Keys of :data:`~privsynth.generation.SYNTHESIZERS`.
        noises: ``"laplace"`` and/or ``"gaussian"``.
        seeds: Random seeds; each setting is repeated once per seed and the
            spread across seeds shows how much results vary by chance.
        holdout_fraction: Share of records held back for evaluation.
        split_seed: Seed for the one train/holdout split shared by all runs.
        synthesizer_options: Extra synthesizer arguments, e.g. ``bin_edges``.
        evaluators: Factory returning a fresh list of evaluators; defaults
            to :func:`default_evaluators` with the same ``bin_edges``.
        include_real_baseline: Add the "real data released" reference row.
        n_synthetic: Records to generate per run (default: training size).

    Example:
        >>> runner = ExperimentRunner(dataset, epsilons=[0.1, 1, 10], seeds=[0, 1])
        >>> results = runner.run(progress=print)
        >>> results.summary(["ml.auc_synthetic", "inference.inference_risk"])
    """

    def __init__(
        self,
        dataset: Dataset,
        epsilons: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
        generators: Sequence[str] = ("independent", "bayesian"),
        noises: Sequence[str] = ("laplace",),
        seeds: Sequence[int] = (0, 1, 2),
        holdout_fraction: float = 0.2,
        split_seed: int = 0,
        synthesizer_options: Optional[Mapping] = None,
        evaluators: Optional[Callable[[], Sequence[Evaluator]]] = None,
        include_real_baseline: bool = True,
        n_synthetic: Optional[int] = None,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Dataset, got {type(dataset).__name__}.")
        self.epsilons = [float(e) for e in epsilons]
        if not self.epsilons or any(not math.isfinite(e) or e <= 0 for e in self.epsilons):
            raise ValueError(f"epsilons must be positive finite numbers, got {list(epsilons)!r}.")
        unknown = [g for g in generators if g not in SYNTHESIZERS]
        if unknown or not generators:
            raise ValueError(f"Unknown generator(s) {unknown}; choose from {sorted(SYNTHESIZERS)}.")
        unknown = [n for n in noises if n not in NOISE_MECHANISMS]
        if unknown or not noises:
            raise ValueError(f"Unknown noise {unknown}; choose from {sorted(NOISE_MECHANISMS)}.")
        if not seeds:
            raise ValueError("At least one seed is needed.")

        self.dataset = dataset
        self.generators = list(generators)
        self.noises = list(noises)
        self.seeds = [int(s) for s in seeds]
        self.synthesizer_options = dict(synthesizer_options or {})
        self._evaluator_factory = evaluators or (
            lambda: default_evaluators(bin_edges=self.synthesizer_options.get("bin_edges"))
        )
        self.include_real_baseline = include_real_baseline
        self.n_synthetic = n_synthetic
        self.train, self.holdout = dataset.split(holdout_fraction, random_state=split_seed)

    def configs(self) -> list[ExperimentConfig]:
        """Every setting in the grid, in run order."""
        return [
            ExperimentConfig(g, n, e, s)
            for g, n, e, s in itertools.product(self.generators, self.noises, self.epsilons, self.seeds)
        ]

    def run(self, progress: Optional[Callable[[str], None]] = None) -> "ExperimentResults":
        """Run every setting and return the results table.

        Args:
            progress: Called with a one-line message after each run (e.g. ``print``).
        """
        rows = []
        configs = self.configs()
        if self.include_real_baseline:
            rows.append(self._real_baseline())
            _report(progress, f"[reference] {REAL_DATA} evaluated")
        for i, cfg in enumerate(configs, start=1):
            start = time.perf_counter()
            synthesizer = SYNTHESIZERS[cfg.generator](
                epsilon=cfg.epsilon, noise=cfg.noise, random_state=cfg.seed, **self.synthesizer_options
            )
            pipeline = PrivSynthPipeline(synthesizer, evaluators=self._evaluator_factory())
            result = pipeline.run_on_split(self.train, self.holdout, n=self.n_synthetic)
            row = {"generator": synthesizer.name, "noise": cfg.noise,
                   "epsilon": cfg.epsilon, "seed": cfg.seed}
            row.update(result.evaluation.flat())
            row["seconds"] = time.perf_counter() - start
            rows.append(row)
            _report(progress, f"[{i}/{len(configs)}] {synthesizer.name}, {cfg.noise}, "
                              f"eps={cfg.epsilon:g}, seed={cfg.seed} ({row['seconds']:.1f}s)")
        return ExperimentResults(pd.DataFrame(rows))

    def _real_baseline(self) -> dict:
        """Evaluate the real training data as if it had been released."""
        suite = EvaluationSuite(self._evaluator_factory())
        start = time.perf_counter()
        report = suite.run(self.train, self.train, self.holdout)
        row = {"generator": REAL_DATA, "noise": "none", "epsilon": math.inf, "seed": -1}
        row.update(report.flat())
        row["seconds"] = time.perf_counter() - start
        return row


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)


class ExperimentResults:
    """A tidy table of experiment runs, with summaries and plots.

    Args:
        runs: One row per run, with columns ``generator``, ``noise``,
            ``epsilon``, ``seed`` and one column per metric.
    """

    #: Series styling: colour follows the generator, line style the noise
    #: (so the two noise types are told apart without relying on colour).
    COLORS = {"Independent marginals": "#2a78d6", "Bayesian network": "#eb6834"}
    LINESTYLES = {"laplace": "-", "gaussian": "--"}
    MARKERS = {"laplace": "o", "gaussian": "s"}
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

    def __init__(self, runs: pd.DataFrame) -> None:
        missing = [c for c in SETTINGS if c not in runs.columns]
        if missing:
            raise ValueError(f"runs is missing column(s) {missing}.")
        self._runs = runs.reset_index(drop=True)

    @property
    def runs(self) -> pd.DataFrame:
        """A copy of the full table (one row per run)."""
        return self._runs.copy()

    @property
    def metrics(self) -> list[str]:
        """Names of the metric columns."""
        return [c for c in self._runs.columns if "." in c]

    @property
    def baseline(self) -> Optional[pd.Series]:
        """The "real data released" reference row, if it was run."""
        rows = self._runs[self._runs["generator"] == REAL_DATA]
        return rows.iloc[0] if len(rows) else None

    def summary(self, metrics: Optional[Sequence[str]] = None, std: bool = False) -> pd.DataFrame:
        """Mean of each metric over seeds, per generator, noise and epsilon.

        Args:
            metrics: Columns to include (default: all metrics).
            std: Also include the standard deviation across seeds.
        """
        metrics = list(metrics) if metrics is not None else self.metrics
        unknown = [m for m in metrics if m not in self._runs.columns]
        if unknown:
            raise KeyError(f"Unknown metric(s): {unknown}")
        grouped = self._runs.groupby(["generator", "noise", "epsilon"], sort=False)[metrics]
        if not std:
            return grouped.mean()
        table = grouped.agg(["mean", "std"])
        table.columns = [f"{m} ({stat})" for m, stat in table.columns]
        return table

    def to_csv(self, path: Union[str, Path]) -> None:
        """Save every run to a CSV file."""
        self._runs.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "ExperimentResults":
        """Load results saved with :meth:`to_csv`."""
        return cls(pd.read_csv(path))

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot(self, metric: str, ax=None, title: Optional[str] = None, ylabel: Optional[str] = None,
             reference: Optional[float] = None, reference_label: Optional[str] = None,
             legend: bool = True):
        """Plot ``metric`` against epsilon: one line per generator and noise.

        Lines show the mean over seeds; the shaded band is one standard
        deviation (never drawn below 0 for metrics that cannot be negative).
        The real-data reference row, if present, is a dotted grey line, and
        ``reference`` adds another horizontal line, e.g. 0.5 for "a coin
        toss". Both are labelled directly on the chart.

        Args:
            legend: Draw a legend on this axes (turn off when a figure
                shares one legend across several panels).

        Returns:
            The matplotlib ``Axes``.
        """
        import matplotlib.pyplot as plt

        if metric not in self._runs.columns:
            raise KeyError(f"Unknown metric {metric!r}.")
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.8))
        synthetic = self._runs[self._runs["generator"] != REAL_DATA]
        stats = synthetic.groupby(["generator", "noise", "epsilon"], sort=False)[metric].agg(["mean", "std"])
        non_negative = bool((self._runs[metric].dropna() >= 0).all())

        for (generator, noise), frame in stats.groupby(level=[0, 1], sort=False):
            frame = frame.droplevel([0, 1]).sort_index()
            color = self.COLORS.get(generator, self.MUTED)
            x, mean, spread = frame.index.to_numpy(), frame["mean"].to_numpy(), frame["std"].fillna(0).to_numpy()
            low = (mean - spread).clip(min=0) if non_negative else mean - spread
            ax.fill_between(x, low, mean + spread, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=color, linestyle=self.LINESTYLES.get(noise, "-"), linewidth=2,
                    marker=self.MARKERS.get(noise, "o"), markersize=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                    label=f"{generator} ({noise})" if len(self.noises_used) > 1 else generator)

        lines = []          # (y, label, place the label above the line?)
        base = self.baseline
        if base is not None and not pd.isna(base.get(metric)):
            ax.axhline(base[metric], color=self.MUTED, linestyle=":", linewidth=1.5, label=REAL_DATA)
            lines.append((float(base[metric]), "real data released", True))
        if reference is not None:
            ax.axhline(reference, color="#b5b3ad", linestyle="-", linewidth=1.2, zorder=0,
                       label=reference_label)
            if reference_label:
                lines.append((float(reference), reference_label, False))
        if lines:           # headroom so a label above the top line clears the title
            low, high = ax.get_ylim()
            top = max(y for y, _, above in lines if above) if any(a for *_, a in lines) else high
            ax.set_ylim(low, max(high, top + 0.1 * (max(high, top) - low)))
        for y, text, above in lines:    # direct labels: real data top-left, reference bottom-right
            ax.annotate(text, xy=(0 if above else 1, y), xycoords=("axes fraction", "data"),
                        xytext=(4 if above else -4, 3 if above else -3), textcoords="offset points",
                        fontsize=7.5, color=self.MUTED, va="bottom" if above else "top",
                        ha="left" if above else "right")

        ax.set_xscale("log")
        epsilons = sorted(synthetic["epsilon"].unique())
        ax.set_xticks(epsilons)
        ax.set_xticklabels([f"{e:g}" for e in epsilons])
        ax.minorticks_off()
        ax.set_xlabel("privacy budget ε (smaller = more private)", color=self.MUTED)
        ax.set_ylabel(ylabel or metric, color=self.MUTED)
        if title:
            ax.set_title(title, loc="left", color=self.INK, fontsize=11, fontweight="bold")
        _style_axes(ax, self.INK, self.MUTED, self.GRID)
        if legend:      # series only (reference lines are labelled on the chart), below the plot
            handles, names = ax.get_legend_handles_labels()
            keep = [(h, n) for h, n in zip(handles, names) if n not in (REAL_DATA, reference_label)]
            if keep:
                ax.legend(*zip(*keep), frameon=False, fontsize=8, labelcolor=self.MUTED, ncol=2,
                          loc="upper left", bbox_to_anchor=(0, -0.2))
        return ax

    @property
    def noises_used(self) -> list[str]:
        synthetic = self._runs[self._runs["generator"] != REAL_DATA]
        return list(dict.fromkeys(synthetic["noise"]))

    def __len__(self) -> int:
        return len(self._runs)

    def __repr__(self) -> str:
        return f"ExperimentResults(runs={len(self._runs)}, metrics={len(self.metrics)})"


def _style_axes(ax, ink: str, muted: str, grid: str) -> None:
    """Recessive axes: no top/right spines, light horizontal grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fcfcfb")
