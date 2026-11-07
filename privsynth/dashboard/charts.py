"""Matplotlib figures for the dashboard and the report.

Every function returns a ``matplotlib.figure.Figure`` and never calls
``plt.show()``, so the same charts work in Streamlit (``st.pyplot``), in
scripts (``fig.savefig``) and in tests. Styling follows the Week 3
figures: real data in blue, synthetic in orange, recessive axes.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from privsynth.data.dataset import Dataset
from privsynth.privacy.risk import RiskReport

REAL, SYNTHETIC = "#2a78d6", "#eb6834"
AT_RISK, SAFE = "#d03b3b", "#9fa7b3"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAX_CATEGORIES = 12


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def _figure(width: float = 7, height: float = 3.4) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def group_size_chart(report: RiskReport) -> Figure:
    """How many people share their quasi-identifier combination with how many others.

    Bars for groups smaller than k (at risk) are red and labelled; the rest grey.
    """
    k = report.k_target
    edges = [1, 2, k, 10, 50, 100, np.inf]
    edges = sorted(set(e for e in edges if e >= 1))
    sizes = report.record_class_sizes
    labels, counts, risky = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        count = int(((sizes >= low) & (sizes < high)).sum())
        if high == np.inf:
            labels.append(f"{int(low)}+")
        elif high - low == 1:
            labels.append(f"{int(low)}")
        else:
            labels.append(f"{int(low)}–{int(high) - 1}")
        counts.append(count)
        risky.append(high <= k)

    fig, ax = _figure(width=9, height=3.2)
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, width=0.7, color=[AT_RISK if r else SAFE for r in risky])
    for bar, count in zip(bars, counts):
        ax.annotate(f"{count:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlabel("size of the group sharing a person's details (1 = unique)", color=MUTED)
    ax.set_ylabel("people", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylim(0, max(counts) * 1.15 if max(counts) else 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(handles=[Patch(color=AT_RISK, label=f"at risk (group < k = {k})"),
                       Patch(color=SAFE, label="safer")],
              frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    fig.tight_layout()
    return fig


def distribution_chart(
    real: Dataset,
    synthetic: Dataset,
    column: str,
    bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
    n_bins: int = 16,
) -> Figure:
    """Share of records in each category (or numeric range): real vs synthetic.

    Raises:
        KeyError: If ``column`` is missing from either dataset.
    """
    if column not in real.columns or column not in synthetic.columns:
        raise KeyError(f"Column '{column}' is not in both datasets.")
    r, s = real.data[column], synthetic.data[column]
    if column in real.numeric_columns:
        values = pd.concat([r, s]).astype(float)
        edges = np.asarray((bin_edges or {}).get(column, []), dtype=float)
        if edges.size < 2:
            edges = np.histogram_bin_edges(values, bins=n_bins)
        else:
            edges = edges.copy()
            edges[0] = min(edges[0], values.min())
            edges[-1] = max(edges[-1], values.max())
        rc, _ = np.histogram(r.astype(float), bins=edges)
        sc, _ = np.histogram(s.astype(float), bins=edges)
        labels = [f"{_short(a)}–{_short(b)}" for a, b in zip(edges[:-1], edges[1:])]
    else:
        r, s = r.astype(str), s.astype(str)
        top = r.value_counts().index[:MAX_CATEGORIES].tolist()
        def counts(series: pd.Series) -> np.ndarray:
            grouped = series.where(series.isin(top), "Other")
            return grouped.value_counts().reindex(top + ["Other"], fill_value=0).to_numpy()
        rc, sc = counts(r), counts(s)
        labels = top + ["Other"]
        if rc[-1] == 0 and sc[-1] == 0:
            rc, sc, labels = rc[:-1], sc[:-1], labels[:-1]

    rp = rc / max(rc.sum(), 1)
    sp = sc / max(sc.sum(), 1)
    fig, ax = _figure(width=9, height=3.2)
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2 - 0.01, rp, width, color=REAL, label=f"Real ({real.n_rows:,})")
    ax.bar(x + width / 2 + 0.01, sp, width, color=SYNTHETIC, label=f"Synthetic ({synthetic.n_rows:,})")
    ax.set_xticks(x, labels, rotation=35 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of records", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title(column, loc="left", fontsize=10, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    return fig


def association_chart(details: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Figure:
    """Side-by-side heatmaps of Cramér's V between columns: real and synthetic.

    Args:
        details: The ``details`` table of an AssociationSimilarity result.
        columns: Column order; defaults to the order in ``details``.
    """
    from privsynth.evaluation.utility import AssociationSimilarity

    real = AssociationSimilarity.matrix(details, "real_v")
    syn = AssociationSimilarity.matrix(details, "synthetic_v")
    if columns is not None:
        real, syn = real.loc[columns, columns], syn.loc[columns, columns]
    n = len(real)
    size = max(3.2, 0.22 * n + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 0.8, size), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, matrix, title in zip(axes, (real, syn), ("Real data", "Synthetic data")):
        shown = matrix.to_numpy(copy=True)
        np.fill_diagonal(shown, np.nan)
        image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), matrix.columns, rotation=90, fontsize=8, color=MUTED)
        ax.set_yticks(range(n), matrix.index, fontsize=8, color=MUTED)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Cramér's V (0 = unrelated, 1 = one decides the other)", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.outline.set_visible(False)
    return fig


def _short(value: float) -> str:
    """Compact axis label: 1500 -> '1.5k', 37.0 -> '37'."""
    if abs(value) >= 1000:
        return f"{value / 1000:.3g}k"
    return f"{value:.3g}"
