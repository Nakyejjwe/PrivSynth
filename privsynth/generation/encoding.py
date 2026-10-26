"""Turn a mixed table into small integer codes, and back again.

The generators count how often each value (or pair of values) occurs.
That needs every column to take a small, fixed set of values, so:

* a **categorical** column becomes codes ``0 .. k-1`` (one per category);
* a **numeric** column is split into bins (ranges), and each value becomes
  its bin number.

Decoding reverses this. A numeric bin is turned back into a number by
drawing uniformly inside the bin, so synthetic ages look like ages rather
than bin labels.

Privacy note: the encoder reads the *domain* of each column (its
categories and its min/max) from the data. Like most DP synthesizers, we
treat the domain as public knowledge (for Adult it is published with the
dataset). To avoid even that, pass ``bin_edges`` and ``categories``
explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.exceptions import NotFittedError, SchemaError


@dataclass(frozen=True)
class ColumnDomain:
    """How one column is encoded.

    Attributes:
        name: Column name.
        kind: ``"categorical"`` or ``"numeric"``.
        size: Number of possible codes.
        categories: Category labels (categorical columns only).
        edges: Bin edges, ``size + 1`` of them (numeric columns only).
        integer: True if the original numeric column held whole numbers.
    """

    name: str
    kind: str
    size: int
    categories: Optional[tuple] = None
    edges: Optional[np.ndarray] = None
    integer: bool = False


class TableEncoder:
    """Encode a :class:`Dataset` as integer codes and decode codes back.

    Args:
        n_bins: Default number of equal-width bins for numeric columns.
        bin_edges: Column -> explicit, increasing bin edges. Useful for
            skewed columns such as ``capital_gain``, which is mostly 0.
        categories: Column -> full list of allowed categories.
        max_exact_integers: Integer columns with at most this many possible
            values (e.g. ``education_num`` 1..16) get one bin per value, so
            they are reproduced exactly.
    """

    def __init__(
        self,
        n_bins: int = 16,
        bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
        categories: Optional[Mapping[str, Sequence]] = None,
        max_exact_integers: int = 32,
    ) -> None:
        if not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}.")
        self.n_bins = n_bins
        self.bin_edges = {col: np.asarray(e, dtype=float) for col, e in (bin_edges or {}).items()}
        for col, edges in self.bin_edges.items():
            if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError(f"bin_edges for '{col}' must be at least two strictly increasing numbers.")
        self.categories = {col: list(c) for col, c in (categories or {}).items()}
        self.max_exact_integers = max_exact_integers
        self._domains: dict[str, ColumnDomain] = {}

    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return bool(self._domains)

    @property
    def domains(self) -> dict[str, ColumnDomain]:
        """Column name -> :class:`ColumnDomain`, in column order."""
        self._require_fitted()
        return dict(self._domains)

    @property
    def columns(self) -> list[str]:
        self._require_fitted()
        return list(self._domains)

    @property
    def sizes(self) -> dict[str, int]:
        """Number of codes per column."""
        return {name: d.size for name, d in self.domains.items()}

    # ------------------------------------------------------------------ #
    def fit(self, dataset: Dataset) -> "TableEncoder":
        """Work out each column's domain from ``dataset``."""
        frame = dataset.data
        numeric = set(dataset.numeric_columns)
        domains: dict[str, ColumnDomain] = {}
        for col in frame.columns:
            if col in numeric:
                domains[col] = self._numeric_domain(col, frame[col])
            else:
                domains[col] = self._categorical_domain(col, frame[col])
        self._domains = domains
        return self

    def encode(self, dataset: Dataset) -> pd.DataFrame:
        """Return a DataFrame of integer codes with the same columns.

        Raises:
            SchemaError: If columns differ from those seen in :meth:`fit`, or
                a category is not in the domain.
        """
        self._require_fitted()
        frame = dataset.data
        if list(frame.columns) != self.columns:
            raise SchemaError("Dataset columns do not match the columns the encoder was fitted on.")
        codes = {}
        for col, dom in self._domains.items():
            if dom.kind == "categorical":
                text = frame[col].astype(str)
                lookup = {cat: i for i, cat in enumerate(dom.categories)}
                unknown = sorted(set(text) - set(lookup))
                if unknown:
                    raise SchemaError(f"Column '{col}' has values outside its domain: {unknown[:5]}")
                codes[col] = text.map(lookup).to_numpy(dtype=np.int64)
            else:
                values = frame[col].to_numpy(dtype=float)
                # digitize against inner edges -> 0..size-1; values outside are clipped
                codes[col] = np.clip(np.digitize(values, dom.edges[1:-1], right=False), 0, dom.size - 1)
        return pd.DataFrame(codes, columns=self.columns)

    def decode(self, codes: pd.DataFrame, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Turn integer codes back into a table of real-looking values."""
        self._require_fitted()
        rng = rng or np.random.default_rng()
        out = {}
        for col, dom in self._domains.items():
            c = codes[col].to_numpy(dtype=np.int64)
            if c.size and (c.min() < 0 or c.max() >= dom.size):
                raise SchemaError(f"Codes for '{col}' are outside 0..{dom.size - 1}.")
            if dom.kind == "categorical":
                out[col] = pd.Series(np.asarray(dom.categories, dtype=object)[c], dtype="string")
            else:
                low, high = dom.edges[c], dom.edges[c + 1]
                values = rng.uniform(low, high)
                if dom.integer:
                    values = np.floor(values).astype(np.int64)
                out[col] = values
        return pd.DataFrame(out, columns=self.columns)

    # ------------------------------------------------------------------ #
    def _numeric_domain(self, col: str, series: pd.Series) -> ColumnDomain:
        values = series.to_numpy(dtype=float)
        integer = bool(np.all(np.mod(values, 1) == 0))
        if col in self.bin_edges:
            edges = self.bin_edges[col]
        else:
            low, high = float(values.min()), float(values.max())
            if integer and high - low + 1 <= self.max_exact_integers:
                edges = np.arange(low, high + 2)          # one bin per whole number
            elif high == low:
                edges = np.array([low, low + 1.0])
            else:
                edges = np.linspace(low, high, self.n_bins + 1)
                if integer:
                    edges[-1] = high + 1                 # so floor() can reach the max value
        return ColumnDomain(col, "numeric", size=len(edges) - 1, edges=edges, integer=integer)

    def _categorical_domain(self, col: str, series: pd.Series) -> ColumnDomain:
        observed = sorted(series.astype(str).unique())
        allowed = self.categories.get(col)
        cats = list(allowed) if allowed is not None else observed
        cats = [str(c) for c in cats]
        return ColumnDomain(col, "categorical", size=len(cats), categories=tuple(cats))

    def _require_fitted(self) -> None:
        if not self._domains:
            raise NotFittedError("TableEncoder has not been fitted; call fit(dataset) first.")
