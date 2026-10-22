"""Re-identification risk: generalisation and the k-anonymity checker.

**k-anonymity** [Sweeney, 2002] asks: for each record, how many other records
share exactly the same quasi-identifier values (e.g. same age, sex, race)?
Records sharing values form an *equivalence class*. A dataset is
k-anonymous when every equivalence class has at least ``k`` records, so an
attacker who knows someone's quasi-identifiers can narrow them down to no
fewer than ``k`` people. A record alone in its class (size 1) is *unique*
and can be re-identified with certainty.

Each record's re-identification risk is ``1 / class size`` (the chance an
attacker picks the right person by guessing within the class).

**l-diversity** adds a check on the sensitive attribute: if everyone in a
class has the same income, knowing someone is in that class reveals their
income even though they are "hidden among k". We report the smallest number
of distinct sensitive values in any class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.exceptions import SchemaError


class Generalizer:
    """Make quasi-identifiers coarser so that more records look alike.

    Generalisation is the classic way to raise k: ages become ranges
    (``37`` -> ``"30-39"``), and rare categories are merged into ``"Other"``.

    Args:
        bin_widths: Numeric column -> width of each range, e.g. ``{"age": 10}``.
        rare_threshold: Category values appearing fewer than this many times
            are replaced by ``"Other"``. ``0`` disables merging.
        columns: Restrict merging of rare categories to these columns
            (defaults to the dataset's categorical quasi-identifiers).
    """

    OTHER = "Other"

    def __init__(
        self,
        bin_widths: Optional[Mapping[str, int]] = None,
        rare_threshold: int = 0,
        columns: Optional[Iterable[str]] = None,
    ) -> None:
        self.bin_widths = dict(bin_widths or {})
        for col, width in self.bin_widths.items():
            if not isinstance(width, (int, np.integer)) or width <= 0:
                raise ValueError(f"Bin width for '{col}' must be a positive integer, got {width!r}.")
        if rare_threshold < 0:
            raise ValueError("rare_threshold cannot be negative.")
        self.rare_threshold = rare_threshold
        self.columns = list(columns) if columns is not None else None

    def generalize(self, dataset: Dataset) -> Dataset:
        """Return a new, generalised Dataset (the original is unchanged)."""
        frame = dataset.data

        for col, width in self.bin_widths.items():
            if col not in frame.columns:
                raise SchemaError(f"Cannot bin '{col}': column not in dataset.")
            if not pd.api.types.is_numeric_dtype(frame[col]):
                raise SchemaError(f"Cannot bin '{col}': it is not numeric.")
            lower = (frame[col] // width * width).astype(int)
            frame[col] = lower.astype(str) + "-" + (lower + width - 1).astype(str)

        if self.rare_threshold > 0:
            targets = self.columns
            if targets is None:
                targets = [c for c in dataset.quasi_identifiers if c in dataset.categorical_columns]
            for col in targets:
                counts = frame[col].value_counts()
                rare = counts[counts < self.rare_threshold].index
                if len(rare):
                    frame[col] = frame[col].astype("string").where(~frame[col].isin(rare), self.OTHER)

        return dataset.with_data(frame, name=f"{dataset.name} (generalised)")


@dataclass
class RiskReport:
    """The result of a k-anonymity check.

    Attributes:
        k_target: The k the checker was asked to satisfy.
        k_achieved: The size of the smallest equivalence class.
        n_records: Number of records checked.
        n_classes: Number of distinct quasi-identifier combinations.
        n_unique: Records alone in their class (certain re-identification).
        n_at_risk: Records in classes smaller than ``k_target``.
        average_risk: Mean of ``1 / class size`` over all records.
        max_risk: Highest single-record risk (1.0 if any record is unique).
        l_diversity: Fewest distinct sensitive values in any class, or
            ``None`` if no sensitive attribute was set.
        quasi_identifiers: The columns used.
        record_class_sizes: Class size for every record (same order as data).
    """

    k_target: int
    k_achieved: int
    n_records: int
    n_classes: int
    n_unique: int
    n_at_risk: int
    average_risk: float
    max_risk: float
    l_diversity: Optional[int]
    quasi_identifiers: list[str]
    record_class_sizes: pd.Series = field(repr=False)

    @property
    def is_k_anonymous(self) -> bool:
        """True when every equivalence class has at least ``k_target`` records."""
        return self.k_achieved >= self.k_target

    @property
    def pct_unique(self) -> float:
        """Percentage of records that are unique."""
        return 100.0 * self.n_unique / self.n_records

    @property
    def pct_at_risk(self) -> float:
        """Percentage of records in classes smaller than ``k_target``."""
        return 100.0 * self.n_at_risk / self.n_records

    @property
    def record_risks(self) -> pd.Series:
        """Re-identification risk (``1 / class size``) for each record."""
        return (1.0 / self.record_class_sizes).rename("risk")

    def to_dict(self) -> dict:
        """Headline numbers as a plain dict (for tables, JSON and the dashboard)."""
        return {
            "k_target": self.k_target,
            "k_achieved": self.k_achieved,
            "k_anonymous": self.is_k_anonymous,
            "records": self.n_records,
            "equivalence_classes": self.n_classes,
            "unique_records": self.n_unique,
            "pct_unique": round(self.pct_unique, 2),
            "records_at_risk": self.n_at_risk,
            "pct_at_risk": round(self.pct_at_risk, 2),
            "average_risk": round(self.average_risk, 4),
            "max_risk": round(self.max_risk, 4),
            "l_diversity": self.l_diversity,
        }

    def __str__(self) -> str:
        verdict = "PASSES" if self.is_k_anonymous else "FAILS"
        lines = [
            f"k-anonymity check (k = {self.k_target}) on {', '.join(self.quasi_identifiers)}: {verdict}",
            f"  smallest group size (k achieved): {self.k_achieved}",
            f"  unique records:  {self.n_unique:,} of {self.n_records:,} ({self.pct_unique:.1f}%)",
            f"  records at risk: {self.n_at_risk:,} ({self.pct_at_risk:.1f}%)",
            f"  average risk:    {self.average_risk:.3f}   max risk: {self.max_risk:.3f}",
        ]
        if self.l_diversity is not None:
            lines.append(f"  l-diversity:     {self.l_diversity}")
        return "\n".join(lines)


class KAnonymityChecker:
    """Measure each record's re-identification risk.

    Args:
        k: The minimum acceptable equivalence-class size (commonly 5 or 10).

    Example:
        >>> checker = KAnonymityChecker(k=5)
        >>> report = checker.evaluate(dataset)
        >>> print(report)
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    @property
    def k(self) -> int:
        """The minimum acceptable equivalence-class size."""
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"k must be a positive integer, got {value!r}.")
        self._k = int(value)

    def evaluate(
        self,
        dataset: Dataset,
        quasi_identifiers: Optional[Iterable[str]] = None,
    ) -> RiskReport:
        """Check ``dataset`` and return a :class:`RiskReport`.

        Args:
            dataset: The data to check.
            quasi_identifiers: Columns to use; defaults to the dataset's own.

        Raises:
            SchemaError: If no quasi-identifiers are available or one is missing.
        """
        qis = list(quasi_identifiers) if quasi_identifiers is not None else dataset.quasi_identifiers
        if not qis:
            raise SchemaError("No quasi-identifiers given; set them on the Dataset or pass them in.")
        missing = [c for c in qis if c not in dataset.columns]
        if missing:
            raise SchemaError(f"Quasi-identifier column(s) not in dataset: {missing}")

        frame = dataset.data
        groups = frame.groupby(qis, dropna=False, observed=True)
        class_sizes = groups[qis[0]].transform("size").astype(int).rename("class_size")

        l_div = None
        sensitive = dataset.sensitive_attribute
        if sensitive is not None and sensitive not in qis:
            l_div = int(groups[sensitive].nunique(dropna=False).min())

        return RiskReport(
            k_target=self.k,
            k_achieved=int(class_sizes.min()),
            n_records=len(frame),
            n_classes=int(groups.ngroups),
            n_unique=int((class_sizes == 1).sum()),
            n_at_risk=int((class_sizes < self.k).sum()),
            average_risk=float((1.0 / class_sizes).mean()),
            max_risk=float(1.0 / class_sizes.min()),
            l_diversity=l_div,
            quasi_identifiers=qis,
            record_class_sizes=class_sizes,
        )

    def risky_records(self, dataset: Dataset, report: Optional[RiskReport] = None) -> pd.DataFrame:
        """Return the records in classes smaller than k, with their risk score."""
        report = report or self.evaluate(dataset)
        frame = dataset.data
        frame["class_size"] = report.record_class_sizes
        frame["risk"] = report.record_risks
        return frame[frame["class_size"] < self.k].sort_values("risk", ascending=False)

    def __repr__(self) -> str:
        return f"KAnonymityChecker(k={self.k})"
