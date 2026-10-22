"""The :class:`Dataset` class: a table plus a description of its columns.

A ``Dataset`` wraps a pandas ``DataFrame`` and records which columns are
numeric, which are categorical, which are *quasi-identifiers* (details such
as age, sex and location that can re-identify a person when combined) and
which is the *sensitive attribute* (the value we want to protect, e.g.
income). Every other part of PrivSynth works with ``Dataset`` objects
rather than raw DataFrames, so this schema travels with the data.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from privsynth.exceptions import SchemaError


class Dataset:
    """A tabular dataset with a privacy-aware schema.

    Args:
        data: The table to wrap. A copy is stored, so later changes to the
            caller's DataFrame cannot silently alter the dataset.
        name: Human-readable name shown in reports and the dashboard.
        quasi_identifiers: Columns that could re-identify a person when
            combined. May be set or changed later.
        sensitive_attribute: Column whose value should be protected.

    Raises:
        SchemaError: If the table is empty or a named column is missing.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: str = "dataset",
        quasi_identifiers: Optional[Iterable[str]] = None,
        sensitive_attribute: Optional[str] = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise SchemaError(f"Expected a pandas DataFrame, got {type(data).__name__}.")
        if data.empty:
            raise SchemaError("Cannot create a Dataset from an empty table.")

        self._data = data.copy().reset_index(drop=True)
        self.name = name
        self._quasi_identifiers: list[str] = []
        self._sensitive_attribute: Optional[str] = None

        if quasi_identifiers is not None:
            self.quasi_identifiers = quasi_identifiers
        if sensitive_attribute is not None:
            self.sensitive_attribute = sensitive_attribute

    # ------------------------------------------------------------------ #
    # Read-only access to the underlying table (encapsulation)
    # ------------------------------------------------------------------ #
    @property
    def data(self) -> pd.DataFrame:
        """A copy of the table, so callers cannot modify the dataset in place."""
        return self._data.copy()

    @property
    def columns(self) -> list[str]:
        """Column names in order."""
        return list(self._data.columns)

    @property
    def n_rows(self) -> int:
        """Number of records."""
        return len(self._data)

    @property
    def numeric_columns(self) -> list[str]:
        """Columns holding numbers (treated as continuous)."""
        return list(self._data.select_dtypes(include="number").columns)

    @property
    def categorical_columns(self) -> list[str]:
        """Columns holding text or category values."""
        return [c for c in self._data.columns if c not in self.numeric_columns]

    # ------------------------------------------------------------------ #
    # Privacy schema (validated setters)
    # ------------------------------------------------------------------ #
    @property
    def quasi_identifiers(self) -> list[str]:
        """Columns that could re-identify a person when combined."""
        return list(self._quasi_identifiers)

    @quasi_identifiers.setter
    def quasi_identifiers(self, columns: Iterable[str]) -> None:
        columns = list(columns)
        if not columns:
            raise SchemaError("At least one quasi-identifier is required.")
        self._check_columns_exist(columns)
        if self._sensitive_attribute in columns:
            raise SchemaError(
                f"'{self._sensitive_attribute}' is the sensitive attribute "
                "and cannot also be a quasi-identifier."
            )
        self._quasi_identifiers = columns

    @property
    def sensitive_attribute(self) -> Optional[str]:
        """The column whose value should be protected, if any."""
        return self._sensitive_attribute

    @sensitive_attribute.setter
    def sensitive_attribute(self, column: Optional[str]) -> None:
        if column is not None:
            self._check_columns_exist([column])
            if column in self._quasi_identifiers:
                raise SchemaError(
                    f"'{column}' is a quasi-identifier and cannot also be "
                    "the sensitive attribute."
                )
        self._sensitive_attribute = column

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #
    def select(self, columns: Iterable[str]) -> "Dataset":
        """Return a new Dataset containing only ``columns``.

        The privacy schema is carried over for the columns that remain.
        """
        columns = list(columns)
        self._check_columns_exist(columns)
        kept_qis = [c for c in self._quasi_identifiers if c in columns]
        sensitive = self._sensitive_attribute if self._sensitive_attribute in columns else None
        return Dataset(
            self._data[columns],
            name=self.name,
            quasi_identifiers=kept_qis or None,
            sensitive_attribute=sensitive,
        )

    def sample(self, n: int, random_state: Optional[int] = None) -> "Dataset":
        """Return a random subset of ``n`` records with the same schema."""
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        n = min(n, self.n_rows)
        return self.with_data(self._data.sample(n=n, random_state=random_state))

    def with_data(self, data: pd.DataFrame, name: Optional[str] = None) -> "Dataset":
        """Return a new Dataset with different rows but the same schema.

        Used by generalisation and, later, by the synthesizers so that
        synthetic data keeps the original's quasi-identifiers.
        """
        return Dataset(
            data,
            name=name or self.name,
            quasi_identifiers=self._quasi_identifiers or None,
            sensitive_attribute=self._sensitive_attribute,
        )

    def summary(self) -> pd.DataFrame:
        """One row per column: type, role, missing values and distinct values."""
        rows = []
        for col in self._data.columns:
            if col in self._quasi_identifiers:
                role = "quasi-identifier"
            elif col == self._sensitive_attribute:
                role = "sensitive"
            else:
                role = "other"
            rows.append(
                {
                    "column": col,
                    "type": "numeric" if col in self.numeric_columns else "categorical",
                    "role": role,
                    "missing": int(self._data[col].isna().sum()),
                    "distinct_values": int(self._data[col].nunique()),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Internal helpers and Python protocols
    # ------------------------------------------------------------------ #
    def _check_columns_exist(self, columns: Iterable[str]) -> None:
        missing = [c for c in columns if c not in self._data.columns]
        if missing:
            raise SchemaError(f"Column(s) not found in '{self.name}': {missing}")

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, rows={self.n_rows}, "
            f"columns={len(self.columns)}, quasi_identifiers={self._quasi_identifiers}, "
            f"sensitive_attribute={self._sensitive_attribute!r})"
        )
