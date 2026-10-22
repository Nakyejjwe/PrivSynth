"""Loaders that turn files into :class:`~privsynth.data.dataset.Dataset` objects.

``DataLoader`` is an abstract base class: it fixes *what* every loader does
(``load`` a source into a Dataset) while each subclass decides *how*.
``CSVLoader`` handles any CSV a user uploads in the dashboard;
``AdultDataLoader`` knows the quirks of the UCI Adult dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Iterable, Optional, Union

import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.exceptions import DataLoadError

Source = Union[str, Path, IO]


class DataLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, drop_missing: bool = True) -> None:
        self.drop_missing = drop_missing

    @abstractmethod
    def _read(self, source: Source) -> pd.DataFrame:
        """Read the raw table from ``source`` (implemented by subclasses)."""

    def load(
        self,
        source: Source,
        name: Optional[str] = None,
        quasi_identifiers: Optional[Iterable[str]] = None,
        sensitive_attribute: Optional[str] = None,
    ) -> Dataset:
        """Read ``source`` and return a cleaned :class:`Dataset`.

        This is a *template method*: the shared steps (error handling,
        cleaning, building the Dataset) live here, and only ``_read`` and
        the defaults differ between subclasses.

        Raises:
            DataLoadError: If the file is missing, unreadable or empty.
        """
        if isinstance(source, (str, Path)) and not Path(source).exists():
            raise DataLoadError(f"File not found: {source}")
        try:
            frame = self._read(source)
        except DataLoadError:
            raise
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
            raise DataLoadError(f"Could not read {source}: {exc}") from exc

        frame = self._clean(frame)
        if frame.empty:
            raise DataLoadError(f"No usable records found in {source}.")

        return Dataset(
            frame,
            name=name or self._default_name(source),
            quasi_identifiers=quasi_identifiers or self.default_quasi_identifiers(frame),
            sensitive_attribute=sensitive_attribute or self.default_sensitive_attribute(frame),
        )

    # Hooks that subclasses may override ---------------------------------
    def _clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Trim whitespace in text columns and optionally drop incomplete rows."""
        frame = frame.copy()
        for col in frame.select_dtypes(include=["object", "string"]).columns:
            frame[col] = frame[col].astype("string").str.strip()
        if self.drop_missing:
            frame = frame.dropna()
        return frame.reset_index(drop=True)

    def default_quasi_identifiers(self, frame: pd.DataFrame) -> Optional[list[str]]:
        """Quasi-identifiers to use when the caller does not name any."""
        return None

    def default_sensitive_attribute(self, frame: pd.DataFrame) -> Optional[str]:
        """Sensitive attribute to use when the caller does not name one."""
        return None

    @staticmethod
    def _default_name(source: Source) -> str:
        if isinstance(source, (str, Path)):
            return Path(source).stem
        return getattr(source, "name", "uploaded_data")


class CSVLoader(DataLoader):
    """Load any CSV file that has a header row."""

    def __init__(self, drop_missing: bool = True, missing_values: Iterable[str] = ("?", "")) -> None:
        super().__init__(drop_missing=drop_missing)
        self.missing_values = list(missing_values)

    def _read(self, source: Source) -> pd.DataFrame:
        return pd.read_csv(source, na_values=self.missing_values, skipinitialspace=True)


class AdultDataLoader(DataLoader):
    """Load the UCI Adult (Census Income) dataset.

    The raw ``adult.data`` file has no header row, marks missing values with
    ``?`` and puts a space after every comma; ``adult.test`` also has a
    comment on its first line and a trailing ``.`` on income labels. This
    loader handles all of that and sets sensible privacy defaults.
    """

    COLUMNS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    QUASI_IDENTIFIERS = ["age", "sex", "race", "marital_status", "native_country"]
    SENSITIVE_ATTRIBUTE = "income"

    def _read(self, source: Source) -> pd.DataFrame:
        frame = pd.read_csv(
            source,
            header=None,
            names=self.COLUMNS,
            na_values="?",
            skipinitialspace=True,
            comment="|",          # first line of adult.test is "|1x3 Cross validator"
        )
        # Files saved with a header row (e.g. our sample) repeat the names in row 0
        if not frame.empty and str(frame.iloc[0, 0]).strip() == "age":
            frame = frame.iloc[1:].copy()
            for col in ["age", "fnlwgt", "education_num", "capital_gain",
                        "capital_loss", "hours_per_week"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if len(frame.columns) != len(self.COLUMNS):
            raise DataLoadError("File does not have the 15 Adult dataset columns.")
        frame["income"] = frame["income"].astype("string").str.strip().str.rstrip(".")
        return frame

    def default_quasi_identifiers(self, frame: pd.DataFrame) -> list[str]:
        return list(self.QUASI_IDENTIFIERS)

    def default_sensitive_attribute(self, frame: pd.DataFrame) -> str:
        return self.SENSITIVE_ATTRIBUTE
