"""Where the dashboard's data comes from, and a first guess at its schema.

A :class:`DataSource` knows how to produce a :class:`~privsynth.data.Dataset`.
The dashboard holds one and calls :meth:`DataSource.load` without caring
which kind it is (polymorphism):

* :class:`SampleDataSource` - the bundled UCI Adult data, for a quick demo.
* :class:`UploadedCSVSource` - a CSV file the user uploads.

Uploaded files have no privacy schema, so :func:`suggest_schema` makes a
first guess from column names and values: which columns *directly*
identify people (names, phone numbers - these must be removed), which are
quasi-identifiers (age, sex, district), and which one is sensitive. The
user can then correct it in the dashboard.
"""

from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from privsynth.data.dataset import Dataset
from privsynth.data.loaders import AdultDataLoader, CSVLoader
from privsynth.exceptions import DataLoadError

#: Largest upload the dashboard accepts, in megabytes.
MAX_UPLOAD_MB = 50

# Name patterns (matched against lower-cased column names split into words).
_DIRECT_ID_WORDS = {
    "id", "name", "names", "firstname", "lastname", "surname", "email", "phone", "mobile",
    "telephone", "tel", "contact", "nin", "ssn", "passport", "address", "street", "account",
}
_QI_WORDS = {
    "age", "dob", "birth", "birthdate", "yob", "sex", "gender", "race", "ethnicity", "ethnic",
    "tribe", "marital", "country", "nationality", "district", "region", "county", "subcounty",
    "parish", "village", "location", "zip", "zipcode", "postcode", "religion",
}
_SENSITIVE_WORDS = {
    "income", "salary", "wage", "wages", "diagnosis", "disease", "hiv", "outcome", "condition",
    "result", "target", "label", "class",
}
_MAX_SENSITIVE_VALUES = 20


@dataclass
class SchemaSuggestion:
    """A first guess at an uploaded table's privacy schema.

    Attributes:
        direct_identifiers: Columns that identify people on their own and
            should be removed before anything else happens.
        quasi_identifiers: Columns that could identify people in combination.
        sensitive_attribute: The column to protect, or None if unclear.
        reasons: Column -> short explanation of why it was flagged.
    """

    direct_identifiers: list[str] = field(default_factory=list)
    quasi_identifiers: list[str] = field(default_factory=list)
    sensitive_attribute: Optional[str] = None
    reasons: dict[str, str] = field(default_factory=dict)


def _words(column: str) -> set[str]:
    """Split a column name into lower-case words: ``"Marital-Status"`` -> {"marital", "status"}."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column))
    return {w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w}


def suggest_schema(frame: pd.DataFrame) -> SchemaSuggestion:
    """Guess direct identifiers, quasi-identifiers and the sensitive attribute.

    The rules are deliberately simple and explainable:

    * a column is a **direct identifier** if its name looks like one (id,
      name, phone, ...) or it is a text column where almost every value is
      different (as names and ID numbers are);
    * a column is a **quasi-identifier** if its name is a typical
      demographic (age, sex, district, ...);
    * the **sensitive attribute** is the first column whose name suggests
      an outcome (income, diagnosis, ...) and that has 2-20 distinct
      values; failing that, the last such column in the table.
    """
    suggestion = SchemaSuggestion()
    n = len(frame)
    for col in frame.columns:
        words = _words(col)
        values = frame[col]
        is_text = not pd.api.types.is_numeric_dtype(values)
        distinct = values.nunique(dropna=True)
        if words & _DIRECT_ID_WORDS:
            suggestion.direct_identifiers.append(col)
            suggestion.reasons[col] = "name looks like a direct identifier"
        elif is_text and n >= 20 and distinct >= 0.9 * n:
            suggestion.direct_identifiers.append(col)
            suggestion.reasons[col] = f"{distinct:,} different values in {n:,} rows"
        elif words & _QI_WORDS:
            suggestion.quasi_identifiers.append(col)
            suggestion.reasons[col] = "typical demographic detail"

    candidates = [
        c for c in frame.columns
        if c not in suggestion.direct_identifiers and c not in suggestion.quasi_identifiers
        and 2 <= frame[c].nunique(dropna=True) <= _MAX_SENSITIVE_VALUES
    ]
    named = [c for c in candidates if _words(c) & _SENSITIVE_WORDS]
    if named:
        suggestion.sensitive_attribute = named[0]
        suggestion.reasons[named[0]] = "name suggests an outcome to protect"
    elif candidates:
        suggestion.sensitive_attribute = candidates[-1]
        suggestion.reasons[candidates[-1]] = "last column with few distinct values"
    return suggestion


def sensitive_candidates(frame: pd.DataFrame) -> list[str]:
    """Columns that can be the sensitive attribute (2-20 distinct values)."""
    return [c for c in frame.columns if 2 <= frame[c].nunique(dropna=True) <= _MAX_SENSITIVE_VALUES]


class DataSource(ABC):
    """Abstract base class: something the dashboard can load a Dataset from."""

    #: Short label shown in the dashboard.
    label: str = "data source"

    @abstractmethod
    def load(self) -> Dataset:
        """Return the data as a :class:`Dataset` with a suggested schema.

        Raises:
            DataLoadError: If the data cannot be read.
        """

    @property
    def bin_edges(self) -> Optional[dict]:
        """Fixed bin edges for skewed numeric columns, if the source knows any."""
        return None

    @property
    def description(self) -> str:
        """One sentence about the data, shown under the selector."""
        return ""


class SampleDataSource(DataSource):
    """The UCI Adult census data shipped with PrivSynth.

    Uses the full ``data/adult.data`` (30,162 complete records) when it
    has been downloaded, otherwise ``data/sample_adult.csv``, a randomly
    generated file in the same format that ships with the repository.

    Args:
        data_dir: Folder holding the files; defaults to the repository's ``data``.
    """

    label = "Sample: UCI Adult census data"

    def __init__(self, data_dir: Union[str, Path, None] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data"

    @property
    def path(self) -> Path:
        full = self.data_dir / "adult.data"
        return full if full.exists() else self.data_dir / "sample_adult.csv"

    @property
    def is_full(self) -> bool:
        return self.path.name == "adult.data"

    def load(self) -> Dataset:
        return AdultDataLoader().load(self.path, name="UCI Adult" if self.is_full else "Adult-format sample")

    @property
    def bin_edges(self) -> dict:
        return dict(AdultDataLoader.BIN_EDGES)

    @property
    def description(self) -> str:
        if self.is_full:
            return ("Public data: the 1994 US census extract (age, sex, race, education, job, "
                    "income ...). No real Ugandan data.")
        return ("A randomly generated 2,000-row file in the Adult format (no real people). "
                "Put the real adult.data in the data folder to use the full census extract.")


class UploadedCSVSource(DataSource):
    """A CSV file uploaded through the dashboard (or any file-like object).

    Args:
        content: The raw bytes of the file.
        filename: Original file name, used as the dataset name.

    Raises:
        DataLoadError: If the file is empty or larger than :data:`MAX_UPLOAD_MB`.
    """

    label = "Upload my own CSV file"

    def __init__(self, content: bytes, filename: str = "uploaded.csv") -> None:
        if not content:
            raise DataLoadError("The uploaded file is empty.")
        size_mb = len(content) / 1e6
        if size_mb > MAX_UPLOAD_MB:
            raise DataLoadError(f"The file is {size_mb:.0f} MB; the limit is {MAX_UPLOAD_MB} MB.")
        self.content = content
        self.filename = filename
        #: Rows removed for missing values by the last :meth:`load`.
        self.rows_dropped = 0

    def load(self) -> Dataset:
        """Read the file, drop incomplete rows and apply :func:`suggest_schema`.

        After loading, :attr:`rows_dropped` says how many rows were removed
        for missing values (blank, ``?``, ``NA``, ``None`` ...), so the user
        is told rather than surprised.
        """
        buffer = io.BytesIO(self.content)
        buffer.name = Path(self.filename).stem
        dataset = CSVLoader().load(buffer)
        try:
            self.rows_dropped = len(pd.read_csv(io.BytesIO(self.content))) - dataset.n_rows
        except (pd.errors.ParserError, ValueError):
            self.rows_dropped = 0
        if len(dataset.columns) < 2:
            raise DataLoadError("The file needs at least two columns; is it really comma-separated?")
        suggestion = suggest_schema(dataset.data)
        if suggestion.sensitive_attribute:
            dataset.sensitive_attribute = suggestion.sensitive_attribute
        if suggestion.quasi_identifiers:
            dataset.quasi_identifiers = suggestion.quasi_identifiers
        return dataset

    @property
    def description(self) -> str:
        return f"Your file: {self.filename}. It stays in this session and is not saved anywhere."
