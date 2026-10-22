"""Tests for the data module: Dataset and loaders."""

import pandas as pd
import pytest

from privsynth import AdultDataLoader, CSVLoader, DataLoadError, Dataset, SchemaError
from privsynth.data.loaders import DataLoader


# --- Dataset -------------------------------------------------------------
def test_dataset_basic_properties(small_dataset):
    assert len(small_dataset) == 8
    assert small_dataset.numeric_columns == ["age"]
    assert small_dataset.categorical_columns == ["sex", "city", "income"]
    assert small_dataset.quasi_identifiers == ["age", "sex"]
    assert small_dataset.sensitive_attribute == "income"


def test_dataset_is_protected_from_outside_changes(small_frame, small_dataset):
    small_frame.loc[0, "age"] = 99          # change the caller's frame
    copy = small_dataset.data
    copy.loc[0, "age"] = 99                 # change the returned copy
    assert small_dataset.data.loc[0, "age"] == 30


def test_empty_or_wrong_input_rejected():
    with pytest.raises(SchemaError):
        Dataset(pd.DataFrame())
    with pytest.raises(SchemaError):
        Dataset([[1, 2]])


def test_unknown_columns_rejected(small_frame):
    with pytest.raises(SchemaError):
        Dataset(small_frame, quasi_identifiers=["age", "district"])
    with pytest.raises(SchemaError):
        Dataset(small_frame, sensitive_attribute="salary")


def test_column_cannot_be_both_qi_and_sensitive(small_dataset):
    with pytest.raises(SchemaError):
        small_dataset.sensitive_attribute = "age"
    with pytest.raises(SchemaError):
        small_dataset.quasi_identifiers = ["age", "income"]


def test_select_and_sample_keep_schema(small_dataset):
    subset = small_dataset.select(["age", "income"])
    assert subset.quasi_identifiers == ["age"]
    assert subset.sensitive_attribute == "income"
    sample = small_dataset.sample(3, random_state=0)
    assert len(sample) == 3 and sample.quasi_identifiers == ["age", "sex"]


def test_summary_reports_roles(small_dataset):
    summary = small_dataset.summary().set_index("column")
    assert summary.loc["age", "role"] == "quasi-identifier"
    assert summary.loc["income", "role"] == "sensitive"
    assert summary.loc["city", "distinct_values"] == 4


# --- Loaders -------------------------------------------------------------
def test_data_loader_is_abstract():
    with pytest.raises(TypeError):
        DataLoader()


def test_csv_loader_cleans_and_drops_missing(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("age, sex, city\n30, F , Kampala\n41, M, ?\n25, F, Gulu\n")
    ds = CSVLoader().load(path, quasi_identifiers=["age", "sex"])
    assert len(ds) == 2
    assert ds.data["sex"].tolist() == ["F", "F"]
    assert ds.name == "people"


def test_csv_loader_can_keep_missing(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("age,city\n30,Kampala\n41,?\n")
    assert len(CSVLoader(drop_missing=False).load(path)) == 2


def test_missing_and_empty_files_raise(tmp_path):
    with pytest.raises(DataLoadError):
        CSVLoader().load(tmp_path / "nope.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataLoadError):
        CSVLoader().load(empty)


ADULT_ROWS = (
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, "
    "White, Male, 2174, 0, 40, United-States, <=50K\n"
    "50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, "
    "Husband, White, Male, 0, 0, 13, United-States, >50K.\n"
    "38, Private, 215646, HS-grad, 9, Divorced, ?, Not-in-family, White, Male, 0, 0, 40, "
    "United-States, <=50K\n"
)


def test_adult_loader_raw_format(tmp_path):
    path = tmp_path / "adult.data"
    path.write_text(ADULT_ROWS)
    ds = AdultDataLoader().load(path)
    assert len(ds) == 2                                   # row with '?' dropped
    assert ds.columns == AdultDataLoader.COLUMNS
    assert ds.data["income"].tolist() == ["<=50K", ">50K"]  # trailing '.' removed
    assert ds.quasi_identifiers == AdultDataLoader.QUASI_IDENTIFIERS
    assert ds.sensitive_attribute == "income"
    assert "age" in ds.numeric_columns


def test_adult_loader_test_file_header_comment(tmp_path):
    path = tmp_path / "adult.test"
    path.write_text("|1x3 Cross validator\n" + ADULT_ROWS)
    assert len(AdultDataLoader().load(path)) == 2


def test_adult_loader_with_header_row(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(",".join(AdultDataLoader.COLUMNS) + "\n" + ADULT_ROWS)
    ds = AdultDataLoader().load(path)
    assert len(ds) == 2 and "age" in ds.numeric_columns


def test_adult_loader_rejects_wrong_shape(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(DataLoadError):
        AdultDataLoader().load(path)
