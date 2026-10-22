"""Shared pytest fixtures."""

import pandas as pd
import pytest

from privsynth import Dataset


@pytest.fixture
def small_frame() -> pd.DataFrame:
    """Eight records with known equivalence classes on (age, sex).

    Classes: (30, F) x3, (30, M) x2, (40, F) x2, (55, M) x1 (unique).
    """
    return pd.DataFrame({
        "age":    [30, 30, 30, 30, 30, 40, 40, 55],
        "sex":    ["F", "F", "F", "M", "M", "F", "F", "M"],
        "city":   ["Kampala", "Kampala", "Gulu", "Mbarara", "Kampala", "Gulu", "Gulu", "Jinja"],
        "income": ["<=50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", ">50K"],
    })


@pytest.fixture
def small_dataset(small_frame) -> Dataset:
    return Dataset(small_frame, name="small", quasi_identifiers=["age", "sex"],
                   sensitive_attribute="income")
