"""Shared pytest fixtures."""

import numpy as np
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


@pytest.fixture
def linked_dataset() -> Dataset:
    """2,000 records where ``degree`` completely determines ``pay``.

    Also an independent ``region`` column and a numeric ``age``.
    """
    rng = np.random.default_rng(7)
    n = 2000
    degree = rng.choice(["none", "diploma", "degree"], size=n, p=[0.5, 0.3, 0.2])
    pay = np.select([degree == "none", degree == "diploma"], ["low", "mid"], "high")
    frame = pd.DataFrame({
        "age": rng.integers(18, 70, size=n),
        "degree": degree,
        "pay": pay,
        "region": rng.choice(["Central", "Eastern", "Northern", "Western"], size=n),
    })
    return Dataset(frame, name="linked", quasi_identifiers=["age", "region"], sensitive_attribute="pay")
