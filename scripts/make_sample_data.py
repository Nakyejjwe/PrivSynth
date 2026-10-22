"""Create data/sample_adult.csv: a small, fake table in the UCI Adult format.

The sample lets the code and tests run before the real dataset is
downloaded. Every row is randomly generated - no real people. For the
experiments in the report, download the real file instead (see README).

Usage:
    python scripts/make_sample_data.py [n_rows]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from privsynth.data.loaders import AdultDataLoader


def make_sample(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda values, p=None: rng.choice(values, size=n_rows, p=p)  # noqa: E731

    education = pick(["HS-grad", "Some-college", "Bachelors", "Masters", "11th", "Doctorate"],
                     [0.33, 0.23, 0.2, 0.1, 0.1, 0.04])
    edu_num = pd.Series(education).map(
        {"11th": 7, "HS-grad": 9, "Some-college": 10, "Bachelors": 13, "Masters": 14, "Doctorate": 16})
    age = np.clip(rng.normal(38, 13, n_rows).round(), 17, 90).astype(int)
    hours = np.clip(rng.normal(40, 12, n_rows).round(), 1, 99).astype(int)

    # Income loosely depends on education, age and hours, as in the real data
    score = 0.35 * (edu_num - 9) + 0.04 * (age - 38) + 0.04 * (hours - 40) + rng.normal(0, 1, n_rows)
    income = np.where(score > 1.0, ">50K", "<=50K")

    frame = pd.DataFrame({
        "age": age,
        "workclass": pick(["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"],
                          [0.74, 0.1, 0.08, 0.05, 0.03]),
        "fnlwgt": rng.integers(20_000, 900_000, n_rows),
        "education": education,
        "education_num": edu_num,
        "marital_status": pick(["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"],
                               [0.46, 0.33, 0.14, 0.04, 0.03]),
        "occupation": pick(["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
                            "Sales", "Other-service", "Machine-op-inspct"]),
        "relationship": pick(["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"],
                             [0.4, 0.26, 0.16, 0.11, 0.07]),
        "race": pick(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                     [0.85, 0.1, 0.03, 0.01, 0.01]),
        "sex": pick(["Male", "Female"], [0.67, 0.33]),
        "capital_gain": np.where(rng.random(n_rows) < 0.08, rng.integers(1_000, 20_000, n_rows), 0),
        "capital_loss": np.where(rng.random(n_rows) < 0.05, rng.integers(500, 3_000, n_rows), 0),
        "hours_per_week": hours,
        "native_country": pick(["United-States", "Mexico", "Philippines", "Germany", "India", "Canada"],
                               [0.91, 0.03, 0.02, 0.015, 0.015, 0.01]),
        "income": income,
    })
    return frame[AdultDataLoader.COLUMNS]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = Path(__file__).resolve().parents[1] / "data" / "sample_adult.csv"
    out.parent.mkdir(exist_ok=True)
    make_sample(n).to_csv(out, index=False)
    print(f"Wrote {n} fake records to {out}")
