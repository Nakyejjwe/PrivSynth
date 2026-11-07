"""Create data/demo_upload.csv: a made-up survey for trying the dashboard's upload.

Every value is randomly generated; no real person is described. The file
deliberately includes direct identifiers (name, phone) and Ugandan
districts, so the dashboard's column detection has something to find.

Usage:
    python scripts/make_demo_upload.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DISTRICTS = ["Kampala", "Wakiso", "Mukono", "Gulu", "Mbarara", "Jinja", "Mbale", "Arua", "Lira", "Masaka"]
FIRST = ["Amina", "Brian", "Grace", "Isaac", "Joan", "Moses", "Ruth", "Sam", "Winnie", "Paul"]
LAST = ["Okello", "Namuli", "Mugisha", "Achieng", "Ssempala", "Nakato", "Opio", "Kato", "Atim", "Byaruhanga"]


def make(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    sex = rng.choice(["Female", "Male"], n)
    district = rng.choice(DISTRICTS, n, p=[0.25, 0.2, 0.1, 0.08, 0.08, 0.08, 0.06, 0.05, 0.05, 0.05])
    education = rng.choice(["No schooling", "Primary", "Secondary", "Tertiary"], n, p=[0.1, 0.4, 0.35, 0.15])
    urban = np.isin(district, ["Kampala", "Wakiso", "Mukono", "Jinja"])
    score = (education == "Tertiary") * 1.5 + (education == "Secondary") * 0.6 + urban * 0.5 - 0.6
    on_art = rng.random(n) < 1 / (1 + np.exp(-(0.02 * (age - 40) - 1.8)))
    return pd.DataFrame({
        "full_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)} {i:04d}" for i in range(n)],
        "phone": [f"07{rng.integers(0, 10**8):08d}" for _ in range(n)],
        "age": age,
        "sex": sex,
        "district": district,
        "education": education,
        "household_size": rng.integers(1, 11, n),
        "monthly_income_band": np.where(rng.random(n) < 1 / (1 + np.exp(-score)), "Above 500k UGX", "500k UGX or less"),
        "hiv_status": np.where(on_art, "Positive", "Negative"),
    })


if __name__ == "__main__":
    path = ROOT / "data" / "demo_upload.csv"
    make().to_csv(path, index=False)
    print(f"Wrote {path} (randomly generated, no real people)")
