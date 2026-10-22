"""Week 1 demo: load the Adult data and measure re-identification risk.

Shows how generalising quasi-identifiers (age ranges, merging rare
countries) lowers the risk.

Usage:
    python scripts/week1_demo.py                 # uses data/adult.data if present
    python scripts/week1_demo.py path/to/file    # any Adult-format file
"""

import sys
from pathlib import Path

from privsynth import AdultDataLoader, Generalizer, KAnonymityChecker, PrivSynthError

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    elif (ROOT / "data" / "adult.data").exists():
        path = ROOT / "data" / "adult.data"
    else:
        path = ROOT / "data" / "sample_adult.csv"

    try:
        dataset = AdultDataLoader().load(path)
    except PrivSynthError as err:
        sys.exit(f"Error: {err}")

    print(dataset, "\n")
    print(dataset.summary().to_string(index=False), "\n")

    checker = KAnonymityChecker(k=5)
    print("== Original data ==")
    print(checker.evaluate(dataset), "\n")

    settings = [
        ("Age in 5-year ranges", Generalizer(bin_widths={"age": 5})),
        ("Age in 10-year ranges", Generalizer(bin_widths={"age": 10})),
        ("10-year ranges + rare values -> 'Other'", Generalizer(bin_widths={"age": 10}, rare_threshold=50)),
    ]
    for label, generalizer in settings:
        print(f"== {label} ==")
        print(checker.evaluate(generalizer.generalize(dataset)), "\n")


if __name__ == "__main__":
    main()
