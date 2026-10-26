"""Week 2 demo: DP mechanisms, the privacy budget and the two generators.

1. Releases one statistic (how many people earn >50K) with Laplace and
   Gaussian noise at several epsilons, to show the privacy/accuracy trade-off.
2. Shows the budget refusing to overspend.
3. Trains both generators at several epsilons and compares how well they
   keep single-column (1-way) and column-pair (2-way) distributions,
   measured by total variation distance (TVD: 0 = identical, 1 = no overlap).
   This is a preview; full evaluation comes in Week 3.

Usage:
    python scripts/week2_demo.py                 # uses data/adult.data if present
    python scripts/week2_demo.py path/to/file    # any Adult-format file
"""

import itertools
import sys
from pathlib import Path

import numpy as np

from privsynth import (
    AdultDataLoader,
    BayesianNetworkSynthesizer,
    GaussianMechanism,
    IndependentMarginalSynthesizer,
    LaplaceMechanism,
    PrivacyBudget,
    PrivacyBudgetExceededError,
    PrivSynthError,
)

ROOT = Path(__file__).resolve().parents[1]
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2]


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation distance between two count arrays of the same shape."""
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def marginal_errors(real_codes, fake_codes, sizes) -> tuple[float, float]:
    """Average TVD over all single columns and over all column pairs."""
    cols = list(sizes)
    one_way = [
        tvd(np.bincount(real_codes[c], minlength=sizes[c]), np.bincount(fake_codes[c], minlength=sizes[c]))
        for c in cols
    ]
    two_way = []
    for a, b in itertools.combinations(cols, 2):
        m = sizes[a] * sizes[b]
        r = np.bincount(real_codes[a] * sizes[b] + real_codes[b], minlength=m)
        f = np.bincount(fake_codes[a] * sizes[b] + fake_codes[b], minlength=m)
        two_way.append(tvd(r, f))
    return float(np.mean(one_way)), float(np.mean(two_way))


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

    # 1. One statistic, two mechanisms ------------------------------------
    true_count = int((dataset.data["income"] == ">50K").sum())
    print(f"== Releasing one count: people earning >50K (true value {true_count:,}) ==")
    print(f"{'epsilon':>8} {'Laplace':>10} {'(std)':>8} {'Gaussian':>10} {'(std)':>8}")
    for eps in EPSILONS:
        lap = LaplaceMechanism(eps, random_state=0)
        gau = GaussianMechanism(eps, delta=1e-5, random_state=0)
        print(f"{eps:>8} {lap.add_noise(true_count):>10.1f} {lap.std:>8.2f} "
              f"{gau.add_noise(true_count):>10.1f} {gau.std:>8.2f}")
    print("Laplace (pure DP) has less noise than Gaussian for a single count;")
    print("Gaussian pays off when many values are released together (L2 sensitivity).\n")

    # 2. The budget refuses to overspend -----------------------------------
    print("== Privacy budget ==")
    budget = PrivacyBudget(epsilon=1.0)
    for label in ["count >50K", "count female", "count over 60"]:
        try:
            LaplaceMechanism(0.4).add_noise(0, budget=budget, label=label)
            print(f"  released '{label}' (remaining epsilon {budget.remaining_epsilon:.1f})")
        except PrivacyBudgetExceededError as err:
            print(f"  refused: {err}")
    print()

    # 3. The two generators ------------------------------------------------
    print(f"== Generators: average TVD over {len(SEEDS)} runs (lower is better) ==")
    print(f"{'epsilon':>8} | {'Independent 1-way':>17} {'2-way':>7} | {'Bayesian 1-way':>14} {'2-way':>7}")
    options = dict(noise="laplace", bin_edges=AdultDataLoader.BIN_EDGES)
    for eps in EPSILONS:
        row = []
        for cls in (IndependentMarginalSynthesizer, BayesianNetworkSynthesizer):
            scores = []
            for seed in SEEDS:
                synth = cls(epsilon=eps, random_state=seed, **options).fit(dataset)
                fake = synth.sample(dataset.n_rows)
                enc = synth.encoder
                scores.append(marginal_errors(enc.encode(dataset), enc.encode(fake), enc.sizes))
            row.extend(np.mean(scores, axis=0))
        print(f"{eps:>8} | {row[0]:>17.3f} {row[1]:>7.3f} | {row[2]:>14.3f} {row[3]:>7.3f}")

    bayes = BayesianNetworkSynthesizer(epsilon=1.0, random_state=0, **options).fit(dataset)
    print("\n== Bayesian network learned at epsilon = 1 (child <- parent) ==")
    for child, parent in bayes.network:
        print(f"  {child} <- {parent or '(root)'}")
    print("\n" + bayes.budget.summary())

    print("\n== Five synthetic records (Bayesian network, epsilon = 1) ==")
    print(bayes.sample(5).data.to_string(index=False))


if __name__ == "__main__":
    main()
