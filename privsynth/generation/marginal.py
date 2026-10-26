"""Generator 1: independent noisy marginals (the baseline).

For each column separately, count how often each value occurs (its
*marginal* distribution), add DP noise to the counts, and sample every
column independently from its own noisy distribution.

* Budget: ``epsilon / d`` per column for ``d`` columns (sequential composition).
* Strength: each column on its own looks right, even at small epsilon.
* Weakness: relationships *between* columns are lost. Synthetic people
  get an education level unrelated to their income, a ``relationship`` of
  "Husband" with ``sex`` "Female", and so on. Generator 2 fixes this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from privsynth.exceptions import NotFittedError
from privsynth.generation.base import Synthesizer


class IndependentMarginalSynthesizer(Synthesizer):
    """Sample each column independently from its noisy 1-way marginal.

    Takes the same arguments as :class:`~privsynth.generation.base.Synthesizer`.

    Example:
        >>> synth = IndependentMarginalSynthesizer(epsilon=1.0, random_state=0)
        >>> fake = synth.fit_sample(dataset, n=1000)
        >>> print(synth.budget.summary())
    """

    name = "Independent marginals"

    def _fit(self, codes: pd.DataFrame, n_records: int) -> None:
        sizes = self._encoder.sizes
        eps_each, delta_each = self.epsilon / len(sizes), self.delta / len(sizes)
        self._marginals: dict[str, np.ndarray] = {}
        for col, size in sizes.items():
            counts = np.bincount(codes[col].to_numpy(), minlength=size)
            noisy = self._measure(counts, eps_each, delta_each, label=f"marginal: {col}")
            self._marginals[col] = self._normalise(noisy)

    def _sample(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {col: self._rng.choice(len(p), size=n, p=p) for col, p in self._marginals.items()}
        )

    @property
    def marginals(self) -> dict[str, np.ndarray]:
        """Column -> noisy probability of each code (copies, safe to inspect)."""
        if not self.is_fitted:
            raise NotFittedError("Marginals are measured during fit().")
        return {col: p.copy() for col, p in self._marginals.items()}
