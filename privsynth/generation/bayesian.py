"""Generator 2: a differentially private Bayesian network (PrivBayes-style).

Generator 1 loses every relationship between columns. This generator keeps
the most important ones by learning a **Bayesian network**: each column
gets one *parent* column it depends on, e.g. ``income <- education``. A
synthetic record is drawn one column at a time: first the root, then each
column given the value already drawn for its parent.

It follows the PrivBayes approach [Zhang et al., 2017] with degree 1
(one parent per column), in two phases that split the budget:

1. **Structure** (``structure_fraction`` of epsilon, default 30%). Start from a
   randomly chosen root. Repeatedly score every (new column, parent
   already in the network) pair by *mutual information* (how much knowing
   one column tells you about the other) and pick a pair with the
   :class:`~privsynth.privacy.mechanisms.ExponentialMechanism`, so strong
   dependencies are likely to be chosen without being revealed exactly.
2. **Parameters** (the rest). Measure the root's noisy 1-way counts and,
   for every edge, the noisy 2-way counts of (child, parent). Normalising
   these gives the conditional probabilities used for sampling.

Like PrivBayes, the structure step treats the number of records ``n`` as
public (it is needed to bound the sensitivity of mutual information).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from privsynth.exceptions import NotFittedError
from privsynth.generation.base import Synthesizer
from privsynth.privacy.mechanisms import ExponentialMechanism


def mutual_information(child: np.ndarray, parent: np.ndarray, child_size: int, parent_size: int) -> float:
    """Mutual information (in bits) between two integer-coded columns."""
    joint = np.bincount(child * parent_size + parent, minlength=child_size * parent_size)
    joint = joint.reshape(child_size, parent_size) / len(child)
    pc = joint.sum(axis=1, keepdims=True)
    pp = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (pc @ pp)[nz])))


def mutual_information_sensitivity(n: int, binary: bool) -> float:
    """Most one record can change the mutual information of two columns.

    From PrivBayes [Zhang et al., 2017, Lemma 4.1]. A smaller bound applies
    when one of the two columns has only two values.
    """
    if n < 2:
        return 1.0
    if binary:
        return (1 / n) * math.log2(n) + ((n - 1) / n) * math.log2(n / (n - 1))
    return (2 / n) * math.log2((n + 1) / 2) + ((n - 1) / n) * math.log2((n + 1) / (n - 1))


class BayesianNetworkSynthesizer(Synthesizer):
    """Sample records from a noisy, privately learned Bayesian network.

    Args:
        structure_fraction: Share of epsilon used to choose the network
            (between 0 and 1, exclusive). The rest measures the tables.
        **kwargs: Passed to :class:`~privsynth.generation.base.Synthesizer`
            (``epsilon``, ``noise``, ``delta``, ``n_bins``, ``random_state``...).

    Example:
        >>> synth = BayesianNetworkSynthesizer(epsilon=1.0, random_state=0)
        >>> fake = synth.fit_sample(dataset, n=1000)
        >>> synth.network          # [('age', None), ('marital_status', 'age'), ...]
    """

    name = "Bayesian network"

    def __init__(self, structure_fraction: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0 < structure_fraction < 1:
            raise ValueError(f"structure_fraction must be between 0 and 1, got {structure_fraction!r}.")
        self.structure_fraction = float(structure_fraction)
        self._network: list[tuple[str, Optional[str]]] = []

    # ------------------------------------------------------------------ #
    @property
    def network(self) -> list[tuple[str, Optional[str]]]:
        """``(column, parent)`` pairs in sampling order; the root's parent is None."""
        if not self.is_fitted:
            raise NotFittedError("The network is learned during fit().")
        return list(self._network)

    def _fit(self, codes: pd.DataFrame, n_records: int) -> None:
        sizes = self._encoder.sizes
        columns = list(sizes)
        d = len(columns)

        eps_structure = self.epsilon * self.structure_fraction if d > 1 else 0.0
        eps_tables = self.epsilon - eps_structure
        self._network = self._learn_structure(codes, sizes, eps_structure, n_records)

        eps_each, delta_each = eps_tables / d, self.delta / d
        self._tables: dict[str, np.ndarray] = {}
        for child, parent in self._network:
            c = codes[child].to_numpy()
            if parent is None:
                counts = np.bincount(c, minlength=sizes[child])
                noisy = self._measure(counts, eps_each, delta_each, label=f"marginal: {child}")
                self._tables[child] = self._normalise(noisy)
            else:
                p = codes[parent].to_numpy()
                counts = np.bincount(c * sizes[parent] + p, minlength=sizes[child] * sizes[parent])
                noisy = self._measure(
                    counts, eps_each, delta_each, label=f"table: {child} | {parent}"
                ).reshape(sizes[child], sizes[parent])
                self._tables[child] = self._conditional(noisy)

    def _learn_structure(
        self, codes: pd.DataFrame, sizes: dict[str, int], epsilon: float, n: int
    ) -> list[tuple[str, Optional[str]]]:
        columns = list(sizes)
        root = columns[int(self._rng.integers(len(columns)))]   # random: uses no data
        network: list[tuple[str, Optional[str]]] = [(root, None)]
        if len(columns) == 1:
            return network

        in_network = [root]
        remaining = [c for c in columns if c != root]
        eps_step = epsilon / len(remaining)
        while remaining:
            pairs = [(child, parent) for child in remaining for parent in in_network]
            scores = [
                mutual_information(codes[ch].to_numpy(), codes[pa].to_numpy(), sizes[ch], sizes[pa])
                for ch, pa in pairs
            ]
            binary = all(min(sizes[ch], sizes[pa]) == 2 for ch, pa in pairs)
            mech = ExponentialMechanism(
                eps_step,
                sensitivity=mutual_information_sensitivity(n, binary),
                random_state=self._rng,
            )
            child, parent = pairs[mech.select(scores, budget=self._budget, label="structure: choose parent")]
            network.append((child, parent))
            in_network.append(child)
            remaining.remove(child)
        return network

    def _conditional(self, joint: np.ndarray) -> np.ndarray:
        """P(child | parent) from noisy joint counts, shape (child_size, parent_size).

        A parent value whose noisy counts are all zero falls back to the
        child's overall distribution rather than a flat uniform guess.
        """
        fallback = self._normalise(joint.sum(axis=1))
        totals = joint.sum(axis=0)
        cond = np.empty_like(joint)
        for j in range(joint.shape[1]):
            cond[:, j] = joint[:, j] / totals[j] if totals[j] > 0 else fallback
        return cond

    def _sample(self, n: int) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for child, parent in self._network:
            table = self._tables[child]
            if parent is None:
                out[child] = self._rng.choice(len(table), size=n, p=table)
                continue
            values = np.empty(n, dtype=np.int64)
            parent_codes = out[parent]
            for j in np.unique(parent_codes):
                idx = np.flatnonzero(parent_codes == j)
                values[idx] = self._rng.choice(table.shape[0], size=idx.size, p=table[:, j])
            out[child] = values
        return pd.DataFrame(out)[self._encoder.columns]
