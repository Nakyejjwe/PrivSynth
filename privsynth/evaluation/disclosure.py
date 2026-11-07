"""Privacy-risk evaluators: what does the synthetic data still reveal?

Differential privacy bounds risk *in theory*; these evaluators measure it
*in practice*, by playing the attacker. Both compare two groups of real
people:

* **members** - records the synthesizer was trained on (``real``);
* **non-members** - records it never saw (``holdout``).

If an attack works equally well on both groups, the synthetic data is only
revealing general patterns (e.g. "older people earn more"), which is the
point of sharing data. If it works better on members, the synthetic data
is leaking something about the specific people it was trained on.

* :class:`DistanceToClosestRecord` - are synthetic records suspiciously
  close copies of training records?
* :class:`AttributeInferenceRisk` - can an attacker who knows a person's
  quasi-identifiers use the synthetic data to guess their sensitive value?
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from privsynth.data.dataset import Dataset
from privsynth.evaluation.base import (
    EvaluationResult,
    Evaluator,
    MixedFeatureSpace,
    encode_together,
)
from privsynth.exceptions import EvaluationError, SchemaError


class DistanceToClosestRecord(Evaluator):
    """Distance to closest record (DCR), compared with a holdout baseline.

    For each synthetic record we find the nearest training record. On its
    own the distance means little, so we also measure how close *new real
    people* (the holdout) are to the training data. Synthetic records that
    sit much closer to training records than new people do are likely
    near-copies.

    The sharpest test is ``closer_to_train_share``: compare each synthetic
    record's distance to an equal-sized sample of training records and to
    the holdout records. If the synthesizer learned only general patterns,
    the training sample is no closer than the holdout, giving about 0.5.
    Values well above 0.5 mean memorisation.

    Args:
        max_records: Most synthetic (and holdout) records to check, to keep
            the nearest-neighbour search fast on large data.
        random_state: Seed for the sub-samples.
    """

    name = "Distance to closest record"
    key = "dcr"
    kind = "risk"
    requires_holdout = True

    def __init__(self, max_records: int = 2000, random_state: int = 0) -> None:
        if not isinstance(max_records, int) or max_records < 1:
            raise ValueError(f"max_records must be a positive integer, got {max_records!r}.")
        self.max_records = max_records
        self.random_state = random_state

    def _evaluate(self, real, synthetic, holdout) -> EvaluationResult:
        rng = np.random.default_rng(self.random_state)
        space = MixedFeatureSpace().fit(
            pd.concat([real.data, synthetic.data, holdout.data], ignore_index=True),
            numeric=real.numeric_columns,
        )
        train_all = space.transform(real.data)
        syn_rows = _rows(synthetic, min(self.max_records, synthetic.n_rows), rng)
        hold_rows = _rows(holdout, min(self.max_records, holdout.n_rows), rng)
        queries, holdout_queries = space.transform(syn_rows), space.transform(hold_rows)
        train_keys = set(_row_keys(real.data))

        # Distances to the WHOLE training set: are records copied or near-copied?
        to_train = _nearest(train_all, queries)
        holdout_to_train = _nearest(train_all, holdout_queries)

        # Closer-to-train test: equal-sized reference sets, so neither one
        # is closer just by being bigger.
        m = min(real.n_rows, holdout.n_rows)
        train_ref = train_all[rng.choice(real.n_rows, size=m, replace=False)]
        holdout_ref = space.transform(_rows(holdout, m, rng))
        near_train, near_holdout = _nearest(train_ref, queries), _nearest(holdout_ref, queries)
        closer = np.mean(np.where(np.isclose(near_train, near_holdout), 0.5, near_train < near_holdout))

        syn_median, base_median = float(np.median(to_train)), float(np.median(holdout_to_train))
        metrics = {
            "dcr_synthetic_median": syn_median,
            "dcr_holdout_median": base_median,
            "dcr_ratio": syn_median / base_median if base_median > 0 else float("nan"),
            "exact_match_synthetic": float(np.mean([k in train_keys for k in _row_keys(syn_rows)])),
            "exact_match_holdout": float(np.mean([k in train_keys for k in _row_keys(hold_rows)])),
            "closer_to_train_share": float(closer),
        }
        details = pd.DataFrame({
            "source": ["synthetic"] * len(to_train) + ["holdout"] * len(holdout_to_train),
            "distance_to_train": np.concatenate([to_train, holdout_to_train]),
        })
        notes = [f"{len(queries):,} synthetic records checked against all {real.n_rows:,} training "
                 f"records; closer-to-train test uses {m:,} training vs {m:,} holdout records."]
        return self._result(
            metrics,
            lower_is_better=["exact_match_synthetic", "closer_to_train_share"],
            details=details,
            notes=notes,
        )


class AttributeInferenceRisk(Evaluator):
    """Can the synthetic data be used to guess a real person's sensitive value?

    The attacker knows a target's quasi-identifiers (age, sex, race, ...)
    and looks up synthetic records with the same values; their most common
    sensitive value (e.g. income) is the guess. Whole-number
    quasi-identifiers with up to ``max_exact_integers`` values (such as age
    in years) are matched exactly; others are binned. If no synthetic
    record matches, the attacker guesses the overall most common value.

    The attack is run on members and non-members. The **inference risk**
    is the attacker's advantage on members, scaled to 0..1::

        risk = max(0, (acc_members - acc_non_members) / (1 - acc_non_members))

    0 means the synthetic data tells the attacker nothing more about the
    people it was trained on than about anyone else. This follows the
    member/control comparison used by Giomi et al. (2023, "Anonymeter").

    Args:
        quasi_identifiers: Columns the attacker knows; defaults to the dataset's.
        sensitive_attribute: Column to guess; defaults to the dataset's.
        max_records: Most targets per group.
        n_bins: Bins for numeric quasi-identifiers that are not matched exactly.
        bin_edges: Explicit bin edges per numeric column.
        max_exact_integers: See above.
        random_state: Seed for choosing targets.
    """

    name = "Attribute inference risk"
    key = "inference"
    kind = "risk"
    requires_holdout = True

    def __init__(
        self,
        quasi_identifiers: Optional[Iterable[str]] = None,
        sensitive_attribute: Optional[str] = None,
        max_records: int = 2000,
        n_bins: int = 16,
        bin_edges: Optional[Mapping[str, Sequence[float]]] = None,
        max_exact_integers: int = 128,
        random_state: int = 0,
    ) -> None:
        if not isinstance(max_records, int) or max_records < 1:
            raise ValueError(f"max_records must be a positive integer, got {max_records!r}.")
        self.max_exact_integers = max_exact_integers
        self.quasi_identifiers = list(quasi_identifiers) if quasi_identifiers is not None else None
        self.sensitive_attribute = sensitive_attribute
        self.max_records = max_records
        self.n_bins = n_bins
        self.bin_edges = dict(bin_edges or {})
        self.random_state = random_state

    def _evaluate(self, real, synthetic, holdout) -> EvaluationResult:
        qis = self.quasi_identifiers or real.quasi_identifiers
        sensitive = self.sensitive_attribute or real.sensitive_attribute
        if not qis or sensitive is None:
            raise EvaluationError(
                "Attribute inference needs quasi-identifiers and a sensitive attribute "
                "(set them on the Dataset or pass them in)."
            )
        missing = [c for c in [*qis, sensitive] if c not in real.columns]
        if missing:
            raise SchemaError(f"Column(s) not in the data: {missing}")
        if sensitive in qis:
            raise EvaluationError(f"'{sensitive}' cannot be both known and guessed.")

        cols = [*qis, sensitive]
        _, (r, s, h) = encode_together(
            [real.select(cols), synthetic.select(cols), holdout.select(cols)],
            self.n_bins, self.bin_edges, max_exact_integers=self.max_exact_integers,
        )
        rng = np.random.default_rng(self.random_state)
        m = min(self.max_records, len(r), len(h))
        members = r.iloc[rng.choice(len(r), size=m, replace=False)]
        non_members = h.iloc[rng.choice(len(h), size=m, replace=False)]

        lookup = s.groupby(qis if len(qis) > 1 else qis[0])[sensitive].agg(lambda v: v.value_counts().idxmax())
        fallback = s[sensitive].value_counts().idxmax()

        rows = []
        for group, targets in (("members", members), ("non-members", non_members)):
            keys = (pd.MultiIndex.from_frame(targets[qis]) if len(qis) > 1
                    else pd.Index(targets[qis[0]]))
            guesses = lookup.reindex(keys)
            matched = guesses.notna().to_numpy()
            guesses = guesses.fillna(fallback).to_numpy()
            truth = targets[sensitive].to_numpy()
            rows.append({
                "group": group,
                "targets": m,
                "match_rate": float(matched.mean()),
                "accuracy": float(np.mean(guesses == truth)),
                "baseline_accuracy": float(np.mean(truth == fallback)),
            })
        details = pd.DataFrame(rows)
        acc_in, acc_out = details.loc[0, "accuracy"], details.loc[1, "accuracy"]
        risk = max(0.0, (acc_in - acc_out) / (1.0 - acc_out)) if acc_out < 1 else 0.0
        metrics = {
            "accuracy_members": acc_in,
            "accuracy_non_members": acc_out,
            "baseline_accuracy": details.loc[0, "baseline_accuracy"],
            "match_rate_members": details.loc[0, "match_rate"],
            "inference_risk": risk,
        }
        notes = [f"Attacker knows {', '.join(qis)} and guesses '{sensitive}' "
                 f"for {m:,} members and {m:,} non-members."]
        return self._result(metrics, lower_is_better=["inference_risk"], details=details, notes=notes)


def _rows(dataset: Dataset, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A random sample of ``n`` rows of ``dataset`` as a DataFrame."""
    frame = dataset.data
    if n >= len(frame):
        return frame
    return frame.iloc[rng.choice(len(frame), size=n, replace=False)]


def _row_keys(frame: pd.DataFrame) -> np.ndarray:
    """A hash of each row's values, so identical records get identical keys.

    Values are compared as text so that, say, ``40`` stored as an integer
    in one table and as ``40.0`` in another still count as the same.
    """
    text = frame.apply(lambda col: col.map(_canonical))
    return pd.util.hash_pandas_object(text, index=False).to_numpy()


def _canonical(value) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _nearest(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Euclidean distance from each query point to its nearest reference point."""
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(reference)
    distances, _ = nn.kneighbors(queries)
    return distances[:, 0]
