"""Utility evaluators: how useful is the synthetic data?

Three views of usefulness, from simple to demanding:

* :class:`MarginalSimilarity` - do single columns and pairs of columns have
  the same distribution? Measured by total variation distance (TVD).
* :class:`AssociationSimilarity` - are the *strengths* of relationships
  between columns kept? Measured by Cramer's V for every pair of columns.
* :class:`MachineLearningUtility` - can a model trained on synthetic data
  make good predictions about **real** people? This "train on synthetic,
  test on real" (TSTR) score is what an analyst ultimately cares about.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from privsynth.data.dataset import Dataset
from privsynth.evaluation.base import (
    EvaluationResult,
    Evaluator,
    encode_together,
    total_variation_distance,
)
from privsynth.exceptions import EvaluationError, SchemaError


class MarginalSimilarity(Evaluator):
    """Compare 1-way and 2-way distributions with total variation distance.

    A *1-way marginal* is the distribution of one column (e.g. the share of
    each education level); a *2-way marginal* is the joint distribution of
    a pair (e.g. education by income). Generator 1 keeps 1-way marginals
    but not 2-way ones, so this evaluator shows the difference clearly.

    Args:
        n_bins: Bins for numeric columns.
        bin_edges: Explicit bin edges, e.g. ``AdultDataLoader.BIN_EDGES``.
    """

    name = "Marginal similarity"
    key = "marginal"
    kind = "utility"

    def __init__(self, n_bins: int = 16, bin_edges: Optional[Mapping[str, Sequence[float]]] = None) -> None:
        self.n_bins = n_bins
        self.bin_edges = dict(bin_edges or {})

    def _evaluate(self, real, synthetic, holdout) -> EvaluationResult:
        encoder, (r, s) = encode_together([real, synthetic], self.n_bins, self.bin_edges)
        sizes = encoder.sizes
        rows = []
        for col in sizes:
            rows.append({
                "type": "1-way",
                "columns": col,
                "tvd": total_variation_distance(
                    np.bincount(r[col], minlength=sizes[col]),
                    np.bincount(s[col], minlength=sizes[col]),
                ),
            })
        for a, b in itertools.combinations(sizes, 2):
            cells = sizes[a] * sizes[b]
            rows.append({
                "type": "2-way",
                "columns": f"{a} x {b}",
                "tvd": total_variation_distance(
                    np.bincount(r[a] * sizes[b] + r[b], minlength=cells),
                    np.bincount(s[a] * sizes[b] + s[b], minlength=cells),
                ),
            })
        details = pd.DataFrame(rows)
        one = details.loc[details["type"] == "1-way", "tvd"]
        two = details.loc[details["type"] == "2-way", "tvd"]
        metrics = {
            "tvd_1way_mean": one.mean(),
            "tvd_1way_max": one.max(),
            "tvd_2way_mean": two.mean() if len(two) else 0.0,
            "tvd_2way_max": two.max() if len(two) else 0.0,
        }
        return self._result(metrics, lower_is_better=metrics.keys(), details=details)


def cramers_v(a: np.ndarray, b: np.ndarray, size_a: int, size_b: int) -> float:
    """Cramer's V between two integer-coded columns (0 = unrelated, 1 = one decides the other).

    ``V = sqrt(chi2 / (n * (min(rows, cols) - 1)))``, computed on the
    contingency table with empty rows and columns removed. Returns 0 when a
    column takes only one value (it cannot be associated with anything).
    """
    table = np.bincount(a * size_b + b, minlength=size_a * size_b).reshape(size_a, size_b).astype(float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = table.sum()
    k = min(table.shape) - 1
    if n == 0 or k == 0:
        return 0.0
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return min(1.0, math.sqrt(chi2 / (n * k)))


class AssociationSimilarity(Evaluator):
    """Compare how strongly every pair of columns is related (Cramer's V).

    Numeric columns are binned first, so one measure works for every pair.
    The headline metric is the mean absolute difference between the real
    and synthetic association strengths.

    Args:
        n_bins: Bins for numeric columns.
        bin_edges: Explicit bin edges per numeric column.
    """

    name = "Association similarity"
    key = "association"
    kind = "utility"

    def __init__(self, n_bins: int = 16, bin_edges: Optional[Mapping[str, Sequence[float]]] = None) -> None:
        self.n_bins = n_bins
        self.bin_edges = dict(bin_edges or {})

    def _evaluate(self, real, synthetic, holdout) -> EvaluationResult:
        encoder, (r, s) = encode_together([real, synthetic], self.n_bins, self.bin_edges)
        sizes = encoder.sizes
        rows = []
        for a, b in itertools.combinations(sizes, 2):
            real_v = cramers_v(r[a].to_numpy(), r[b].to_numpy(), sizes[a], sizes[b])
            syn_v = cramers_v(s[a].to_numpy(), s[b].to_numpy(), sizes[a], sizes[b])
            rows.append({"column_a": a, "column_b": b, "real_v": real_v,
                         "synthetic_v": syn_v, "abs_diff": abs(real_v - syn_v)})
        details = pd.DataFrame(rows, columns=["column_a", "column_b", "real_v", "synthetic_v", "abs_diff"])
        metrics = {
            "assoc_diff_mean": details["abs_diff"].mean() if len(details) else 0.0,
            "assoc_diff_max": details["abs_diff"].max() if len(details) else 0.0,
        }
        return self._result(metrics, lower_is_better=metrics.keys(), details=details)

    @staticmethod
    def matrix(details: pd.DataFrame, value: str = "real_v") -> pd.DataFrame:
        """Turn the ``details`` table into a square matrix (for a heatmap)."""
        cols = list(dict.fromkeys(list(details["column_a"]) + list(details["column_b"])))
        m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for row in details.itertuples(index=False):
            v = getattr(row, value)
            m.loc[row.column_a, row.column_b] = v
            m.loc[row.column_b, row.column_a] = v
        return m


class MachineLearningUtility(Evaluator):
    """Train on synthetic, test on real (TSTR).

    Two copies of the same model are trained: one on the real training data
    (the best we could hope for) and one on the synthetic data. Both are
    tested on the **holdout** set of real records. The closer the synthetic
    model's score is to the real model's, the more useful the synthetic
    data is for prediction.

    Args:
        target: Column to predict; defaults to the dataset's sensitive attribute.
        model: ``"logistic"`` (logistic regression) or ``"random_forest"``.
        random_state: Seed for the model.
    """

    name = "Machine-learning utility"
    key = "ml"
    kind = "utility"
    requires_holdout = True
    MODELS = ("logistic", "random_forest")

    def __init__(self, target: Optional[str] = None, model: str = "logistic", random_state: int = 0) -> None:
        if model not in self.MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {list(self.MODELS)}.")
        self.target = target
        self.model = model
        self.random_state = random_state

    def _evaluate(self, real, synthetic, holdout) -> EvaluationResult:
        target = self.target or real.sensitive_attribute
        if target is None:
            raise EvaluationError("No target column: pass target=... or set the dataset's sensitive attribute.")
        if target not in real.columns:
            raise SchemaError(f"Target column '{target}' is not in the data.")

        features = [c for c in real.columns if c != target]
        if not features:
            raise EvaluationError("Machine-learning utility needs at least one column besides the target.")
        numeric = [c for c in real.numeric_columns if c != target]
        categorical = [c for c in features if c not in numeric]

        X_test, y_test = self._split(holdout, features, categorical, target)
        y_real = real.data[target].astype(str)
        # Positive class for AUC: the rarer real outcome (">50K" for Adult).
        positive = y_real.value_counts().idxmin() if y_real.nunique() > 1 else y_real.iloc[0]
        binary = y_real.nunique() == 2

        notes: list[str] = []
        scores = {}
        for source, data in (("real", real), ("synthetic", synthetic)):
            X, y = self._split(data, features, categorical, target)
            model = self._fit(X, y, numeric, categorical)
            if y.nunique() < 2:
                notes.append(f"The {source} data has only one value of '{target}'; "
                             "its model always predicts that value.")
            predicted = model.predict(X_test)
            scores[f"accuracy_{source}"] = accuracy_score(y_test, predicted)
            scores[f"f1_{source}"] = f1_score(y_test, predicted, average="macro", zero_division=0)
            if binary:
                scores[f"auc_{source}"] = _auc(y_test, _positive_proba(model, X_test, positive), positive)

        majority = y_real.value_counts().idxmax()
        scores["accuracy_majority"] = float((y_test == majority).mean())
        main = "auc" if binary else "f1"
        real_score, syn_score = scores[f"{main}_real"], scores[f"{main}_synthetic"]
        scores["utility_ratio"] = syn_score / real_score if real_score > 0 else float("nan")

        details = pd.DataFrame(
            {"trained_on": ["real", "synthetic"],
             "accuracy": [scores["accuracy_real"], scores["accuracy_synthetic"]],
             "f1_macro": [scores["f1_real"], scores["f1_synthetic"]],
             **({"auc": [scores["auc_real"], scores["auc_synthetic"]]} if binary else {})}
        )
        notes.append(f"Predicting '{target}' with {self.model.replace('_', ' ')}; "
                     f"tested on {len(y_test):,} holdout records.")
        return self._result(scores, details=details, notes=notes)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split(data: Dataset, features, categorical, target) -> tuple[pd.DataFrame, pd.Series]:
        frame = data.data
        X = frame[features].copy()
        for col in categorical:
            X[col] = X[col].astype(str)
        return X, frame[target].astype(str)

    def _fit(self, X, y, numeric, categorical):
        if y.nunique() < 2:
            return DummyClassifier(strategy="most_frequent").fit(X, y)
        prep = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        if self.model == "logistic":
            clf = LogisticRegression(max_iter=2000)
        else:
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                         random_state=self.random_state, n_jobs=-1)
        model = Pipeline([("prep", prep), ("model", clf)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return model.fit(X, y)

    def __repr__(self) -> str:
        return f"MachineLearningUtility(target={self.target!r}, model={self.model!r})"


def _positive_proba(model, X: pd.DataFrame, positive: str) -> np.ndarray:
    """Probability of the positive class (0 if the model never saw that class)."""
    classes = list(model.classes_)
    if positive not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(positive)]


def _auc(y_true: pd.Series, score: np.ndarray, positive: str) -> float:
    """ROC AUC, or 0.5 (no better than chance) when it is undefined."""
    labels = (y_true == positive).astype(int)
    if labels.nunique() < 2 or np.ptp(score) == 0:
        return 0.5
    return float(roc_auc_score(labels, score))
