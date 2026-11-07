"""Utility and privacy-risk evaluation.

Utility (is the synthetic data useful?)
    * :class:`MarginalSimilarity`     - 1-way and 2-way distributions (TVD)
    * :class:`AssociationSimilarity`  - strength of column relationships (Cramer's V)
    * :class:`MachineLearningUtility` - train on synthetic, test on real

Risk (does it still reveal real people?)
    * :class:`DistanceToClosestRecord` - near-copies of training records
    * :class:`AttributeInferenceRisk`  - guessing a person's sensitive value

:class:`EvaluationSuite` runs any list of them and returns an
:class:`EvaluationReport`.
"""

from privsynth.evaluation.base import (
    EvaluationResult,
    Evaluator,
    MixedFeatureSpace,
    encode_together,
    total_variation_distance,
)
from privsynth.evaluation.disclosure import AttributeInferenceRisk, DistanceToClosestRecord
from privsynth.evaluation.suite import EvaluationReport, EvaluationSuite, default_evaluators
from privsynth.evaluation.utility import (
    AssociationSimilarity,
    MachineLearningUtility,
    MarginalSimilarity,
    cramers_v,
)

__all__ = [
    "AssociationSimilarity",
    "AttributeInferenceRisk",
    "DistanceToClosestRecord",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationSuite",
    "Evaluator",
    "MachineLearningUtility",
    "MarginalSimilarity",
    "MixedFeatureSpace",
    "cramers_v",
    "default_evaluators",
    "encode_together",
    "total_variation_distance",
]
