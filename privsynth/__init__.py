"""PrivSynth: privacy-protected synthetic data generation and evaluation.

The package is organised into four modules, connected later by a
``PrivSynthPipeline`` object:

* ``privsynth.data``       - loading datasets and describing their schema
* ``privsynth.privacy``    - re-identification risk and differential privacy
* ``privsynth.generation`` - synthetic data generators
* ``privsynth.evaluation`` - utility and risk metrics

``PrivSynthPipeline`` (``privsynth.pipeline``) runs them in order, and
``ExperimentRunner`` (``privsynth.experiments``) repeats the pipeline over
many privacy settings.
"""

from privsynth.data import AdultDataLoader, CSVLoader, Dataset
from privsynth.evaluation import (
    AssociationSimilarity,
    AttributeInferenceRisk,
    DistanceToClosestRecord,
    EvaluationReport,
    EvaluationResult,
    EvaluationSuite,
    Evaluator,
    MachineLearningUtility,
    MarginalSimilarity,
    default_evaluators,
)
from privsynth.exceptions import (
    DataLoadError,
    EvaluationError,
    NotFittedError,
    PrivacyBudgetExceededError,
    PrivSynthError,
    SchemaError,
)
from privsynth.generation import (
    BayesianNetworkSynthesizer,
    IndependentMarginalSynthesizer,
    Synthesizer,
)
from privsynth.privacy import (
    ExponentialMechanism,
    GaussianMechanism,
    Generalizer,
    KAnonymityChecker,
    LaplaceMechanism,
    PrivacyBudget,
    RiskReport,
)
from privsynth.experiments import ExperimentResults, ExperimentRunner
from privsynth.pipeline import PipelineResult, PrivSynthPipeline

__version__ = "0.3.0"

__all__ = [
    "AssociationSimilarity",
    "AttributeInferenceRisk",
    "DistanceToClosestRecord",
    "EvaluationError",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationSuite",
    "Evaluator",
    "ExperimentResults",
    "ExperimentRunner",
    "MachineLearningUtility",
    "MarginalSimilarity",
    "PipelineResult",
    "PrivSynthPipeline",
    "default_evaluators",
    "AdultDataLoader",
    "BayesianNetworkSynthesizer",
    "ExponentialMechanism",
    "GaussianMechanism",
    "IndependentMarginalSynthesizer",
    "LaplaceMechanism",
    "NotFittedError",
    "PrivacyBudget",
    "Synthesizer",
    "CSVLoader",
    "Dataset",
    "DataLoadError",
    "Generalizer",
    "KAnonymityChecker",
    "PrivacyBudgetExceededError",
    "PrivSynthError",
    "RiskReport",
    "SchemaError",
]
