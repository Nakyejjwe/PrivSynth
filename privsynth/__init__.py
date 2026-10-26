"""PrivSynth: privacy-protected synthetic data generation and evaluation.

The package is organised into four modules, connected later by a
``PrivSynthPipeline`` object:

* ``privsynth.data``       - loading datasets and describing their schema
* ``privsynth.privacy``    - re-identification risk and differential privacy
* ``privsynth.generation`` - synthetic data generators
* ``privsynth.evaluation`` - utility and risk metrics (Week 3)
"""

from privsynth.data import AdultDataLoader, CSVLoader, Dataset
from privsynth.exceptions import (
    DataLoadError,
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

__version__ = "0.2.0"

__all__ = [
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
