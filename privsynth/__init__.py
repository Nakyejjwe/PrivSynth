"""PrivSynth: privacy-protected synthetic data generation and evaluation.

The package is organised into four modules, connected later by a
``PrivSynthPipeline`` object:

* ``privsynth.data``       - loading datasets and describing their schema
* ``privsynth.privacy``    - re-identification risk and differential privacy
* ``privsynth.generation`` - synthetic data generators (Week 2)
* ``privsynth.evaluation`` - utility and risk metrics (Week 3)
"""

from privsynth.data import AdultDataLoader, CSVLoader, Dataset
from privsynth.exceptions import (
    DataLoadError,
    PrivacyBudgetExceededError,
    PrivSynthError,
    SchemaError,
)
from privsynth.privacy import Generalizer, KAnonymityChecker, RiskReport

__version__ = "0.1.0"

__all__ = [
    "AdultDataLoader",
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
