"""Synthetic data generators.

* :class:`IndependentMarginalSynthesizer` - generator 1, the baseline
* :class:`BayesianNetworkSynthesizer`     - generator 2, keeps column relationships
"""

from privsynth.generation.base import Synthesizer
from privsynth.generation.bayesian import BayesianNetworkSynthesizer
from privsynth.generation.encoding import ColumnDomain, TableEncoder
from privsynth.generation.marginal import IndependentMarginalSynthesizer

#: Generators available by name (for the pipeline and dashboard drop-down).
SYNTHESIZERS: dict[str, type[Synthesizer]] = {
    "independent": IndependentMarginalSynthesizer,
    "bayesian": BayesianNetworkSynthesizer,
}

__all__ = [
    "BayesianNetworkSynthesizer",
    "ColumnDomain",
    "IndependentMarginalSynthesizer",
    "SYNTHESIZERS",
    "Synthesizer",
    "TableEncoder",
]
