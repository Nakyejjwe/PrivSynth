"""Building blocks of the PrivSynth Streamlit dashboard (``app.py``).

Everything that is not screen layout lives here, so it can be tested
without a browser and reused by other front ends:

* :mod:`~privsynth.dashboard.sources`  - where data comes from, schema guesses
* :mod:`~privsynth.dashboard.settings` - privacy presets and validated settings
* :mod:`~privsynth.dashboard.verdicts` - plain-language judgements of results
* :mod:`~privsynth.dashboard.charts`   - matplotlib figures
"""

from privsynth.dashboard.settings import (
    GENERATOR_INFO,
    NOISE_INFO,
    PRIVACY_LEVELS,
    PrivacyLevel,
    PrivacySettings,
    level_for,
)
from privsynth.dashboard.sources import (
    DataSource,
    SampleDataSource,
    SchemaSuggestion,
    UploadedCSVSource,
    sensitive_candidates,
    suggest_schema,
)
from privsynth.dashboard.verdicts import (
    Verdict,
    assess_disclosure,
    assess_original_risk,
    assess_utility,
    overall_recommendation,
)

__all__ = [
    "DataSource",
    "GENERATOR_INFO",
    "NOISE_INFO",
    "PRIVACY_LEVELS",
    "PrivacyLevel",
    "PrivacySettings",
    "SampleDataSource",
    "SchemaSuggestion",
    "UploadedCSVSource",
    "Verdict",
    "assess_disclosure",
    "assess_original_risk",
    "assess_utility",
    "level_for",
    "overall_recommendation",
    "sensitive_candidates",
    "suggest_schema",
]
