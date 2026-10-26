"""Custom exceptions for PrivSynth.

Every error raised by the toolkit inherits from :class:`PrivSynthError`, so
callers (for example the Streamlit dashboard) can catch one base class and
show a friendly message instead of a raw traceback.
"""


class PrivSynthError(Exception):
    """Base class for all PrivSynth errors."""


class DataLoadError(PrivSynthError):
    """Raised when a file cannot be found, read or parsed."""


class SchemaError(PrivSynthError):
    """Raised when columns are missing or have the wrong type."""


class PrivacyBudgetExceededError(PrivSynthError):
    """Raised when a request would spend more epsilon (or delta) than remains."""


class NotFittedError(PrivSynthError):
    """Raised when a synthesizer is asked to sample before it has been fitted."""
