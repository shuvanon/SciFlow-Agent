"""Application-specific exception types.

Every error shown to a user should originate from one of these classes so the
UI can distinguish expected, explainable failures from genuine bugs.
"""

from __future__ import annotations


class SciFlowError(Exception):
    """Base class for all SciFlow Agent errors."""


class ImageValidationError(SciFlowError):
    """Raised when an uploaded or selected image cannot be accepted."""


class ToolInputError(SciFlowError):
    """Raised when a tool receives an invalid image or parameter value."""


class UnknownToolError(SciFlowError):
    """Raised when a plan references a tool that is not in the registry."""


class PlannerError(SciFlowError):
    """Raised when planning fails (connection, configuration, or invalid output).

    The message is always safe to show to the user: it never contains
    secrets, and it includes guidance (retry, rephrase, or switch to demo
    mode) where applicable.
    """
