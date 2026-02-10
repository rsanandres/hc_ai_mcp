"""Context variables for auto-injecting values into tool calls.

This module provides a mechanism to pass patient_id (and potentially other context)
to tools without requiring the LLM to explicitly pass it in every call.

Usage:
    # In agent node (before tool execution):
    from agent.tools.context import set_patient_context
    set_patient_context(state.get("patient_id"))

    # In tool (to get patient_id if not provided):
    from agent.tools.context import get_patient_context
    if not patient_id:
        patient_id = get_patient_context()
"""

from contextvars import ContextVar
from typing import Optional


# Context variable for patient_id - thread-safe for async operations
_current_patient_id: ContextVar[Optional[str]] = ContextVar(
    'current_patient_id',
    default=None
)


def set_patient_context(patient_id: Optional[str]) -> None:
    """Set the current patient context for tool auto-injection.

    Args:
        patient_id: The patient UUID to use for subsequent tool calls,
                   or None to clear the context.
    """
    _current_patient_id.set(patient_id)


def get_patient_context() -> Optional[str]:
    """Get the current patient context.

    Returns:
        The patient UUID if set, None otherwise.
    """
    return _current_patient_id.get()


def clear_patient_context() -> None:
    """Clear the patient context (sets to None)."""
    _current_patient_id.set(None)
