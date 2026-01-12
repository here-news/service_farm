"""
REEE Explain - Trace formatting and explanation.

This module provides:
- Deterministic trace formatting (always available, no LLM)
- LLM explanation integration (optional, outside kernel)
"""

from .formatter import format_trace, format_belief_trace, TraceStyle

__all__ = [
    "format_trace",
    "format_belief_trace",
    "TraceStyle",
]
