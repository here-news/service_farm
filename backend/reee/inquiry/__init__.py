"""
REEE Inquiry MVP
================

Inquiry layer that wraps REEE for user-facing "is this true?" questions.

Components:
- Inquiry: User-scoped question with typed target
- InquiryEngine: Orchestrates REEE for inquiries
- Contribution: User submission (evidence, refutation, etc.)
"""

from .types import (
    Inquiry, InquirySchema, InquiryStatus, RigorLevel,
    Contribution, ContributionType,
    InquiryTask, TaskType
)
from .engine import InquiryEngine

__all__ = [
    'Inquiry',
    'InquirySchema',
    'InquiryStatus',
    'RigorLevel',
    'Contribution',
    'ContributionType',
    'InquiryTask',
    'TaskType',
    'InquiryEngine',
]
