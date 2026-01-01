"""
REEE Inquiry MVP
================

Inquiry layer that wraps REEE for user-facing "is this true?" questions.

Components:
- Inquiry: User-scoped question with typed target
- InquiryEngine: Orchestrates REEE for inquiries
- Contribution: User submission (evidence, refutation, etc.)
- ProtoInquiry: System-generated inquiry from surface + meta-claim
- InquirySeeder: Seeds proto-inquiries from weaver outputs
"""

from .types import (
    Inquiry, InquirySchema, InquiryStatus, RigorLevel,
    Contribution, ContributionType,
    InquiryTask, TaskType
)
from .engine import InquiryEngine
from .seeder import (
    ProtoInquiry, ProtoInquiryType, SchemaType,
    ViewId, MetaClaimRef, ScopeSignature,
    InquirySeeder, QUESTION_TEMPLATES
)

__all__ = [
    # User inquiries (REEE2 contract)
    'Inquiry',
    'InquirySchema',
    'InquiryStatus',
    'RigorLevel',
    'Contribution',
    'ContributionType',
    'InquiryTask',
    'TaskType',
    'InquiryEngine',
    # Proto-inquiries (REEE1 emergence)
    'ProtoInquiry',
    'ProtoInquiryType',
    'SchemaType',
    'ViewId',
    'MetaClaimRef',
    'ScopeSignature',
    'InquirySeeder',
    'QUESTION_TEMPLATES',
]
