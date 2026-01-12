"""
Artifact Extraction Package

Transforms raw incident data into typed artifacts for the membrane compiler.
This is the boundary between LLM/perception and deterministic compilation.
"""

from .extractor import (
    # Core function
    extract_artifact,
    # Schema
    EntityClassification,
    ReferentType,
    ExtractionResult,
    # Inquiry seeds
    InquirySeed,
)

__all__ = [
    "extract_artifact",
    "EntityClassification",
    "ReferentType",
    "ExtractionResult",
    "InquirySeed",
]
