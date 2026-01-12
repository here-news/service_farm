"""
REEE Compiler Package

The membrane compiler is the sole authority for topology mutations.
No other code may create spine edges or merge cases.

Architecture:
- membrane.py: Deterministic decision rules (compile_pair)
- artifacts/: LLM-based entity classification (extract_artifact)
- weaver_compiler.py: Orchestration (artifact extraction → membrane → cases)
"""

from .membrane import (
    # Structural language
    Action,
    EdgeType,
    ReferentRole,
    # Artifacts
    Referent,
    IncidentArtifact,
    # Decisions
    MembraneDecision,
    # Parameters
    CompilerParams,
    DEFAULT_PARAMS,
    # Core function
    compile_pair,
    # Testing
    assert_invariants,
    # Thresholds (defaults)
    CONFIDENCE_THRESHOLD_MERGE,
    CONFIDENCE_THRESHOLD_PERIPHERY,
    TIME_WITNESS_WINDOW_SECONDS,
    UPDATE_THRESHOLD_SECONDS,
)

from .weaver_compiler import (
    # Output schema
    CompiledEdge,
    Case,
    CompilationResult,
    # Main functions
    compile_incidents,
    compile_incremental,
    # Utilities
    UnionFind,
)

from .artifacts import (
    extract_artifact,
    EntityClassification,
    ReferentType,
    ExtractionResult,
    InquirySeed,
)

from .guards import (
    # Exceptions
    SpineEdgeViolationError,
    UnauthorizedSpineEdgeError,
    MissingProvenanceError,
    ActionEdgeMismatchError,
    MetabolicMembershipViolationError,
    # Classification
    SPINE_EDGE_TYPES,
    METABOLIC_EDGE_TYPES,
    is_spine_edge,
    is_metabolic_edge,
    # Validation
    validate_edge_for_persistence,
    validate_decision_for_persistence,
    # Metabolic isolation
    assert_metabolic_edges_not_in_membership,
    validate_case_spine_only,
    # Batch validation
    SpineEdgeGuard,
    # Integration
    GuardedNeo4jWriter,
)

from .l4_authority import (
    # Events
    CaseChangedEvent,
    CasePresentationUpdate,
    # Weaver authority
    persist_case_structure,
    persist_metabolic_edges,
    emit_case_changed_event,
    # Canonical worker authority
    persist_case_presentation,
    get_case_for_enrichment,
    # Invariants
    verify_case_membership_unchanged,
)

__all__ = [
    # Membrane (structural language)
    "Action",
    "EdgeType",
    "ReferentRole",
    "Referent",
    "IncidentArtifact",
    "MembraneDecision",
    "CompilerParams",
    "DEFAULT_PARAMS",
    "compile_pair",
    "assert_invariants",
    "CONFIDENCE_THRESHOLD_MERGE",
    "CONFIDENCE_THRESHOLD_PERIPHERY",
    "TIME_WITNESS_WINDOW_SECONDS",
    "UPDATE_THRESHOLD_SECONDS",
    # Weaver compiler (orchestration)
    "CompiledEdge",
    "Case",
    "CompilationResult",
    "compile_incidents",
    "compile_incremental",
    "UnionFind",
    # Artifacts (extraction)
    "extract_artifact",
    "EntityClassification",
    "ReferentType",
    "ExtractionResult",
    "InquirySeed",
    # Guards (runtime invariants)
    "SpineEdgeViolationError",
    "UnauthorizedSpineEdgeError",
    "MissingProvenanceError",
    "ActionEdgeMismatchError",
    "MetabolicMembershipViolationError",
    "SPINE_EDGE_TYPES",
    "METABOLIC_EDGE_TYPES",
    "is_spine_edge",
    "is_metabolic_edge",
    "validate_edge_for_persistence",
    "validate_decision_for_persistence",
    "assert_metabolic_edges_not_in_membership",
    "validate_case_spine_only",
    "SpineEdgeGuard",
    "GuardedNeo4jWriter",
    # L4 Authority (case persistence)
    "CaseChangedEvent",
    "CasePresentationUpdate",
    "persist_case_structure",
    "persist_metabolic_edges",
    "emit_case_changed_event",
    "persist_case_presentation",
    "get_case_for_enrichment",
    "verify_case_membership_unchanged",
]
