"""
REEE Kernel - Pure epistemic computation.

This module contains pure functions for topology emergence.
NO database imports. NO LLM calls. NO network access.

All state is passed in via contracts, changes returned via TopologyDelta.
"""

from .scope import compute_scope_id, DEFAULT_HUB_ENTITIES, extract_primary_anchors
from .question_key import (
    extract_question_key,
    FallbackLevel,
    QuestionKeyResult,
    DEATH_PATTERNS,
    INJURY_PATTERNS,
    STATUS_PATTERNS,
    POLICY_PATTERNS,
)
from .surface_update import (
    compute_surface_key,
    apply_claim_to_surface,
    SurfaceKeyParams,
    SurfaceKeyResult,
)
from .incident_routing import (
    find_candidates,
    decide_route,
    RoutingParams,
    RoutingResult,
    RouteOutcome,
    CandidateScore,
)
from .topology_kernel import TopologyKernel, KernelParams

__all__ = [
    # Scope computation
    "compute_scope_id",
    "DEFAULT_HUB_ENTITIES",
    "extract_primary_anchors",
    # Question key extraction
    "extract_question_key",
    "FallbackLevel",
    "QuestionKeyResult",
    "DEATH_PATTERNS",
    "INJURY_PATTERNS",
    "STATUS_PATTERNS",
    "POLICY_PATTERNS",
    # Surface update
    "compute_surface_key",
    "apply_claim_to_surface",
    "SurfaceKeyParams",
    "SurfaceKeyResult",
    # Incident routing
    "find_candidates",
    "decide_route",
    "RoutingParams",
    "RoutingResult",
    "RouteOutcome",
    "CandidateScore",
    # Topology kernel
    "TopologyKernel",
    "KernelParams",
]
