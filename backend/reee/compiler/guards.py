"""
Runtime Invariant Guards for Persisted Edges

This module provides runtime validation for edges before they are persisted.
It ensures that:

1. Spine edges (SAME_HAPPENING, UPDATE_TO) can ONLY come from CompilationResult
2. All persisted spine edges have valid MembraneDecision provenance
3. Edge types match their action classification

These guards are the last line of defense - even if code bypasses the CI guard,
the runtime guard will reject invalid writes.

Usage:
    from reee.compiler.guards import validate_edge_for_persistence, SpineEdgeGuard

    # Option 1: Direct validation
    validate_edge_for_persistence(edge_type, decision)

    # Option 2: Context manager for batch writes
    with SpineEdgeGuard() as guard:
        for edge in edges:
            guard.register_edge(edge)
        guard.validate_all()  # Raises if any violations
"""

from dataclasses import dataclass
from typing import Optional, List, Set
from enum import Enum
import logging

from .membrane import Action, EdgeType, MembraneDecision


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class SpineEdgeViolationError(Exception):
    """Raised when spine edge invariants are violated at runtime."""
    pass


class UnauthorizedSpineEdgeError(SpineEdgeViolationError):
    """Raised when a spine edge is created without compiler authorization."""
    pass


class MissingProvenanceError(SpineEdgeViolationError):
    """Raised when a spine edge lacks MembraneDecision provenance."""
    pass


class ActionEdgeMismatchError(SpineEdgeViolationError):
    """Raised when edge type doesn't match the action in the decision."""
    pass


class MetabolicMembershipViolationError(SpineEdgeViolationError):
    """Raised when metabolic edges are used for case membership (union-find)."""
    pass


# =============================================================================
# Edge Classification
# =============================================================================

# Spine edges are case-forming edges that require compiler authorization
SPINE_EDGE_TYPES: Set[EdgeType] = {EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO}

# Metabolic edges are peripheral and don't form cases
METABOLIC_EDGE_TYPES: Set[EdgeType] = {EdgeType.CONTEXT_FOR}


def is_spine_edge(edge_type: EdgeType) -> bool:
    """Check if an edge type is a spine (case-forming) edge."""
    return edge_type in SPINE_EDGE_TYPES


def is_metabolic_edge(edge_type: EdgeType) -> bool:
    """Check if an edge type is a metabolic (peripheral) edge."""
    return edge_type in METABOLIC_EDGE_TYPES


# =============================================================================
# Validation Functions
# =============================================================================

def validate_edge_for_persistence(
    edge_type: EdgeType,
    decision: Optional[MembraneDecision],
    incident_a: str = "",
    incident_b: str = "",
) -> None:
    """
    Validate that an edge is authorized for persistence.

    Spine edges REQUIRE:
    1. A valid MembraneDecision with Action.MERGE
    2. Edge type matches decision.edge_type

    Args:
        edge_type: The type of edge being persisted
        decision: The MembraneDecision that authorized this edge (required for spine)
        incident_a: First incident ID (for error messages)
        incident_b: Second incident ID (for error messages)

    Raises:
        UnauthorizedSpineEdgeError: If spine edge lacks decision
        MissingProvenanceError: If decision is None for spine edge
        ActionEdgeMismatchError: If edge type doesn't match decision
    """
    pair_str = f"({incident_a}, {incident_b})" if incident_a and incident_b else ""

    # Spine edges require compiler authorization
    if is_spine_edge(edge_type):
        if decision is None:
            raise MissingProvenanceError(
                f"Spine edge {edge_type.name} {pair_str} requires MembraneDecision provenance. "
                "Spine edges can ONLY be created via compile_pair()."
            )

        if decision.action != Action.MERGE:
            raise UnauthorizedSpineEdgeError(
                f"Spine edge {edge_type.name} {pair_str} requires Action.MERGE, "
                f"but decision has {decision.action.name}. "
                "Only compile_pair() returning MERGE can create spine edges."
            )

        if decision.edge_type != edge_type:
            raise ActionEdgeMismatchError(
                f"Edge type mismatch {pair_str}: persisting {edge_type.name} "
                f"but decision specifies {decision.edge_type.name if decision.edge_type else 'None'}."
            )

    # Metabolic edges with decision should match
    elif is_metabolic_edge(edge_type) and decision is not None:
        if decision.action == Action.MERGE:
            raise ActionEdgeMismatchError(
                f"Metabolic edge {edge_type.name} {pair_str} cannot come from Action.MERGE. "
                "MERGE decisions create spine edges, not metabolic edges."
            )


def validate_decision_for_persistence(decision: MembraneDecision) -> None:
    """
    Validate that a MembraneDecision is well-formed for persistence.

    Args:
        decision: The decision to validate

    Raises:
        SpineEdgeViolationError: If decision is malformed
    """
    if decision.action == Action.MERGE:
        if decision.edge_type not in SPINE_EDGE_TYPES:
            raise ActionEdgeMismatchError(
                f"Action.MERGE must have spine edge type, got {decision.edge_type}"
            )
        if decision.confidence < 0.0 or decision.confidence > 1.0:
            raise SpineEdgeViolationError(
                f"Confidence must be in [0, 1], got {decision.confidence}"
            )

    elif decision.action == Action.PERIPHERY:
        if decision.edge_type in SPINE_EDGE_TYPES:
            raise ActionEdgeMismatchError(
                f"Action.PERIPHERY cannot have spine edge type {decision.edge_type}"
            )

    elif decision.action == Action.DEFER:
        # DEFER should not have edge_type set
        pass

    elif decision.action == Action.DISTINCT:
        # DISTINCT has no edge
        pass


# =============================================================================
# Metabolic Edge Isolation Check
# =============================================================================

def assert_metabolic_edges_not_in_membership(
    spine_edges: List["CompiledEdge"],
    metabolic_edges: List["CompiledEdge"],
    case_incident_ids: Set[str],
) -> None:
    """
    Validate that case membership was formed ONLY from spine edges.

    This is a critical invariant check that ensures metabolic edges (CONTEXT_FOR)
    never contributed to union-find case formation. Case membership must be
    strictly determined by spine edges (SAME_HAPPENING, UPDATE_TO).

    Args:
        spine_edges: List of spine edges that formed the case
        metabolic_edges: List of metabolic edges (should not affect membership)
        case_incident_ids: Set of incident IDs in the case

    Raises:
        MetabolicMembershipViolationError: If metabolic edges could have caused membership
    """
    # Build the set of incidents that are connected by spine edges
    spine_connected = set()
    for edge in spine_edges:
        spine_connected.add(edge.incident_a)
        spine_connected.add(edge.incident_b)

    # Check each metabolic edge
    for edge in metabolic_edges:
        a_in_case = edge.incident_a in case_incident_ids
        b_in_case = edge.incident_b in case_incident_ids

        # If both endpoints are in the case but NOT connected by spine edges,
        # the metabolic edge might have been used for membership
        if a_in_case and b_in_case:
            # Check if they're both reachable via spine edges
            a_spine_connected = edge.incident_a in spine_connected
            b_spine_connected = edge.incident_b in spine_connected

            # If one or both aren't spine-connected, we have a problem
            if not (a_spine_connected and b_spine_connected):
                raise MetabolicMembershipViolationError(
                    f"Metabolic edge ({edge.incident_a}, {edge.incident_b}) "
                    f"has both endpoints in case but not connected via spine. "
                    f"This suggests metabolic edges were used for case formation."
                )


def validate_case_spine_only(
    case_incident_ids: Set[str],
    spine_edges: List["CompiledEdge"],
) -> bool:
    """
    Verify that a case's membership can be fully explained by spine edges alone.

    Uses union-find on spine edges only and checks that the resulting component
    matches the case membership exactly.

    Args:
        case_incident_ids: Set of incident IDs that are in the case
        spine_edges: Spine edges that should define the case

    Returns:
        True if case is spine-only, raises if not

    Raises:
        MetabolicMembershipViolationError: If case membership doesn't match spine-only union-find
    """
    if not case_incident_ids:
        return True

    if not spine_edges:
        # No spine edges but we have incidents - only valid if single incident
        if len(case_incident_ids) == 1:
            return True
        raise MetabolicMembershipViolationError(
            f"Case has {len(case_incident_ids)} incidents but no spine edges. "
            "Multi-incident cases require spine edge justification."
        )

    # Build union-find from spine edges only
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Initialize all case incidents
    for iid in case_incident_ids:
        find(iid)

    # Apply spine edges
    for edge in spine_edges:
        if edge.incident_a in case_incident_ids and edge.incident_b in case_incident_ids:
            union(edge.incident_a, edge.incident_b)

    # Check: all case incidents should be in the same component
    if case_incident_ids:
        root = find(next(iter(case_incident_ids)))
        for iid in case_incident_ids:
            if find(iid) != root:
                raise MetabolicMembershipViolationError(
                    f"Case membership cannot be explained by spine edges alone. "
                    f"Incident {iid} has different root than expected. "
                    "This suggests metabolic edges were used for union-find."
                )

    return True


# =============================================================================
# Batch Validation Context Manager
# =============================================================================

@dataclass
class PendingEdge:
    """An edge pending validation and persistence."""
    incident_a: str
    incident_b: str
    edge_type: EdgeType
    decision: Optional[MembraneDecision]


class SpineEdgeGuard:
    """
    Context manager for batch edge validation before persistence.

    Usage:
        with SpineEdgeGuard() as guard:
            for compiled_edge in compilation_result.all_edges:
                if compiled_edge.decision.action == Action.MERGE:
                    guard.register_edge(
                        incident_a=compiled_edge.incident_a,
                        incident_b=compiled_edge.incident_b,
                        edge_type=compiled_edge.decision.edge_type,
                        decision=compiled_edge.decision,
                    )
            guard.validate_all()  # Raises on first violation

        # After exiting the context, all edges have been validated
        # Now safe to persist to Neo4j
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, raise on any violation. If False, log warnings.
        """
        self.strict = strict
        self.pending: List[PendingEdge] = []
        self.violations: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't suppress exceptions
        return False

    def register_edge(
        self,
        incident_a: str,
        incident_b: str,
        edge_type: EdgeType,
        decision: Optional[MembraneDecision],
    ) -> None:
        """Register an edge for batch validation."""
        self.pending.append(PendingEdge(
            incident_a=incident_a,
            incident_b=incident_b,
            edge_type=edge_type,
            decision=decision,
        ))

    def validate_all(self) -> bool:
        """
        Validate all registered edges.

        Returns:
            True if all edges are valid, False if violations found (strict=False)

        Raises:
            SpineEdgeViolationError: If any violation found (strict=True)
        """
        self.violations = []

        for edge in self.pending:
            try:
                validate_edge_for_persistence(
                    edge_type=edge.edge_type,
                    decision=edge.decision,
                    incident_a=edge.incident_a,
                    incident_b=edge.incident_b,
                )
            except SpineEdgeViolationError as e:
                if self.strict:
                    raise
                self.violations.append(str(e))
                logger.warning(f"Edge validation warning: {e}")

        if self.violations and not self.strict:
            logger.warning(
                f"SpineEdgeGuard found {len(self.violations)} violations "
                f"(strict=False, continuing)"
            )

        return len(self.violations) == 0

    def get_violations(self) -> List[str]:
        """Get list of violation messages (only populated after validate_all)."""
        return self.violations.copy()


# =============================================================================
# Neo4j Write Interceptor (for integration)
# =============================================================================

class GuardedNeo4jWriter:
    """
    Wrapper around Neo4j edge writes that enforces compiler authority.

    This is the production-grade runtime guard. All edge writes should
    go through this class to ensure invariants are maintained.

    Usage:
        writer = GuardedNeo4jWriter(neo4j_service)
        await writer.write_spine_edge(compiled_edge)  # Validates first
    """

    def __init__(self, neo4j_service):
        self.neo4j = neo4j_service

    async def write_spine_edge(
        self,
        incident_a: str,
        incident_b: str,
        edge_type: EdgeType,
        decision: MembraneDecision,
        properties: dict = None,
    ) -> None:
        """
        Write a spine edge with validation.

        Args:
            incident_a: Source incident ID
            incident_b: Target incident ID
            edge_type: Type of spine edge
            decision: The MembraneDecision authorizing this edge
            properties: Additional edge properties

        Raises:
            SpineEdgeViolationError: If edge is not authorized
        """
        # Validate first
        validate_edge_for_persistence(
            edge_type=edge_type,
            decision=decision,
            incident_a=incident_a,
            incident_b=incident_b,
        )
        validate_decision_for_persistence(decision)

        # Build properties with provenance
        props = {
            "confidence": decision.confidence,
            "reason": decision.reason,
            "witnesses": list(decision.witnesses) if decision.witnesses else [],
            "compiler_authorized": True,  # Mark as validated
        }
        if properties:
            props.update(properties)

        # Write to Neo4j
        query = f"""
        MATCH (a:Incident {{id: $incident_a}})
        MATCH (b:Incident {{id: $incident_b}})
        MERGE (a)-[r:{edge_type.name}]->(b)
        SET r += $properties
        RETURN r
        """
        await self.neo4j._execute_write(
            query,
            {
                "incident_a": incident_a,
                "incident_b": incident_b,
                "properties": props,
            }
        )

    async def write_metabolic_edge(
        self,
        incident_a: str,
        incident_b: str,
        edge_type: EdgeType,
        properties: dict = None,
    ) -> None:
        """
        Write a metabolic edge.

        Metabolic edges don't require compiler authorization but are
        validated to ensure they're not accidentally spine edges.
        """
        if is_spine_edge(edge_type):
            raise UnauthorizedSpineEdgeError(
                f"Cannot write spine edge {edge_type.name} via write_metabolic_edge(). "
                "Use write_spine_edge() with MembraneDecision."
            )

        props = properties or {}
        query = f"""
        MATCH (a:Incident {{id: $incident_a}})
        MATCH (b:Incident {{id: $incident_b}})
        MERGE (a)-[r:{edge_type.name}]->(b)
        SET r += $properties
        RETURN r
        """
        await self.neo4j._execute_write(
            query,
            {
                "incident_a": incident_a,
                "incident_b": incident_b,
                "properties": props,
            }
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Exceptions
    "SpineEdgeViolationError",
    "UnauthorizedSpineEdgeError",
    "MissingProvenanceError",
    "ActionEdgeMismatchError",
    "MetabolicMembershipViolationError",
    # Classification
    "SPINE_EDGE_TYPES",
    "METABOLIC_EDGE_TYPES",
    "is_spine_edge",
    "is_metabolic_edge",
    # Validation
    "validate_edge_for_persistence",
    "validate_decision_for_persistence",
    # Metabolic isolation
    "assert_metabolic_edges_not_in_membership",
    "validate_case_spine_only",
    # Batch validation
    "SpineEdgeGuard",
    # Integration
    "GuardedNeo4jWriter",
]
