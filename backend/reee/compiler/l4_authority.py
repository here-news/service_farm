"""
L4 Structural Authority: Weaver-Owned Case Persistence

This module defines the contract for L4 case persistence where:
- WEAVER is the sole authority for case membership (incident_ids)
- CANONICAL WORKER only enriches presentation (title, description)

Key invariants:
1. Case.incident_ids can ONLY be modified by weaver (via SpineEdgeGuard-validated writes)
2. Case.title, Case.description can be modified by canonical worker
3. Spine edges (SAME_HAPPENING, UPDATE_TO) MUST have MembraneDecision provenance
4. Metabolic edges (CONTEXT_FOR) never affect case membership

The release criterion:
    "canonical worker must be able to delete and rerun without changing
    any case membership—only presentation changes."

Usage:
    # In weaver:
    from reee.compiler.l4_authority import (
        persist_case_structure,    # Persists membership-defining fields
        emit_case_changed_event,   # Notifies canonical worker
    )

    # In canonical worker:
    from reee.compiler.l4_authority import (
        persist_case_presentation,  # Persists presentation-only fields
        CasePresentationUpdate,     # Type for presentation updates
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Set, List, Optional, Dict, Any
import json
import logging

from .guards import (
    SpineEdgeGuard,
    validate_edge_for_persistence,
    is_spine_edge,
)
from .weaver_compiler import CompilationResult, Case as CompilerCase, CompiledEdge

logger = logging.getLogger(__name__)


# =============================================================================
# Event Schema for Weaver → Canonical Worker Communication
# =============================================================================

@dataclass
class CaseChangedEvent:
    """
    Event emitted by weaver when case membership changes.

    Canonical worker subscribes to this and enriches with title/description.
    """
    event_type: str  # "case_formed" | "case_updated" | "case_dissolved"
    timestamp: str
    case_id: str
    kernel_signature: str
    incident_ids: List[str]
    core_entities: List[str]
    is_new: bool = True
    compiler_version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "type": self.event_type,
            "timestamp": self.timestamp,
            "case_id": self.case_id,
            "kernel_signature": self.kernel_signature,
            "incident_ids": self.incident_ids,
            "core_entities": self.core_entities,
            "is_new": self.is_new,
            "compiler_version": self.compiler_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class CasePresentationUpdate:
    """
    Presentation-only fields that canonical worker can modify.

    These fields NEVER affect case membership.
    """
    case_id: str
    title: str = ""
    description: str = ""
    case_type: str = "general"  # breaking, developing, ongoing

    # Computed metrics (read from Neo4j, don't modify structure)
    surface_count: int = 0
    source_count: int = 0
    claim_count: int = 0


# =============================================================================
# Weaver Authority: Persist Case Structure
# =============================================================================

async def persist_case_structure(
    neo4j,
    case: CompilerCase,
    compilation_result: CompilationResult,
    guard: Optional[SpineEdgeGuard] = None,
) -> bool:
    """
    Persist case structure (membership-defining fields) to Neo4j.

    This is the ONLY function authorized to modify case.incident_ids.
    All spine edges must have been validated by SpineEdgeGuard.

    Args:
        neo4j: Neo4jService instance
        case: CompilerCase from compilation result
        compilation_result: Full result for provenance
        guard: Optional SpineEdgeGuard (validates before persist)

    Returns:
        True if case was newly created, False if updated
    """
    # If guard provided, validate all spine edges
    if guard:
        for edge in case.spine_edges:
            guard.register_edge(
                incident_a=edge.incident_a,
                incident_b=edge.incident_b,
                edge_type=edge.decision.edge_type,
                decision=edge.decision,
            )
        guard.validate_all()  # Raises on violation

    # Check if case exists
    existing = await neo4j._execute_read("""
        MATCH (c:Case {kernel_signature: $sig})
        RETURN c.id as id
    """, {"sig": case.case_id})

    is_new = not existing

    # Persist case with structural fields (WEAVER AUTHORITY)
    await neo4j._execute_write("""
        MERGE (c:Case {kernel_signature: $kernel_sig})
        ON CREATE SET
            c.id = $id,
            c.created_at = datetime(),
            c.created_by = 'weaver'
        SET
            c.incident_ids = $incident_ids,
            c.core_entities = $core_entities,
            c.structural_updated_at = datetime(),
            c.compiler_version = $compiler_version,
            c.spine_edge_count = $spine_edge_count
        WITH c
        // Manage CONTAINS relationships (structural)
        OPTIONAL MATCH (c)-[old:CONTAINS]->(:Incident)
        DELETE old
        WITH c
        UNWIND $incident_ids as inc_id
        MATCH (i:Incident {id: inc_id})
        MERGE (c)-[:CONTAINS {source: 'weaver'}]->(i)
    """, {
        "id": case.case_id,
        "kernel_sig": case.case_id,
        "incident_ids": list(case.incident_ids),
        "core_entities": list(case.core_entities) if hasattr(case, 'core_entities') else [],
        "compiler_version": compilation_result.stats.get("compiler_version", "1.0.0"),
        "spine_edge_count": len(case.spine_edges),
    })

    # Persist spine edges with provenance
    for edge in case.spine_edges:
        await _persist_spine_edge(neo4j, edge)

    return is_new


async def _persist_spine_edge(neo4j, edge: CompiledEdge) -> None:
    """Persist a spine edge with full MembraneDecision provenance."""
    decision = edge.decision
    edge_type = decision.edge_type.name if decision.edge_type else "SAME_HAPPENING"

    # Build provenance properties
    props = {
        "confidence": decision.confidence,
        "reason": decision.reason,
        "witnesses": list(decision.witnesses) if decision.witnesses else [],
        "compiler_authorized": True,
        "persisted_at": datetime.utcnow().isoformat(),
    }

    # Add shared referent info
    if decision.shared_referents:
        props["shared_referent_count"] = len(decision.shared_referents)
        props["shared_referent_ids"] = [
            r.entity_id for r in list(decision.shared_referents)[:5]
        ]

    query = f"""
        MATCH (a:Incident {{id: $incident_a}})
        MATCH (b:Incident {{id: $incident_b}})
        MERGE (a)-[r:{edge_type}]->(b)
        SET r += $properties
        RETURN r
    """

    await neo4j._execute_write(query, {
        "incident_a": edge.incident_a,
        "incident_b": edge.incident_b,
        "properties": props,
    })


async def persist_metabolic_edges(
    neo4j,
    case: CompilerCase,
    compiler_version: str = "1.0.0",
) -> int:
    """
    Persist metabolic edges (CONTEXT_FOR) with provenance.

    Metabolic edges do NOT affect case membership. They represent contextual
    relationships between incidents that share entities but are not part
    of the same happening.

    Critical invariant: Metabolic edges must NEVER be used for union-find
    or case formation. They are peripheral, not structural.

    Args:
        neo4j: Neo4jService instance
        case: CompilerCase containing metabolic_edges
        compiler_version: Version string for provenance

    Returns:
        Number of metabolic edges persisted
    """
    persisted = 0

    for edge in case.metabolic_edges:
        decision = edge.decision
        edge_type = decision.edge_type.name if decision.edge_type else "CONTEXT_FOR"

        # Build provenance properties (similar to spine but marked as metabolic)
        props = {
            "confidence": decision.confidence,
            "reason": decision.reason,
            "is_metabolic": True,  # CRITICAL: Mark as non-structural
            "compiler_authorized": False,  # Metabolic edges don't need compiler auth
            "persisted_at": datetime.utcnow().isoformat(),
            "compiler_version": compiler_version,
        }

        # Add artifact context if available
        if decision.shared_referents:
            props["shared_context_count"] = len(decision.shared_referents)
            props["shared_context_ids"] = [
                r.entity_id for r in list(decision.shared_referents)[:5]
            ]

        query = f"""
            MATCH (a:Incident {{id: $incident_a}})
            MATCH (b:Incident {{id: $incident_b}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r += $properties
            RETURN r
        """

        await neo4j._execute_write(query, {
            "incident_a": edge.incident_a,
            "incident_b": edge.incident_b,
            "properties": props,
        })
        persisted += 1

    logger.debug(f"Persisted {persisted} metabolic edges for case {case.case_id}")
    return persisted


async def emit_case_changed_event(
    redis_client,
    case: CompilerCase,
    is_new: bool,
    channel: str = "weaver_events",
) -> None:
    """
    Emit case_changed event to Redis for canonical worker.

    Args:
        redis_client: Redis client instance
        case: CompilerCase that changed
        is_new: True if newly created, False if updated
        channel: Redis pubsub channel
    """
    if not redis_client:
        logger.debug("No Redis client - skipping event emission")
        return

    event = CaseChangedEvent(
        event_type="case_formed" if is_new else "case_updated",
        timestamp=datetime.utcnow().isoformat(),
        case_id=case.case_id,
        kernel_signature=case.case_id,
        incident_ids=list(case.incident_ids),
        core_entities=[],  # Will be populated from incidents
        is_new=is_new,
    )

    try:
        await redis_client.publish(channel, event.to_json())
        logger.debug(f"Emitted {event.event_type} for case {case.case_id}")
    except Exception as e:
        logger.warning(f"Failed to emit case event: {e}")


# =============================================================================
# Canonical Worker Authority: Persist Presentation ONLY
# =============================================================================

async def persist_case_presentation(
    neo4j,
    update: CasePresentationUpdate,
) -> None:
    """
    Persist presentation-only fields to an existing case.

    This function CANNOT modify:
    - incident_ids
    - kernel_signature
    - CONTAINS relationships
    - spine edges

    It CAN only modify:
    - title, description, case_type
    - Computed metrics (surface_count, etc.)

    Args:
        neo4j: Neo4jService instance
        update: CasePresentationUpdate with new values
    """
    # CRITICAL: This query NEVER touches incident_ids or structural relationships
    await neo4j._execute_write("""
        MATCH (c:Case {id: $case_id})
        SET
            c.title = $title,
            c.canonical_title = $title,
            c.description = $description,
            c.case_type = $case_type,
            c.surface_count = $surface_count,
            c.source_count = $source_count,
            c.claim_count = $claim_count,
            c.presentation_updated_at = datetime(),
            c.presentation_updated_by = 'canonical_worker'
        // DO NOT touch incident_ids or CONTAINS relationships
    """, {
        "case_id": update.case_id,
        "title": update.title,
        "description": update.description,
        "case_type": update.case_type,
        "surface_count": update.surface_count,
        "source_count": update.source_count,
        "claim_count": update.claim_count,
    })


async def get_case_for_enrichment(neo4j, case_id: str) -> Optional[Dict[str, Any]]:
    """
    Load case data for canonical worker enrichment.

    Returns the case with its incidents but does NOT allow modification
    of structural fields.
    """
    result = await neo4j._execute_read("""
        MATCH (c:Case {id: $case_id})
        OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
        WITH c, collect(i.id) as incident_ids, collect(i.anchor_entities) as all_anchors
        RETURN
            c.id as id,
            c.kernel_signature as kernel_signature,
            incident_ids,
            c.title as title,
            c.description as description,
            c.case_type as case_type,
            all_anchors
    """, {"case_id": case_id})

    if not result:
        return None

    row = result[0]
    return {
        "id": row["id"],
        "kernel_signature": row["kernel_signature"],
        "incident_ids": row["incident_ids"],
        "title": row.get("title"),
        "description": row.get("description"),
        "case_type": row.get("case_type"),
        "anchor_entities": set().union(*[set(a or []) for a in row.get("all_anchors", [])]),
    }


# =============================================================================
# Invariant Checks
# =============================================================================

async def verify_case_membership_unchanged(
    neo4j,
    case_id: str,
    expected_incident_ids: Set[str],
) -> bool:
    """
    Verify that a case's membership hasn't been modified unexpectedly.

    Use this after canonical worker runs to ensure it didn't touch structure.
    """
    result = await neo4j._execute_read("""
        MATCH (c:Case {id: $case_id})
        RETURN c.incident_ids as incident_ids
    """, {"case_id": case_id})

    if not result:
        logger.error(f"Case {case_id} not found!")
        return False

    actual_ids = set(result[0]["incident_ids"] or [])

    if actual_ids != expected_incident_ids:
        logger.error(
            f"Case membership changed! Expected {expected_incident_ids}, "
            f"got {actual_ids}. Diff: +{actual_ids - expected_incident_ids}, "
            f"-{expected_incident_ids - actual_ids}"
        )
        return False

    return True


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Events
    "CaseChangedEvent",
    "CasePresentationUpdate",
    # Weaver authority
    "persist_case_structure",
    "persist_metabolic_edges",
    "emit_case_changed_event",
    # Canonical worker authority
    "persist_case_presentation",
    "get_case_for_enrichment",
    # Invariants
    "verify_case_membership_unchanged",
]
