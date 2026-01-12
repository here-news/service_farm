"""
DeltaApplier - Apply TopologyDelta to Neo4j.

Converts kernel output back to DB operations:
- Upsert Surface nodes using kernel signature (MERGE key)
- Upsert Event/Incident nodes using kernel signature
- Create edges (CONTAINS, PART_OF)
- Persist traces/signals (optional - can log first)

CRITICAL: Neo4j MERGE keys must match kernel's deterministic signatures.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from services.neo4j_service import Neo4jService

from ..contracts.state import SurfaceState, IncidentState
from ..contracts.delta import TopologyDelta, Link
from ..contracts.traces import DecisionTrace

logger = logging.getLogger(__name__)


class DeltaApplier:
    """Apply TopologyDelta to Neo4j.

    Thin adapter - converts kernel output to DB writes.
    Uses kernel signatures as MERGE keys for idempotent upserts.
    """

    def __init__(self, neo4j: Neo4jService):
        self.neo4j = neo4j

    async def apply(
        self,
        delta: TopologyDelta,
        persist_traces: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Apply delta to Neo4j.

        Args:
            delta: TopologyDelta from kernel
            persist_traces: If True, store traces to Neo4j
            dry_run: If True, only log what would be done

        Returns:
            Summary of applied changes
        """
        summary = {
            "surfaces_upserted": 0,
            "incidents_upserted": 0,
            "links_created": 0,
            "traces_persisted": 0,
            "dry_run": dry_run,
        }

        if dry_run:
            logger.info(f"[DRY RUN] Would apply: {delta.to_summary()}")
            return summary

        # Upsert surfaces
        for surface in delta.surface_upserts:
            await self._upsert_surface(surface)
            summary["surfaces_upserted"] += 1

        # Upsert incidents
        for incident in delta.incident_upserts:
            await self._upsert_incident(incident)
            summary["incidents_upserted"] += 1

        # Create links
        for link in delta.links:
            await self._create_link(link)
            summary["links_created"] += 1

        # Persist traces (optional)
        if persist_traces:
            for trace in delta.decision_traces:
                await self._persist_trace(trace)
                summary["traces_persisted"] += 1

        logger.info(f"Applied delta: {summary}")
        return summary

    async def _upsert_surface(self, surface: SurfaceState) -> None:
        """Upsert Surface node using kernel signature.

        MERGE key: kernel_signature (deterministic from scope_id + question_key)
        This ensures convergence across rebuilds.
        """
        await self.neo4j._execute_write("""
            MERGE (s:Surface {kernel_signature: $signature})
            ON CREATE SET
                s.id = $id,
                s.scope_id = $scope_id,
                s.question_key = $question_key,
                s.created_at = datetime()
            SET s.claim_count = $claim_count,
                s.entities = $entities,
                s.anchor_entities = $anchors,
                s.sources = $sources,
                s.time_start = $time_start,
                s.time_end = $time_end,
                s.params_hash = $params_hash,
                s.kernel_version = $kernel_version,
                s.updated_at = datetime()
        """, {
            'signature': surface.key.signature,
            'id': f"sf_{surface.key.signature[3:]}",  # sf_ + hash
            'scope_id': surface.key.scope_id,
            'question_key': surface.key.question_key,
            'claim_count': len(surface.claim_ids),
            'entities': list(surface.entities),
            'anchors': list(surface.anchor_entities),
            'sources': list(surface.sources),
            'time_start': surface.time_start.isoformat() if surface.time_start else None,
            'time_end': surface.time_end.isoformat() if surface.time_end else None,
            'params_hash': surface.params_hash,
            'kernel_version': surface.kernel_version,
        })

        # CONTAINS edges to claims
        if surface.claim_ids:
            for claim_id in surface.claim_ids:
                await self.neo4j._execute_write("""
                    MATCH (s:Surface {kernel_signature: $signature})
                    MATCH (c:Claim {id: $claim_id})
                    MERGE (s)-[:CONTAINS]->(c)
                """, {
                    'signature': surface.key.signature,
                    'claim_id': claim_id,
                })

    async def _upsert_incident(self, incident: IncidentState) -> None:
        """Upsert Incident node using kernel signature.

        MERGE key: kernel_signature (deterministic from anchors + time)
        """
        await self.neo4j._execute_write("""
            MERGE (i:Incident {kernel_signature: $signature})
            ON CREATE SET
                i.id = $id,
                i.scope_id = $scope_id,
                i.created_at = datetime()
            SET i.anchor_entities = $anchors,
                i.companion_entities = $companions,
                i.time_start = $time_start,
                i.time_end = $time_end,
                i.params_hash = $params_hash,
                i.kernel_version = $kernel_version,
                i.updated_at = datetime()
        """, {
            'signature': incident.signature,
            'id': incident.id,
            'scope_id': incident.scope_id,
            'anchors': list(incident.anchor_entities),
            'companions': list(incident.companion_entities),
            'time_start': incident.time_start.isoformat() if incident.time_start else None,
            'time_end': incident.time_end.isoformat() if incident.time_end else None,
            'params_hash': incident.params_hash,
            'kernel_version': incident.kernel_version,
        })

        # CONTAINS edges to surfaces
        if incident.surface_ids:
            for surface_sig in incident.surface_ids:
                await self.neo4j._execute_write("""
                    MATCH (i:Incident {kernel_signature: $inc_sig})
                    MATCH (s:Surface {kernel_signature: $surf_sig})
                    MERGE (i)-[:CONTAINS]->(s)
                """, {
                    'inc_sig': incident.signature,
                    'surf_sig': surface_sig,
                })

    async def _create_link(self, link: Link) -> None:
        """Create a link/edge in the graph.

        Links use from_id/to_id which can be:
        - Claim IDs (cl_xxx)
        - Surface signatures (sf_xxx)
        - Incident signatures (inc_xxx)
        """
        # Determine node types from ID prefixes
        from_type = self._infer_node_type(link.from_id)
        to_type = self._infer_node_type(link.to_id)

        # Build dynamic query based on types
        await self.neo4j._execute_write(f"""
            MATCH (from:{from_type} {{{self._match_key(from_type)}: $from_id}})
            MATCH (to:{to_type} {{{self._match_key(to_type)}: $to_id}})
            MERGE (from)-[r:{link.relation}]->(to)
            SET r.created_at = coalesce(r.created_at, datetime())
        """, {
            'from_id': link.from_id,
            'to_id': link.to_id,
        })

    async def _persist_trace(self, trace: DecisionTrace) -> None:
        """Persist a decision trace to Neo4j."""
        await self.neo4j._execute_write("""
            CREATE (t:DecisionTrace {
                id: $id,
                decision_type: $decision_type,
                outcome: $outcome,
                subject_id: $subject_id,
                target_id: $target_id,
                rules_fired: $rules_fired,
                params_hash: $params_hash,
                kernel_version: $kernel_version,
                timestamp: $timestamp
            })
        """, {
            'id': trace.id,
            'decision_type': trace.decision_type,
            'outcome': trace.outcome,
            'subject_id': trace.subject_id,
            'target_id': trace.target_id,
            'rules_fired': list(trace.rules_fired),
            'params_hash': trace.params_hash,
            'kernel_version': trace.kernel_version,
            'timestamp': trace.timestamp.isoformat(),
        })

    def _infer_node_type(self, node_id: str) -> str:
        """Infer Neo4j node label from ID prefix."""
        if node_id.startswith("cl_"):
            return "Claim"
        elif node_id.startswith("sf_"):
            return "Surface"
        elif node_id.startswith("inc_") or node_id.startswith("in_"):
            return "Incident"
        else:
            return "Node"  # Fallback

    def _match_key(self, node_type: str) -> str:
        """Get the match key property for a node type."""
        if node_type == "Claim":
            return "id"
        elif node_type in ("Surface", "Incident"):
            return "kernel_signature"
        else:
            return "id"

    def _extract_scope_from_signature(self, signature: str) -> str:
        """Extract scope_id from incident signature (best effort)."""
        # Incident signatures are hashes - we can't reverse them
        # Store scope_id separately in incident state if needed
        return ""


class ShadowDeltaApplier(DeltaApplier):
    """DeltaApplier that compares kernel output with legacy weaver.

    For Phase 3/4: Run kernel in shadow mode alongside legacy weaver,
    compare outputs, log differences, but don't persist kernel output.
    """

    def __init__(self, neo4j: Neo4jService, legacy_weaver=None):
        super().__init__(neo4j)
        self.legacy_weaver = legacy_weaver
        self.diff_log: List[Dict[str, Any]] = []

    async def apply_shadow(
        self,
        delta: TopologyDelta,
        legacy_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare kernel delta with legacy result.

        Args:
            delta: Kernel's TopologyDelta
            legacy_result: Legacy weaver's output

        Returns:
            Diff summary
        """
        diff = self._compute_diff(delta, legacy_result)
        self.diff_log.append(diff)

        if diff["has_differences"]:
            logger.warning(f"Kernel/legacy diff: {diff}")
        else:
            logger.debug("Kernel output matches legacy")

        return diff

    def _compute_diff(
        self,
        delta: TopologyDelta,
        legacy_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute diff between kernel and legacy outputs."""
        # Compare surface counts
        kernel_surfaces = len(delta.surface_upserts)
        legacy_surfaces = legacy_result.get("surface_count", 0)

        # Compare incident counts
        kernel_incidents = len(delta.incident_upserts)
        legacy_incidents = legacy_result.get("incident_count", 0)

        # Check if routing decisions match
        kernel_routing = [
            t.outcome for t in delta.decision_traces
            if t.decision_type == "incident_membership"
        ]
        legacy_routing = legacy_result.get("routing_outcomes", [])

        has_differences = (
            kernel_surfaces != legacy_surfaces
            or kernel_incidents != legacy_incidents
            or kernel_routing != legacy_routing
        )

        return {
            "has_differences": has_differences,
            "kernel_surfaces": kernel_surfaces,
            "legacy_surfaces": legacy_surfaces,
            "kernel_incidents": kernel_incidents,
            "legacy_incidents": legacy_incidents,
            "kernel_routing": kernel_routing,
            "legacy_routing": legacy_routing,
            "signals": [s.signal_type.value for s in delta.signals],
        }

    def get_diff_summary(self) -> Dict[str, Any]:
        """Get summary of all diffs."""
        total = len(self.diff_log)
        matches = sum(1 for d in self.diff_log if not d["has_differences"])

        return {
            "total_comparisons": total,
            "matches": matches,
            "differences": total - matches,
            "match_rate": matches / total if total > 0 else 0.0,
        }
