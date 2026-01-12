"""
TopologyKernel - Pure orchestrator for evidence processing.

This is the main entry point for the kernel. It:
1. Computes surface key
2. Updates/creates surface
3. Routes surface to incident
4. Returns TopologyDelta with all changes

Pure function - no DB, no LLM.
All candidates provided via PartitionSnapshot.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import uuid

from ..contracts.evidence import ClaimEvidence
from ..contracts.state import (
    SurfaceKey,
    SurfaceState,
    IncidentState,
    PartitionSnapshot,
    compute_incident_signature,
)
from ..contracts.delta import TopologyDelta, Link
from ..contracts.traces import DecisionTrace
from ..contracts.signals import EpistemicSignal, SignalType, Severity, generate_signal_id

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
)


@dataclass(frozen=True)
class KernelParams:
    """Combined parameters for kernel processing."""
    surface_key_params: SurfaceKeyParams = SurfaceKeyParams()
    routing_params: RoutingParams = RoutingParams()

    @property
    def kernel_version(self) -> str:
        return self.routing_params.kernel_version


def generate_incident_id() -> str:
    """Generate a new incident ID."""
    return f"inc_{uuid.uuid4().hex[:12]}"


class TopologyKernel:
    """Pure kernel for topology computation.

    Stateless - all state comes from PartitionSnapshot.
    Returns TopologyDelta with changes.

    Usage:
        kernel = TopologyKernel(params)
        delta = kernel.process_evidence(snapshot, evidence)
    """

    def __init__(self, params: KernelParams = KernelParams()):
        self.params = params

    def process_evidence(
        self,
        snapshot: PartitionSnapshot,
        evidence: ClaimEvidence,
    ) -> TopologyDelta:
        """Process a single piece of evidence.

        Pure function:
        1. Compute surface key
        2. Find/create surface
        3. Route to incident
        4. Return delta

        Args:
            snapshot: Current partition state (surfaces + incidents)
            evidence: Claim evidence to process

        Returns:
            TopologyDelta with all changes
        """
        traces: List[DecisionTrace] = []
        signals: List[EpistemicSignal] = []
        surface_upserts: List[SurfaceState] = []
        incident_upserts: List[IncidentState] = []
        links: List[Link] = []

        # Step 1: Compute surface key
        key_result = compute_surface_key(
            evidence=evidence,
            params=self.params.surface_key_params,
        )
        traces.append(key_result.trace)

        # Emit signal for low-confidence key
        if key_result.question_key_result.confidence < 0.5:
            signals.append(EpistemicSignal(
                id=generate_signal_id(),
                signal_type=SignalType.EXTRACTION_SPARSE,
                subject_id=evidence.claim_id,
                subject_type="claim",
                severity=Severity.INFO,
                evidence={
                    "question_key": key_result.question_key,
                    "fallback_level": key_result.question_key_result.fallback_level.name,
                    "confidence": key_result.question_key_result.confidence,
                },
                resolution_hint="Consider LLM extraction for question_key",
                timestamp=datetime.utcnow(),
            ))

        # Step 2: Find/update surface
        existing_surface = self._find_surface(snapshot, key_result.key)
        updated_surface, is_new_surface = apply_claim_to_surface(
            surface=existing_surface,
            evidence=evidence,
            key=key_result.key,
        )
        surface_upserts.append(updated_surface)

        # Link: claim → surface
        links.append(Link(
            from_id=evidence.claim_id,
            relation="MEMBER_OF",
            to_id=key_result.key.signature,
        ))

        # Step 3: Route surface to incident
        candidates = find_candidates(
            surface=updated_surface,
            snapshot=snapshot,
            params=self.params.routing_params,
        )

        route_result = decide_route(
            surface=updated_surface,
            candidates=candidates,
            params=self.params.routing_params,
            snapshot=snapshot,
        )
        traces.append(route_result.trace)
        signals.extend(route_result.signals)

        # Step 4: Create/update incident
        surface_companions = updated_surface.entities - updated_surface.anchor_entities
        if route_result.outcome == RouteOutcome.JOINED_BY_SIGNATURE:
            # Exact signature match - update existing incident
            existing_incident = self._find_incident_by_signature(
                snapshot, route_result.incident_signature
            )
            if existing_incident:
                updated_incident = existing_incident.with_surface(
                    surface_id=key_result.key.signature,
                    anchors=updated_surface.anchor_entities,
                    companions=surface_companions,
                    surface_time=updated_surface.time_start,
                )
                incident_upserts.append(updated_incident)

                # Link: surface → incident
                links.append(Link(
                    from_id=key_result.key.signature,
                    relation="PART_OF",
                    to_id=existing_incident.signature,
                ))
        elif route_result.outcome == RouteOutcome.CREATED_NEW:
            # Create new incident
            new_incident = IncidentState(
                id=generate_incident_id(),
                signature=route_result.incident_signature,
                scope_id=snapshot.scope_id,  # Set scope_id for DB lookup
                surface_ids=frozenset({key_result.key.signature}),
                anchor_entities=updated_surface.anchor_entities,
                companion_entities=surface_companions,
                time_start=updated_surface.time_start,
                time_end=updated_surface.time_end,
            )
            incident_upserts.append(new_incident)

            # Link: surface → incident
            links.append(Link(
                from_id=key_result.key.signature,
                relation="PART_OF",
                to_id=new_incident.signature,
            ))
        else:
            # Update existing incident
            existing_incident = self._find_incident_by_signature(
                snapshot, route_result.incident_signature
            )
            if existing_incident:
                updated_incident = existing_incident.with_surface(
                    surface_id=key_result.key.signature,
                    anchors=updated_surface.anchor_entities,
                    companions=surface_companions,
                    surface_time=updated_surface.time_start,
                )
                incident_upserts.append(updated_incident)

                # Link: surface → incident
                links.append(Link(
                    from_id=key_result.key.signature,
                    relation="PART_OF",
                    to_id=existing_incident.signature,
                ))

        return TopologyDelta(
            surface_upserts=surface_upserts,
            incident_upserts=incident_upserts,
            links=links,
            decision_traces=traces,
            signals=signals,
            inquiries=[],  # Populated by signal_to_inquiry outside kernel
        )

    def process_batch(
        self,
        snapshot: PartitionSnapshot,
        evidence_list: List[ClaimEvidence],
    ) -> TopologyDelta:
        """Process multiple pieces of evidence.

        Processes sequentially, updating snapshot as we go.
        Returns merged delta.

        Args:
            snapshot: Initial partition state
            evidence_list: List of evidence to process

        Returns:
            Merged TopologyDelta with all changes
        """
        merged_delta = TopologyDelta()

        # Build mutable lookup for intermediate state
        surfaces_by_sig: Dict[str, SurfaceState] = {
            s.key.signature: s for s in snapshot.surfaces
        }
        incidents_by_sig: Dict[str, IncidentState] = {
            i.signature: i for i in snapshot.incidents
        }

        for evidence in evidence_list:
            # Create updated snapshot with intermediate state
            current_snapshot = PartitionSnapshot(
                scope_id=snapshot.scope_id,
                surfaces=list(surfaces_by_sig.values()),
                incidents=list(incidents_by_sig.values()),
            )

            delta = self.process_evidence(current_snapshot, evidence)

            # Update intermediate state
            for surface in delta.surface_upserts:
                surfaces_by_sig[surface.key.signature] = surface

            for incident in delta.incident_upserts:
                incidents_by_sig[incident.signature] = incident

            # Merge into result
            merged_delta = merged_delta.merge(delta)

        return merged_delta

    def _find_surface(
        self,
        snapshot: PartitionSnapshot,
        key: SurfaceKey,
    ) -> Optional[SurfaceState]:
        """Find surface by key in snapshot."""
        for surface in snapshot.surfaces:
            if surface.key.signature == key.signature:
                return surface
        return None

    def _find_incident_by_signature(
        self,
        snapshot: PartitionSnapshot,
        signature: str,
    ) -> Optional[IncidentState]:
        """Find incident by signature in snapshot."""
        for incident in snapshot.incidents:
            if incident.signature == signature:
                return incident
        return None
