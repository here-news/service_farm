"""
Snapshot Recorder for Kernel Regression Testing
================================================

Records kernel inputs (incidents, surfaces, entities, constraints)
to JSON snapshot files for deterministic replay testing.

Usage:
    from reee.tests.replay import record_snapshot

    # Record from live kernel run
    record_snapshot(
        output_path="fixtures/replay_my_snapshot.json",
        incidents=incidents,
        surfaces=surfaces,
        hub_entities=hub_entities,
        meta={
            "description": "My test scenario",
            "expected_outcome": {...}
        }
    )
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reee.types import Event, Surface


@dataclass
class SnapshotMeta:
    """Metadata for a replay snapshot."""
    description: str
    frozen_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    vocabulary_version: str = "2025-01-05"
    postmortem_reference: Optional[str] = None
    expected_outcome: Optional[Dict[str, Any]] = None


@dataclass
class IncidentSnapshot:
    """Snapshot of an incident's inputs."""
    anchor_entities: List[str]
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    surface_ids: List[str] = field(default_factory=list)
    expected_membership: Optional[str] = None  # "core", "periphery", "reject"
    expected_reason: Optional[str] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    _note: Optional[str] = None


@dataclass
class SurfaceSnapshot:
    """Snapshot of a surface's inputs."""
    question_key: str
    claim_ids: Optional[List[str]] = None


class SnapshotRecorder:
    """Records kernel inputs to replay snapshot files."""

    def __init__(self):
        self.meta: Optional[SnapshotMeta] = None
        self.hub_entities: Set[str] = set()
        self.incidents: Dict[str, IncidentSnapshot] = {}
        self.surfaces: Dict[str, SurfaceSnapshot] = {}
        self.witness_ledger: Dict[str, Any] = {}
        self.unrelated_background: Dict[str, Any] = {}

    def set_meta(
        self,
        description: str,
        expected_outcome: Optional[Dict[str, Any]] = None,
        postmortem_reference: Optional[str] = None,
    ) -> None:
        """Set snapshot metadata."""
        self.meta = SnapshotMeta(
            description=description,
            expected_outcome=expected_outcome,
            postmortem_reference=postmortem_reference,
        )

    def add_hub_entity(self, entity: str) -> None:
        """Add a hub entity."""
        self.hub_entities.add(entity)

    def add_incident(
        self,
        incident_id: str,
        event: Event,
        expected_membership: Optional[str] = None,
        expected_reason: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        """Add an incident from an Event object."""
        time_start = None
        time_end = None
        if event.time_window:
            ts, te = event.time_window
            if ts:
                time_start = ts.isoformat() + "Z" if not ts.tzinfo else ts.isoformat()
            if te:
                time_end = te.isoformat() + "Z" if not te.tzinfo else te.isoformat()

        self.incidents[incident_id] = IncidentSnapshot(
            anchor_entities=sorted(event.anchor_entities),
            time_start=time_start,
            time_end=time_end,
            surface_ids=sorted(event.surface_ids),
            expected_membership=expected_membership,
            expected_reason=expected_reason,
            _note=note,
        )

    def add_incident_raw(
        self,
        incident_id: str,
        anchor_entities: List[str],
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        surface_ids: Optional[List[str]] = None,
        expected_membership: Optional[str] = None,
        expected_reason: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        """Add an incident from raw data."""
        self.incidents[incident_id] = IncidentSnapshot(
            anchor_entities=anchor_entities,
            time_start=time_start,
            time_end=time_end,
            surface_ids=surface_ids or [],
            expected_membership=expected_membership,
            expected_reason=expected_reason,
            _note=note,
        )

    def add_surface(
        self,
        surface_id: str,
        surface: Surface,
    ) -> None:
        """Add a surface from a Surface object."""
        self.surfaces[surface_id] = SurfaceSnapshot(
            question_key=surface.question_key or "unknown",
            claim_ids=sorted(surface.claim_ids) if surface.claim_ids else None,
        )

    def add_surface_raw(
        self,
        surface_id: str,
        question_key: str,
        claim_ids: Optional[List[str]] = None,
    ) -> None:
        """Add a surface from raw data."""
        self.surfaces[surface_id] = SurfaceSnapshot(
            question_key=question_key,
            claim_ids=claim_ids,
        )

    def set_witness_ledger(self, ledger: Dict[str, Any]) -> None:
        """Set the witness ledger (geo entities, event types, etc.)."""
        self.witness_ledger = ledger

    def set_unrelated_background(self, count: int, pattern: str) -> None:
        """Set info about unrelated background incidents for hub dilution."""
        self.unrelated_background = {
            "_note": "Background incidents to dilute hub threshold",
            "count": count,
            "pattern": pattern,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        result = {}

        # Meta
        if self.meta:
            result["_meta"] = asdict(self.meta)

        # Hub entities
        result["hub_entities"] = sorted(self.hub_entities)

        # Incidents
        result["incidents"] = {}
        for inc_id, inc in sorted(self.incidents.items()):
            inc_dict = asdict(inc)
            # Remove None values and empty lists
            inc_dict = {k: v for k, v in inc_dict.items() if v is not None and v != []}
            result["incidents"][inc_id] = inc_dict

        # Surfaces
        result["surfaces"] = {}
        for surf_id, surf in sorted(self.surfaces.items()):
            surf_dict = asdict(surf)
            surf_dict = {k: v for k, v in surf_dict.items() if v is not None}
            result["surfaces"][surf_id] = surf_dict

        # Witness ledger
        if self.witness_ledger:
            result["witness_ledger"] = self.witness_ledger

        # Unrelated background
        if self.unrelated_background:
            result["unrelated_background"] = self.unrelated_background

        return result

    def save(self, path: Path) -> None:
        """Save snapshot to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_kernel_inputs(
        cls,
        incidents: Dict[str, Event],
        surfaces: Dict[str, Surface],
        hub_entities: Set[str],
        description: str,
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> "SnapshotRecorder":
        """Create recorder from kernel inputs."""
        recorder = cls()
        recorder.set_meta(description, expected_outcome)

        for entity in hub_entities:
            recorder.add_hub_entity(entity)

        for inc_id, event in incidents.items():
            recorder.add_incident(inc_id, event)

        for surf_id, surface in surfaces.items():
            recorder.add_surface(surf_id, surface)

        return recorder


def record_snapshot(
    output_path: str,
    incidents: Dict[str, Event],
    surfaces: Dict[str, Surface],
    hub_entities: Optional[Set[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Convenience function to record a snapshot file.

    Args:
        output_path: Path to save snapshot file
        incidents: Dict of Event objects
        surfaces: Dict of Surface objects
        hub_entities: Set of hub entity names
        meta: Dict with keys: description, expected_outcome, postmortem_reference

    Returns:
        Path to saved snapshot file
    """
    path = Path(output_path)
    recorder = SnapshotRecorder.from_kernel_inputs(
        incidents=incidents,
        surfaces=surfaces,
        hub_entities=hub_entities or set(),
        description=meta.get("description", "Unnamed snapshot") if meta else "Unnamed snapshot",
        expected_outcome=meta.get("expected_outcome") if meta else None,
    )

    if meta and meta.get("postmortem_reference"):
        recorder.meta.postmortem_reference = meta["postmortem_reference"]

    recorder.save(path)
    return path
