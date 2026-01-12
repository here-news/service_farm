"""
Incremental Weaver: Top-Down Routing with Bottom-Up Growth

Key insight: Route new claims to the HIGHEST existing structure first,
then drill down. Structure grows bottom-up as claims arrive.

Routing order (for each new claim):
1. CASE level: Does claim's anchor set intersect any case's membrane?
   - If yes: route to that case, continue to step 2 within that case
   - If no: this claim may seed a new case

2. INCIDENT level: Within candidate case(s), does claim match an incident?
   - Match = shared referent (not just context) + time compatibility
   - If yes: attach to that incident
   - If no: create new incident (may merge cases later via spine edge)

3. SURFACE level: Within the incident, does claim match a surface?
   - Match = same proposition key (what question is being answered)
   - If yes: update the surface (may detect conflict)
   - If no: create new surface

This is O(cases + incidents_in_case + surfaces_in_incident) per claim,
not O(all_claims) like pairwise embedding comparison.

Indices maintained:
- case_by_anchor: entity → set of case_ids containing that entity
- incident_by_referent: entity → set of incident_ids with that referent
- surface_by_proposition: (scope, question_key) → surface_id
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, FrozenSet, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum, auto
import hashlib
import uuid


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WeaverConfig:
    """Tunable parameters for the weaver."""
    # Time window for same-happening (hours)
    same_happening_window_hours: float = 72.0

    # Minimum referent overlap for incident merge
    min_referent_overlap: int = 1

    # Hub entities that don't count for case membrane
    hub_entities: FrozenSet[str] = frozenset({
        'Hong Kong', 'China', 'United States', 'US', 'UK',
        'United Kingdom', 'New York', 'Washington'
    })

    # Confidence threshold for creating structure
    min_confidence: float = 0.5


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class ClaimArtifact:
    """Extracted 'genes' from a claim - immutable, cacheable."""
    claim_id: str
    text: str
    referents: FrozenSet[str]      # Specific entities (identity witnesses)
    contexts: FrozenSet[str]       # Broad entities (never merge alone)
    anchor_entities: FrozenSet[str]  # Combined for membrane check
    event_time: Optional[datetime]
    proposition_key: str           # What variable this asserts
    source: str
    confidence: float
    embedding: Optional[List[float]] = None


@dataclass
class Surface:
    """L2: A proposition about a scoped topic."""
    surface_id: str
    scope_id: str                  # Hash of anchor entities
    proposition_key: str
    claim_ids: Set[str]
    values: List[Any]              # Extracted values (for conflict detection)
    entropy: float = 0.0           # Uncertainty measure
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_claim(self, artifact: ClaimArtifact, value: Any = None):
        self.claim_ids.add(artifact.claim_id)
        if value is not None:
            self.values.append(value)
            # Simple entropy: unique values / total values
            if len(self.values) > 1:
                unique = len(set(str(v) for v in self.values))
                self.entropy = unique / len(self.values)


@dataclass
class Incident:
    """L3: A specific happening in the world."""
    incident_id: str
    referents: Set[str]            # Identity-defining entities (mutable - grows)
    contexts: Set[str]             # Ambient entities
    surface_ids: Set[str]
    time_start: Optional[datetime]
    time_end: Optional[datetime]
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def anchor_entities(self) -> Set[str]:
        return self.referents | self.contexts

    def add_surface(self, surface_id: str, artifact: ClaimArtifact):
        self.surface_ids.add(surface_id)
        # Expand referents/contexts
        self.referents.update(artifact.referents)
        self.contexts.update(artifact.contexts)
        # Expand time window
        if artifact.event_time:
            if self.time_start is None:
                self.time_start = artifact.event_time
                self.time_end = artifact.event_time
            else:
                if artifact.event_time < self.time_start:
                    self.time_start = artifact.event_time
                if artifact.event_time > self.time_end:
                    self.time_end = artifact.event_time


@dataclass
class Case:
    """L4: A story/organism - spine-connected incidents."""
    case_id: str
    incident_ids: Set[str]
    membrane: Set[str]             # All anchor entities in this case
    spine_edges: Set[Tuple[str, str]]  # Pairs defining membership
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_incident(self, incident: Incident):
        self.incident_ids.add(incident.incident_id)
        self.membrane.update(incident.anchor_entities)


# =============================================================================
# Routing Decisions
# =============================================================================

class RouteLevel(Enum):
    CASE = auto()
    INCIDENT = auto()
    SURFACE = auto()
    NEW_STRUCTURE = auto()


@dataclass
class RouteDecision:
    """Result of routing a claim."""
    level: RouteLevel
    case_id: Optional[str]
    incident_id: Optional[str]
    surface_id: Optional[str]
    action: str                    # What happened
    created_new: bool              # Did we create new structure?
    signals: List[str] = field(default_factory=list)


# =============================================================================
# Incremental Weaver
# =============================================================================

class IncrementalWeaver:
    """
    Builds topology incrementally with top-down routing.

    Key principle: Route to highest existing structure, create only when needed.
    """

    def __init__(self, config: WeaverConfig = None):
        self.config = config or WeaverConfig()

        # Core state
        self.surfaces: Dict[str, Surface] = {}
        self.incidents: Dict[str, Incident] = {}
        self.cases: Dict[str, Case] = {}

        # Routing indices (the key to O(1) lookups)
        self._case_by_anchor: Dict[str, Set[str]] = {}      # entity → case_ids
        self._incident_by_referent: Dict[str, Set[str]] = {} # entity → incident_ids
        self._surface_by_key: Dict[Tuple[str, str], str] = {} # (scope, prop) → surface_id

        # Audit trail
        self.decisions: List[RouteDecision] = []
        self.claims_processed: int = 0

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def process_claim(self, artifact: ClaimArtifact) -> RouteDecision:
        """
        Process a single claim through the routing hierarchy.

        Returns: RouteDecision with full trace
        """
        self.claims_processed += 1

        # Step 1: Find candidate cases by membrane intersection
        candidate_cases = self._find_candidate_cases(artifact)

        # Step 2: Within candidates, find matching incident
        matched_incident = None
        matched_case = None

        for case_id in candidate_cases:
            case = self.cases[case_id]
            incident = self._find_matching_incident(artifact, case)
            if incident:
                matched_incident = incident
                matched_case = case
                break

        # Step 3: Route based on what we found
        if matched_incident:
            # Attach to existing incident
            return self._attach_to_incident(artifact, matched_incident, matched_case)
        elif candidate_cases:
            # Case exists but no matching incident - create new incident in case
            case = self.cases[list(candidate_cases)[0]]
            return self._create_incident_in_case(artifact, case)
        else:
            # No matching case - create entirely new structure
            return self._create_new_structure(artifact)

    # -------------------------------------------------------------------------
    # Case-Level Routing
    # -------------------------------------------------------------------------

    def _find_candidate_cases(self, artifact: ClaimArtifact) -> Set[str]:
        """
        Find cases whose membrane intersects claim's anchors.

        O(|anchors|) lookups, not O(|cases|).
        """
        candidates = set()

        # Check each anchor entity
        for entity in artifact.anchor_entities:
            # Skip hub entities
            if entity in self.config.hub_entities:
                continue

            # Find cases containing this entity
            if entity in self._case_by_anchor:
                candidates.update(self._case_by_anchor[entity])

        return candidates

    # -------------------------------------------------------------------------
    # Incident-Level Routing
    # -------------------------------------------------------------------------

    def _find_matching_incident(
        self,
        artifact: ClaimArtifact,
        case: Case
    ) -> Optional[Incident]:
        """
        Find an incident in the case that matches this claim.

        Match criteria:
        - Shared referent (not just context)
        - Time compatibility (within window)
        """
        # Get candidate incidents by referent
        candidate_ids = set()
        for ref in artifact.referents:
            if ref in self._incident_by_referent:
                # Only incidents in this case
                candidate_ids.update(
                    self._incident_by_referent[ref] & case.incident_ids
                )

        # Check each candidate for full match
        for iid in candidate_ids:
            incident = self.incidents[iid]

            # Check referent overlap
            overlap = artifact.referents & incident.referents
            if len(overlap) < self.config.min_referent_overlap:
                continue

            # Check time compatibility
            if artifact.event_time and incident.time_start:
                time_diff = abs(
                    (artifact.event_time - incident.time_start).total_seconds() / 3600
                )
                if time_diff > self.config.same_happening_window_hours:
                    continue  # Too far apart in time

            # Found a match!
            return incident

        return None

    # -------------------------------------------------------------------------
    # Surface-Level Routing
    # -------------------------------------------------------------------------

    def _find_or_create_surface(
        self,
        artifact: ClaimArtifact,
        incident: Incident
    ) -> Tuple[Surface, bool]:
        """
        Find existing surface or create new one.

        Returns: (surface, created_new)
        """
        scope_id = self._compute_scope_id(artifact.anchor_entities)
        key = (scope_id, artifact.proposition_key)

        if key in self._surface_by_key:
            surface = self.surfaces[self._surface_by_key[key]]
            surface.add_claim(artifact)
            return surface, False

        # Create new surface
        surface = Surface(
            surface_id=f"surf_{uuid.uuid4().hex[:8]}",
            scope_id=scope_id,
            proposition_key=artifact.proposition_key,
            claim_ids={artifact.claim_id},
            values=[],
        )

        self.surfaces[surface.surface_id] = surface
        self._surface_by_key[key] = surface.surface_id

        return surface, True

    # -------------------------------------------------------------------------
    # Structure Creation
    # -------------------------------------------------------------------------

    def _attach_to_incident(
        self,
        artifact: ClaimArtifact,
        incident: Incident,
        case: Case
    ) -> RouteDecision:
        """Attach claim to existing incident."""
        surface, surface_created = self._find_or_create_surface(artifact, incident)
        incident.add_surface(surface.surface_id, artifact)

        # Update case membrane
        case.membrane.update(artifact.anchor_entities)
        self._update_case_index(case)

        signals = []
        if surface.entropy > 0.5:
            signals.append("HIGH_ENTROPY")

        decision = RouteDecision(
            level=RouteLevel.INCIDENT,
            case_id=case.case_id,
            incident_id=incident.incident_id,
            surface_id=surface.surface_id,
            action=f"attached to incident {incident.incident_id[:8]}",
            created_new=surface_created,
            signals=signals,
        )
        self.decisions.append(decision)
        return decision

    def _create_incident_in_case(
        self,
        artifact: ClaimArtifact,
        case: Case
    ) -> RouteDecision:
        """Create new incident within existing case."""
        # Create surface
        surface, _ = self._find_or_create_surface(artifact, None)

        # Create incident
        incident = Incident(
            incident_id=f"inc_{uuid.uuid4().hex[:8]}",
            referents=set(artifact.referents),
            contexts=set(artifact.contexts),
            surface_ids={surface.surface_id},
            time_start=artifact.event_time,
            time_end=artifact.event_time,
        )

        self.incidents[incident.incident_id] = incident
        self._update_incident_index(incident)

        # Add to case
        case.add_incident(incident)
        self._update_case_index(case)

        decision = RouteDecision(
            level=RouteLevel.CASE,
            case_id=case.case_id,
            incident_id=incident.incident_id,
            surface_id=surface.surface_id,
            action=f"new incident in case {case.case_id[:8]}",
            created_new=True,
        )
        self.decisions.append(decision)
        return decision

    def _create_new_structure(self, artifact: ClaimArtifact) -> RouteDecision:
        """Create entirely new case/incident/surface."""
        # Create surface
        scope_id = self._compute_scope_id(artifact.anchor_entities)
        surface = Surface(
            surface_id=f"surf_{uuid.uuid4().hex[:8]}",
            scope_id=scope_id,
            proposition_key=artifact.proposition_key,
            claim_ids={artifact.claim_id},
            values=[],
        )
        self.surfaces[surface.surface_id] = surface
        self._surface_by_key[(scope_id, artifact.proposition_key)] = surface.surface_id

        # Create incident
        incident = Incident(
            incident_id=f"inc_{uuid.uuid4().hex[:8]}",
            referents=set(artifact.referents),
            contexts=set(artifact.contexts),
            surface_ids={surface.surface_id},
            time_start=artifact.event_time,
            time_end=artifact.event_time,
        )
        self.incidents[incident.incident_id] = incident
        self._update_incident_index(incident)

        # Create case
        case = Case(
            case_id=f"case_{uuid.uuid4().hex[:8]}",
            incident_ids={incident.incident_id},
            membrane=set(incident.anchor_entities),
            spine_edges=set(),
        )
        self.cases[case.case_id] = case
        self._update_case_index(case)

        decision = RouteDecision(
            level=RouteLevel.NEW_STRUCTURE,
            case_id=case.case_id,
            incident_id=incident.incident_id,
            surface_id=surface.surface_id,
            action="created new case/incident/surface",
            created_new=True,
        )
        self.decisions.append(decision)
        return decision

    # -------------------------------------------------------------------------
    # Index Management
    # -------------------------------------------------------------------------

    def _compute_scope_id(self, entities: FrozenSet[str]) -> str:
        """Hash of sorted entities for scope identity."""
        key = "|".join(sorted(entities))
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _update_case_index(self, case: Case):
        """Update case-by-anchor index."""
        for entity in case.membrane:
            if entity not in self.config.hub_entities:
                self._case_by_anchor.setdefault(entity, set()).add(case.case_id)

    def _update_incident_index(self, incident: Incident):
        """Update incident-by-referent index."""
        for ref in incident.referents:
            self._incident_by_referent.setdefault(ref, set()).add(incident.incident_id)

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "claims_processed": self.claims_processed,
            "surfaces": len(self.surfaces),
            "incidents": len(self.incidents),
            "cases": len(self.cases),
            "multi_incident_cases": sum(1 for c in self.cases.values() if len(c.incident_ids) > 1),
            "largest_case": max((len(c.incident_ids) for c in self.cases.values()), default=0),
            "decisions_by_level": {
                level.name: sum(1 for d in self.decisions if d.level == level)
                for level in RouteLevel
            },
        }

    def get_case_for_entity(self, entity: str) -> Optional[Case]:
        """Find case containing an entity."""
        if entity in self._case_by_anchor:
            case_ids = self._case_by_anchor[entity]
            if case_ids:
                return self.cases[next(iter(case_ids))]
        return None


# =============================================================================
# Experiment Runner
# =============================================================================

def run_incremental_experiment(claims: List[Dict]) -> Dict[str, Any]:
    """
    Run incremental weaver on a list of claims.

    Each claim dict should have:
    - id, text, entities, anchor_entities, event_time, source
    """
    weaver = IncrementalWeaver()

    for claim in claims:
        # Convert to artifact
        artifact = ClaimArtifact(
            claim_id=claim["id"],
            text=claim.get("text", ""),
            referents=frozenset(claim.get("anchor_entities", [])),
            contexts=frozenset(claim.get("entities", set()) - set(claim.get("anchor_entities", []))),
            anchor_entities=frozenset(claim.get("anchor_entities", [])),
            event_time=claim.get("event_time"),
            proposition_key=claim.get("proposition_key", f"claim_{claim['id'][:8]}"),
            source=claim.get("source", "unknown"),
            confidence=claim.get("confidence", 0.8),
            embedding=claim.get("embedding"),
        )

        weaver.process_claim(artifact)

    return {
        "summary": weaver.summary(),
        "cases": {
            c.case_id: {
                "incidents": len(c.incident_ids),
                "membrane": sorted(list(c.membrane))[:10],
            }
            for c in weaver.cases.values()
        },
        "decisions": [
            {
                "level": d.level.name,
                "action": d.action,
                "created_new": d.created_new,
            }
            for d in weaver.decisions[-20:]  # Last 20
        ],
    }


# =============================================================================
# Test with Synthetic Data
# =============================================================================

if __name__ == "__main__":
    from datetime import timedelta

    base_time = datetime(2024, 11, 26, 10, 0, 0)

    # Simulate WFC fire claims arriving in order
    test_claims = [
        # First WFC claim - creates new structure
        {
            "id": "c1",
            "text": "Fire breaks out at Wang Fuk Court in Tai Po",
            "entities": {"Wang Fuk Court", "Tai Po", "Hong Kong"},
            "anchor_entities": {"Wang Fuk Court", "Tai Po"},
            "event_time": base_time,
            "source": "reuters",
            "proposition_key": "wfc:fire_reported",
        },
        # Second WFC claim - should attach to existing case/incident
        {
            "id": "c2",
            "text": "Two confirmed dead in Wang Fuk Court fire",
            "entities": {"Wang Fuk Court", "Tai Po"},
            "anchor_entities": {"Wang Fuk Court"},
            "event_time": base_time + timedelta(hours=2),
            "source": "bbc",
            "proposition_key": "wfc:death_count",
        },
        # Third WFC claim - same incident, different surface
        {
            "id": "c3",
            "text": "Firefighters battle blaze at Wang Fuk Court",
            "entities": {"Wang Fuk Court", "Hong Kong Fire Services"},
            "anchor_entities": {"Wang Fuk Court"},
            "event_time": base_time + timedelta(hours=1),
            "source": "scmp",
            "proposition_key": "wfc:response",
        },
        # Unrelated claim - should create new case
        {
            "id": "c4",
            "text": "Trump announces new tariffs on China",
            "entities": {"Donald Trump", "China", "United States"},
            "anchor_entities": {"Donald Trump"},
            "event_time": base_time + timedelta(hours=1),
            "source": "cnn",
            "proposition_key": "trump:tariff",
        },
        # Another WFC update - death toll rises
        {
            "id": "c5",
            "text": "Death toll rises to 17 in Wang Fuk Court fire",
            "entities": {"Wang Fuk Court", "Tai Po"},
            "anchor_entities": {"Wang Fuk Court"},
            "event_time": base_time + timedelta(hours=8),
            "source": "nyt",
            "proposition_key": "wfc:death_count",  # Same proposition - should update surface
        },
    ]

    result = run_incremental_experiment(test_claims)

    print("=" * 60)
    print("INCREMENTAL WEAVER EXPERIMENT")
    print("=" * 60)
    print()
    print("Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print()
    print("Cases formed:")
    for case_id, info in result["cases"].items():
        print(f"  {case_id}: {info['incidents']} incidents")
        print(f"    membrane: {info['membrane']}")
    print()
    print("Routing decisions:")
    for d in result["decisions"]:
        print(f"  [{d['level']}] {d['action']} (new={d['created_new']})")
