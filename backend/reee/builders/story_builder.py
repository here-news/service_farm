"""
Story Builder: Spine + Mode + Membrane Approach
================================================

This replaces the k=2 motif recurrence approach with a spine-based model
for L4 story formation. Designed for star-shaped news patterns where:
- One focal entity (spine) appears across many incidents
- Companion entities rotate (no pair recurs)
- The story coheres via shared spine + temporal mode

THEORY (from principled discussion):
- k=2 motif recurrence is a clique detector applied to star-shaped data
- News stories often follow: Wang Fuk Court + Fire Services, Wang Fuk Court + John Lee, etc.
- Zero recurring pairs, but obviously one story

NEW APPROACH:
- Story = Spine + Episodes + Facets
- Spine: focal entity (non-hub) that defines the story
- Episodes: L3 incidents attached to the spine
- Facets: L2 surfaces representing what we know (death_count, injuries, cause, etc.)
- Temporal Mode: separates same-spine stories in different time periods

CORE vs PERIPHERY ATTACHMENT:
- Core: incident contains spine anchor + same temporal mode + context compatible
- Periphery: semantic similarity to story gist, or relation backbone

JAYNES INTEGRATION:
- Jaynes is NOT for forming story boundaries
- Jaynes is for inside-story truth work on facets:
  - Compute typed posterior for each facet
  - Emit inquiries: conflicts, gaps, quality issues

FACET CONTRACT (for story completeness):
=========================================
The facet system tracks what we know about a story across four orthogonal dimensions:

1. FACET PRESENCE (what we have):
   - Which question_keys have at least one surface in this story
   - Computed: set(surface.question_key for surface in story_surfaces)
   - Example: {"fire_death_count", "fire_cause", "fire_response"}

2. FACET COVERAGE (present ∩ schema):
   - Intersection of what we have with what we expect
   - Computed: present_facets & expected_facets
   - completeness_score = |coverage| / |expected_facets|
   - Example: If schema expects 5 fire facets and we have 3, coverage=3, score=60%

3. FACET NOISE (present - expected):
   - Facets we have but don't expect for this story type
   - Computed: present_facets - expected_facets
   - Not necessarily bad (unexpected discoveries), but tracked for audit
   - Example: "victim_social_media" facet on a fire story (unexpected)

4. BLOCKED FACETS (expected but structurally impossible):
   - Expected facets that cannot exist due to structural reasons:
     a) typed_coverage_zero: No sources have this typed claim
     b) missing_time: Facet requires temporal data we don't have
     c) extraction_gap: Extractor doesn't support this question type yet
     d) source_quality: All sources for this facet fail quality threshold
   - These are NOT gaps (gaps = could exist but don't yet)
   - Blocked facets should emit quality_inquiries with reason

FACET INQUIRY GENERATION:
- conflict_inquiries: Facets with entropy > threshold (conflicting values)
- gap_inquiries: expected_facets - present_facets (missing data)
- quality_inquiries: Facets with blocked reasons or low source quality

The schema drives what we expect, not what we observe. This prevents
tautological completeness (100% because we only expect what we have).

Usage:
    builder = StoryBuilder()
    result = builder.build_from_incidents(incidents, surfaces)
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any

from ..types import (
    Event, Surface, Story, StoryScale,
    Constraint, ConstraintType, ConstraintLedger,
    MembershipLevel, EventJustification,
)
from ..membrane import (
    Membership, CoreReason, LinkType,
    FocalSet, MembershipDecision,
    classify_incident_membership,
    compute_core_leak_rate,
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class TemporalMode:
    """
    A temporal mode represents a distinct time period for a spine's story.

    Same spine can have multiple stories in different temporal modes:
    - Wang Fuk Court Fire 2025 (Nov-Dec 2025)
    - Wang Fuk Court other incident (if one happened in 2020)

    Mode detection uses gap-based clustering:
    - If incidents are within mode_gap_days, same mode
    - If gap > mode_gap_days, new mode
    """
    mode_id: str
    spine: str
    time_start: datetime
    time_end: datetime
    incident_ids: Set[str]
    explanation: str


@dataclass
class StoryFacet:
    """
    A facet represents a question/aspect of the story.

    Facets are L2 surfaces grouped by question_key:
    - fire_death_count → multiple claims from sources
    - fire_injury_count → multiple claims
    - fire_cause → categorical/text claims

    Each facet can have a TypedBeliefState for Jaynes inference.
    """
    question_key: str
    surface_ids: Set[str]
    claim_count: int
    source_count: int
    typed_domain: Optional[str] = None  # "count", "categorical", "text"
    belief_state: Optional[Any] = None  # TypedBeliefState when computed
    entropy: Optional[float] = None
    map_value: Optional[Any] = None
    has_conflict: bool = False
    has_gap: bool = False  # True if expected facet is missing data


@dataclass
class StorySpine:
    """
    A story spine is a non-hub entity that defines a story.

    The spine collects:
    - Episodes (incidents mentioning the spine)
    - Facets (surfaces grouped by question_key)
    - Temporal modes (to separate same-spine stories)
    """
    entity: str
    spine_id: str
    temporal_modes: List[TemporalMode]
    total_incidents: int
    incident_fraction: float  # What fraction of all incidents
    is_hub: bool
    explanation: str


@dataclass
class BlockedCoreB:
    """Tracks an incident that was considered for Core-B but blocked."""
    incident_id: str
    reason: str  # Why it was blocked (e.g., "only 1 witness, need ≥2")
    witnesses_found: List[str]  # Constraint IDs that were found
    witnesses_missing: str  # What was missing


@dataclass
class CompleteStory:
    """
    A complete story = spine + mode + episodes + facets.

    This replaces CaseCore and EntityCase with a unified model.
    """
    # Required fields (no defaults) - must come first
    story_id: str
    spine: str  # Focal entity
    mode: TemporalMode  # Temporal mode
    core_incident_ids: Set[str]  # Core-A (anchor) + Core-B (warrant)
    facets: Dict[str, StoryFacet]  # question_key → facet
    title: str
    description: str
    time_start: Optional[datetime]
    time_end: Optional[datetime]
    surface_count: int
    source_count: int
    claim_count: int
    # FACET CONTRACT fields (see module docstring for details)
    expected_facets: Set[str]  # Schema-driven: what facets we expect for this story type
    present_facets: Set[str]   # Facet presence: what facets we actually have
    coverage_facets: Set[str]  # Facet coverage: present ∩ expected (what we have that matters)
    missing_facets: Set[str]   # Gap facets: expected - present (what we're missing)
    noise_facets: Set[str]     # Facet noise: present - expected (unexpected facets)
    blocked_facets: Dict[str, str]  # Facet -> reason it's structurally blocked
    completeness_score: float  # |coverage| / |expected| (not present/expected!)
    conflict_inquiries: List[Dict]  # Facets with high entropy / conflicting values
    gap_inquiries: List[Dict]  # Expected facets with no data
    quality_inquiries: List[Dict]  # Facets with quality issues
    membership_weights: Dict[str, MembershipLevel]  # incident_id → level
    explanation: str

    # Optional fields with defaults - must come after required fields
    focal_set: FocalSet = None  # Membrane focal set (single spine by default)
    core_a_ids: Set[str] = field(default_factory=set)  # Core-A: spine is anchor
    core_b_ids: Set[str] = field(default_factory=set)  # Core-B: structural witnesses
    periphery_incident_ids: Set[str] = field(default_factory=set)  # Semantic-only
    rejected_incident_ids: Set[str] = field(default_factory=set)  # Hub-only or unrelated
    membrane_decisions: Dict[str, MembershipDecision] = field(default_factory=dict)  # Audit
    core_leak_rate: float = 0.0  # % core without spine anchor (should be 0)
    core_b_ratio: float = 0.0    # % of core that is Core-B (vs Core-A)

    # OBSERVABILITY: Track candidates that didn't make it to core
    blocked_core_b: List[BlockedCoreB] = field(default_factory=list)  # Core-B candidates that failed
    periphery_candidates: Set[str] = field(default_factory=set)  # Incidents considered but only periphery
    candidate_pool_size: int = 0  # Total incidents considered (for stats)

    def to_entity_case(self) -> "EntityCase":
        """
        Convert CompleteStory to EntityCase for API compatibility.

        This enables migration from PrincipledCaseBuilder to StoryBuilder
        without changing downstream consumers that expect EntityCase format.

        NOTE: Only includes CORE incidents (Core-A + Core-B).
        Periphery incidents are NOT included to prevent mega-case formation.
        Periphery incidents lack structural binding and would create hub-like behavior.
        """
        from .case_builder import EntityCase, MembershipLevel as CaseMembershipLevel

        # CORE ONLY: Exclude periphery to prevent mega-cases
        core_ids = self.core_a_ids | self.core_b_ids

        # Extract companion entities (all non-spine anchors in core incidents)
        companion_entities: Dict[str, int] = {}
        sample_headlines: List[str] = []

        # Note: We don't have direct access to incidents here, but canonical_worker
        # will handle this via incident_contexts. For now, companions are inferred.
        # The caller (canonical_worker) will enrich this via incident_contexts.

        # Map membership weights to case_builder format (core only)
        membership_weights = {}
        for inc_id, level in self.membership_weights.items():
            if inc_id in core_ids:  # Only include core
                membership_weights[inc_id] = level

        return EntityCase(
            entity=self.spine,
            entity_case_id=self.story_id,
            incident_ids=core_ids,  # CORE ONLY
            core_incident_ids=core_ids,
            periphery_incident_ids=set(),  # Empty - periphery excluded
            total_incidents=len(core_ids),
            time_start=self.time_start,
            time_end=self.time_end,
            companion_entities=companion_entities,  # Enriched by caller
            sample_headlines=sample_headlines,  # Enriched by caller
            membership_weights=membership_weights,
            is_hub=False,  # Stories are built from non-hub spines
            explanation=self.explanation,
        )


@dataclass
class StoryBuilderResult:
    """Result of story building."""
    stories: Dict[str, CompleteStory]  # story_id → CompleteStory
    spines: Dict[str, StorySpine]  # entity → spine info
    facet_index: Dict[str, Set[str]]  # question_key → surface_ids (global)
    ledger: ConstraintLedger
    stats: Dict[str, Any]


# =============================================================================
# STORY BUILDER
# =============================================================================

class StoryBuilder:
    """
    Builds stories from incidents using spine + mode + membrane approach.

    This replaces PrincipledCaseBuilder's k=2 motif recurrence with a model
    that works for star-shaped news patterns.
    """

    def __init__(
        self,
        hub_fraction_threshold: float = 0.20,  # Entity in >20% of incidents = hub
        hub_min_incidents: int = 5,  # Need ≥5 incidents to compute hubness
        min_incidents_for_story: int = 3,  # Minimum incidents to form a story
        mode_gap_days: int = 30,  # Gap to separate temporal modes
        expected_facets_fire: Set[str] = None,  # Expected facets for fire stories
    ):
        self.hub_fraction_threshold = hub_fraction_threshold
        self.hub_min_incidents = hub_min_incidents
        self.min_incidents_for_story = min_incidents_for_story
        self.mode_gap_days = mode_gap_days

        # Default expected facets for fire stories
        self.expected_facets_fire = expected_facets_fire or {
            "fire_death_count",
            "fire_injury_count",
            "fire_cause",
            "fire_status",
            "fire_response",
        }

        self.ledger = ConstraintLedger()

    def build_from_incidents(
        self,
        incidents: Dict[str, Event],
        surfaces: Dict[str, Surface] = None,
    ) -> StoryBuilderResult:
        """
        Build stories from incidents using spine-based clustering.

        Strategy:
        1. Compute hubness for all entities
        2. Find spine candidates (non-hub entities with sufficient incidents)
        3. For each spine, detect temporal modes
        4. For each spine + mode, build story with facets
        5. Compute Jaynes-driven inquiries for story completeness
        """
        self.ledger = ConstraintLedger()
        surfaces = surfaces or {}

        # Step 1: Compute entity hubness
        entity_incidents, hubness = self._compute_hubness(incidents)

        # Step 2: Find spine candidates
        spines = self._find_spine_candidates(entity_incidents, hubness, incidents)

        # Step 3: Build stories for each spine + mode
        stories = {}
        for spine_entity, spine in spines.items():
            if spine.is_hub:
                continue  # Hubs cannot define stories

            for mode in spine.temporal_modes:
                if len(mode.incident_ids) < self.min_incidents_for_story:
                    continue

                story = self._build_story(
                    spine_entity, mode, incidents, surfaces, hubness
                )
                if story:
                    stories[story.story_id] = story

        # Build global facet index
        facet_index = self._build_facet_index(surfaces)

        stats = {
            "total_incidents": len(incidents),
            "total_surfaces": len(surfaces),
            "spine_candidates": len(spines),
            "hub_entities": sum(1 for s in spines.values() if s.is_hub),
            "stories_formed": len(stories),
            "total_facets": len(facet_index),
        }

        return StoryBuilderResult(
            stories=stories,
            spines=spines,
            facet_index=facet_index,
            ledger=self.ledger,
            stats=stats,
        )

    def _compute_hubness(
        self,
        incidents: Dict[str, Event],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, bool]]:
        """
        Compute which entities are hubs (too ubiquitous to define stories).
        """
        entity_incidents: Dict[str, Set[str]] = defaultdict(set)

        for incident_id, incident in incidents.items():
            for entity in incident.anchor_entities:
                entity_incidents[entity].add(incident_id)

        total = len(incidents)
        hubness = {}

        for entity, inc_set in entity_incidents.items():
            fraction = len(inc_set) / total if total > 0 else 0

            if total < self.hub_min_incidents:
                is_hub = False
            elif fraction >= self.hub_fraction_threshold:
                is_hub = True
            else:
                is_hub = False

            hubness[entity] = is_hub

            if is_hub:
                self.ledger.add(Constraint(
                    constraint_type=ConstraintType.STRUCTURAL,
                    assertion=f"Entity '{entity}' is hub ({fraction:.0%} of incidents)",
                    evidence={
                        "entity": entity,
                        "incident_count": len(inc_set),
                        "fraction": round(fraction, 3),
                        "threshold": self.hub_fraction_threshold,
                    },
                    provenance="hub_detection"
                ), scope=f"hub:{entity}")

        return entity_incidents, hubness

    def _find_spine_candidates(
        self,
        entity_incidents: Dict[str, Set[str]],
        hubness: Dict[str, bool],
        incidents: Dict[str, Event],
    ) -> Dict[str, StorySpine]:
        """
        Find entities that can serve as story spines.
        """
        spines = {}
        total = len(incidents)

        for entity, inc_set in entity_incidents.items():
            if len(inc_set) < self.min_incidents_for_story:
                continue

            is_hub = hubness.get(entity, False)
            fraction = len(inc_set) / total if total > 0 else 0

            # Detect temporal modes
            modes = self._detect_temporal_modes(entity, inc_set, incidents)

            spine_id = f"spine_{hashlib.md5(entity.encode()).hexdigest()[:12]}"

            explanation = (
                f"{'HUB (cannot define story)' if is_hub else 'Valid spine'}: "
                f"{entity} appears in {len(inc_set)} incidents ({fraction:.0%}), "
                f"{len(modes)} temporal modes"
            )

            spines[entity] = StorySpine(
                entity=entity,
                spine_id=spine_id,
                temporal_modes=modes,
                total_incidents=len(inc_set),
                incident_fraction=fraction,
                is_hub=is_hub,
                explanation=explanation,
            )

        return spines

    def _detect_temporal_modes(
        self,
        spine: str,
        incident_ids: Set[str],
        incidents: Dict[str, Event],
    ) -> List[TemporalMode]:
        """
        Detect temporal modes for a spine entity.

        Uses gap-based clustering: if gap between incidents > mode_gap_days,
        they belong to different temporal modes.
        """
        # Get timestamps for incidents
        timed_incidents = []
        for inc_id in incident_ids:
            incident = incidents.get(inc_id)
            if incident and incident.time_window[0]:
                timed_incidents.append((incident.time_window[0], inc_id))

        if not timed_incidents:
            # No timestamps, single mode
            mode_id = f"mode_{hashlib.md5(spine.encode()).hexdigest()[:8]}_0"
            return [TemporalMode(
                mode_id=mode_id,
                spine=spine,
                time_start=None,
                time_end=None,
                incident_ids=incident_ids,
                explanation="No timestamps available, single mode assumed",
            )]

        # Sort by time
        timed_incidents.sort(key=lambda x: x[0])

        # Cluster by gap
        modes = []
        current_mode_incidents = {timed_incidents[0][1]}
        current_mode_start = timed_incidents[0][0]
        current_mode_end = timed_incidents[0][0]

        for i in range(1, len(timed_incidents)):
            ts, inc_id = timed_incidents[i]
            gap = (ts - current_mode_end).days

            if gap > self.mode_gap_days:
                # New mode
                mode_id = f"mode_{hashlib.md5(spine.encode()).hexdigest()[:8]}_{len(modes)}"
                modes.append(TemporalMode(
                    mode_id=mode_id,
                    spine=spine,
                    time_start=current_mode_start,
                    time_end=current_mode_end,
                    incident_ids=current_mode_incidents,
                    explanation=f"Temporal mode {len(modes)}: {current_mode_start.date()} to {current_mode_end.date()}",
                ))
                current_mode_incidents = {inc_id}
                current_mode_start = ts
                current_mode_end = ts
            else:
                # Same mode
                current_mode_incidents.add(inc_id)
                current_mode_end = max(current_mode_end, ts)

        # Final mode
        mode_id = f"mode_{hashlib.md5(spine.encode()).hexdigest()[:8]}_{len(modes)}"
        modes.append(TemporalMode(
            mode_id=mode_id,
            spine=spine,
            time_start=current_mode_start,
            time_end=current_mode_end,
            incident_ids=current_mode_incidents,
            explanation=f"Temporal mode {len(modes)}: {current_mode_start.date()} to {current_mode_end.date()}",
        ))

        # Add incidents without timestamps to the most recent mode
        untimed = incident_ids - {inc_id for _, inc_id in timed_incidents}
        if untimed and modes:
            modes[-1].incident_ids.update(untimed)

        return modes

    def _build_story(
        self,
        spine: str,
        mode: TemporalMode,
        incidents: Dict[str, Event],
        surfaces: Dict[str, Surface],
        hubness: Dict[str, bool],
    ) -> Optional[CompleteStory]:
        """
        Build a complete story for a spine + mode.

        Uses classify_incident_membership() as the single membership gate.

        OBSERVABILITY FIX: We now consider ALL incidents in the time window
        as potential candidates, not just those where spine is anchor.
        This allows Core-B candidates (structural witnesses) to be discovered
        and tracked even when they don't have spine as anchor.
        """
        # Create focal set for this story (single spine by default)
        focal_set = FocalSet(primary=spine)

        # Hub entities set for membrane
        hub_entities = {e for e, is_hub in hubness.items() if is_hub}

        # STEP 1: Build candidate pool = mode incidents + time-window incidents
        # Mode incidents are where spine is anchor (Core-A candidates)
        # Time-window incidents are potential Core-B candidates
        candidate_ids = set(mode.incident_ids)

        # Expand to all incidents in the time window
        if mode.time_start and mode.time_end:
            time_buffer = timedelta(days=7)  # 1 week buffer around mode
            window_start = mode.time_start - time_buffer
            window_end = mode.time_end + time_buffer

            for inc_id, incident in incidents.items():
                if inc_id in candidate_ids:
                    continue
                if incident.time_window[0]:
                    if window_start <= incident.time_window[0] <= window_end:
                        candidate_ids.add(inc_id)

        # STEP 2: Classify all candidates using membrane
        core_incidents = set()
        core_a_ids = set()
        core_b_ids = set()
        periphery_incidents = set()
        rejected_incidents = set()
        blocked_core_b = []
        periphery_candidates = set()
        membership = {}
        membrane_decisions = {}

        for inc_id in candidate_ids:
            incident = incidents.get(inc_id)
            if not incident:
                continue

            # Get constraints for this incident (currently empty - future: from ledger)
            # TODO: Load constraints from ledger when available
            constraints = []

            # Call the membrane decision table
            decision = classify_incident_membership(
                incident_anchors=incident.anchor_entities,
                focal_set=focal_set,
                constraints=constraints,
                hub_entities=hub_entities,
            )

            membrane_decisions[inc_id] = decision

            # Map membrane decision to our categories
            if decision.membership == Membership.CORE:
                core_incidents.add(inc_id)
                membership[inc_id] = MembershipLevel.CORE

                if decision.core_reason == CoreReason.ANCHOR:
                    core_a_ids.add(inc_id)
                elif decision.core_reason == CoreReason.WARRANT:
                    core_b_ids.add(inc_id)

            elif decision.membership == Membership.PERIPHERY:
                periphery_incidents.add(inc_id)
                periphery_candidates.add(inc_id)
                membership[inc_id] = MembershipLevel.PERIPHERY

                # Track blocked Core-B candidates (had some witnesses but not enough)
                if decision.witnesses and decision.blocked_reason:
                    blocked_core_b.append(BlockedCoreB(
                        incident_id=inc_id,
                        reason=decision.blocked_reason,
                        witnesses_found=decision.witnesses,
                        witnesses_missing="need ≥2 structural witnesses with ≥1 non-time",
                    ))

            elif decision.membership == Membership.REJECT:
                rejected_incidents.add(inc_id)
                # Don't add to membership - rejected incidents are not persisted

        # Collect surfaces from incidents
        all_surface_ids = set()
        for inc_id in mode.incident_ids:
            incident = incidents.get(inc_id)
            if incident:
                all_surface_ids.update(incident.surface_ids)

        # Build facets from surfaces
        facets = self._build_facets(all_surface_ids, surfaces)

        # ===========================================
        # FACET CONTRACT COMPUTATION (see module docstring)
        # ===========================================

        # Determine expected facets based on story type (schema-driven)
        # For now, detect fire stories by spine name or question_keys
        is_fire_story = any(
            "fire" in qk.lower() for qk in facets.keys()
        )

        if is_fire_story:
            expected_facets = self.expected_facets_fire
        else:
            # Default: expect nothing specific (non-fire stories)
            # This prevents tautological 100% completeness
            expected_facets = set()

        # 1. FACET PRESENCE: what we actually have
        present_facets = set(facets.keys())

        # 2. FACET COVERAGE: present ∩ expected (what we have that matters)
        coverage_facets = present_facets & expected_facets

        # 3. FACET NOISE: present - expected (unexpected facets)
        noise_facets = present_facets - expected_facets

        # 4. BLOCKED FACETS: expected but structurally impossible
        #    TODO: This requires deeper analysis of available sources
        #    For now, we track empty blocked_facets
        blocked_facets: Dict[str, str] = {}

        # 5. MISSING FACETS: expected - present (gaps, not blocked)
        #    True missing = expected - present - blocked
        missing_facets = expected_facets - present_facets - set(blocked_facets.keys())

        # COMPLETENESS SCORE: |coverage| / |expected|
        # Note: blocked facets should be removed from expected for fair scoring
        effective_expected = expected_facets - set(blocked_facets.keys())
        completeness = len(coverage_facets) / len(effective_expected) if effective_expected else 1.0

        # Generate inquiries
        conflict_inquiries = []
        gap_inquiries = []
        quality_inquiries = []

        for facet in facets.values():
            if facet.has_conflict:
                conflict_inquiries.append({
                    "facet": facet.question_key,
                    "type": "conflict",
                    "reason": f"Conflicting values for {facet.question_key}",
                })

        for missing in missing_facets:
            gap_inquiries.append({
                "facet": missing,
                "type": "gap",
                "reason": f"Expected facet {missing} not found",
            })

        # Aggregate stats
        total_claims = sum(f.claim_count for f in facets.values())
        total_sources = sum(f.source_count for f in facets.values())

        # Generate title
        title = f"{spine}"
        if mode.time_start:
            title += f" ({mode.time_start.strftime('%b %Y')})"

        description = (
            f"Story about {spine}. Candidate pool: {len(candidate_ids)} incidents. "
            f"Facets: {len(present_facets)}/{len(expected_facets)} ({completeness:.0%} complete). "
            f"Core: {len(core_incidents)} (A:{len(core_a_ids)}, B:{len(core_b_ids)}), "
            f"Periphery: {len(periphery_incidents)}, Rejected: {len(rejected_incidents)}, "
            f"Blocked Core-B: {len(blocked_core_b)}."
        )

        story_id = f"story_{hashlib.md5(f'{spine}_{mode.mode_id}'.encode()).hexdigest()[:12]}"

        # Compute membrane health metrics
        # Core leak rate: % core incidents without spine anchor (should be 0 with proper membrane)
        incidents_with_spine_anchor = core_a_ids  # Core-A is exactly spine-as-anchor
        leak_rate = compute_core_leak_rate(core_incidents, incidents_with_spine_anchor)

        # Core-B ratio: what fraction of core is Core-B (vs Core-A)
        core_b_ratio = len(core_b_ids) / len(core_incidents) if core_incidents else 0.0

        explanation = (
            f"Spine-based story: {spine} + mode {mode.mode_id}. "
            f"Core-A: {len(core_a_ids)} (spine as anchor), Core-B: {len(core_b_ids)} (structural witnesses). "
            f"Completeness: {completeness:.0%}. Leak rate: {leak_rate:.0%}."
        )

        return CompleteStory(
            story_id=story_id,
            spine=spine,
            mode=mode,
            focal_set=focal_set,
            core_incident_ids=core_incidents,
            core_a_ids=core_a_ids,
            core_b_ids=core_b_ids,
            periphery_incident_ids=periphery_incidents,
            rejected_incident_ids=rejected_incidents,
            membrane_decisions=membrane_decisions,
            facets=facets,
            title=title,
            description=description,
            time_start=mode.time_start,
            time_end=mode.time_end,
            surface_count=len(all_surface_ids),
            source_count=total_sources,
            claim_count=total_claims,
            expected_facets=expected_facets,
            present_facets=present_facets,
            coverage_facets=coverage_facets,
            missing_facets=missing_facets,
            noise_facets=noise_facets,
            blocked_facets=blocked_facets,
            completeness_score=completeness,
            conflict_inquiries=conflict_inquiries,
            gap_inquiries=gap_inquiries,
            quality_inquiries=quality_inquiries,
            membership_weights=membership,
            core_leak_rate=leak_rate,
            core_b_ratio=core_b_ratio,
            explanation=explanation,
            # OBSERVABILITY fields
            blocked_core_b=blocked_core_b,
            periphery_candidates=periphery_candidates,
            candidate_pool_size=len(candidate_ids),
        )

    def _build_facets(
        self,
        surface_ids: Set[str],
        surfaces: Dict[str, Surface],
    ) -> Dict[str, StoryFacet]:
        """
        Build facets by grouping surfaces by question_key.
        """
        facets: Dict[str, StoryFacet] = {}

        # Group surfaces by question_key
        by_qkey: Dict[str, Set[str]] = defaultdict(set)

        for surf_id in surface_ids:
            surface = surfaces.get(surf_id)
            if not surface:
                continue

            qkey = getattr(surface, 'question_key', None)
            if qkey:
                by_qkey[qkey].add(surf_id)

        # Build facet for each question_key
        for qkey, surf_set in by_qkey.items():
            # Count claims and sources
            claim_count = 0
            sources = set()

            for surf_id in surf_set:
                surface = surfaces.get(surf_id)
                if surface:
                    claim_count += getattr(surface, 'claim_count', 1)
                    source = getattr(surface, 'primary_source', None)
                    if source:
                        sources.add(source)

            # Detect domain type from question_key
            if "_count" in qkey:
                typed_domain = "count"
            elif "_status" in qkey or "_cause" in qkey:
                typed_domain = "categorical"
            else:
                typed_domain = "text"

            # TODO: Compute actual TypedBeliefState and detect conflicts
            has_conflict = len(surf_set) > 1  # Simplified: multiple surfaces = potential conflict

            facets[qkey] = StoryFacet(
                question_key=qkey,
                surface_ids=surf_set,
                claim_count=claim_count,
                source_count=len(sources),
                typed_domain=typed_domain,
                has_conflict=has_conflict,
            )

        return facets

    def _build_facet_index(
        self,
        surfaces: Dict[str, Surface],
    ) -> Dict[str, Set[str]]:
        """
        Build global index of question_key → surface_ids.
        """
        index: Dict[str, Set[str]] = defaultdict(set)

        for surf_id, surface in surfaces.items():
            qkey = getattr(surface, 'question_key', None)
            if qkey:
                index[qkey].add(surf_id)

        return dict(index)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StoryBuilder',
    'StoryBuilderResult',
    'CompleteStory',
    'StorySpine',
    'TemporalMode',
    'StoryFacet',
    'BlockedCoreB',
]
