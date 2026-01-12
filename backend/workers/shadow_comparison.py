"""
Shadow Comparison - Weaver v2 vs Legacy Principled Weaver
==========================================================

Run both weavers on same claims WITHOUT persisting, compare decisions.

Measures:
- Precision risk: where does v2 merge that legacy wouldn't?
- Recall gain: where does v2 join that legacy splits?
- Cost/latency: LLM calls per claim after caching + conditional calling

Outputs per-claim diff records and aggregate metrics.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional, List, Set, Dict, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.neo4j_service import Neo4jService
from services.job_queue import JobQueue
from repositories.claim_repository import ClaimRepository
from models.domain.claim import Claim

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DECISION RECORDS
# =============================================================================

@dataclass
class SurfaceDecision:
    """Surface routing decision."""
    scope_id: str
    question_key: str
    surface_signature: str  # Hash of (scope_id, question_key)
    is_new: bool
    existing_surface_id: Optional[str] = None
    confidence: float = 1.0
    method: str = "pattern"  # pattern, llm, singleton


@dataclass
class IncidentDecision:
    """Incident routing decision."""
    incident_signature: str  # Hash of anchor set
    is_new: bool
    existing_incident_id: Optional[str] = None
    shared_anchors: Set[str] = field(default_factory=set)
    companion_overlap: float = 0.0
    bridge_blocked: bool = False
    blocking_entity: Optional[str] = None


@dataclass
class ClaimDecisionRecord:
    """Complete decision record for a single claim."""
    claim_id: str
    claim_text_preview: str
    weaver_type: str  # "legacy" or "v2"

    # Surface decision
    surface: SurfaceDecision = None

    # Incident decision
    incident: IncidentDecision = None

    # Meta
    processing_time_ms: float = 0.0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    meta_claims: List[str] = field(default_factory=list)
    error: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiffRecord:
    """Difference between legacy and v2 decisions for same claim."""
    claim_id: str
    claim_text_preview: str

    # Surface diff
    surface_agrees: bool = True
    legacy_surface_key: Optional[str] = None
    v2_surface_key: Optional[str] = None
    surface_diff_type: str = "none"  # none, key_diff, new_vs_join, join_vs_new

    # Incident diff
    incident_agrees: bool = True
    legacy_incident_sig: Optional[str] = None
    v2_incident_sig: Optional[str] = None
    incident_diff_type: str = "none"  # none, split, merge, bridge_diff

    # Analysis
    v2_more_precise: bool = False  # v2 splits what legacy merges (precision)
    v2_more_recall: bool = False  # v2 merges what legacy splits (recall)
    v2_llm_calls: int = 0
    processing_time_diff_ms: float = 0.0


@dataclass
class AggregateMetrics:
    """Aggregate comparison metrics."""
    total_claims: int = 0

    # Agreement rates (by semantic key, comparing same questions get same key)
    surface_key_agreement_rate: float = 0.0  # Same question_key for same claim
    incident_membership_agreement_rate: float = 0.0  # Same new/join decision

    # Precision/Recall (head-to-head comparison)
    v2_precision_cases: int = 0  # v2 splits what legacy merges (more conservative)
    v2_recall_cases: int = 0  # v2 merges what legacy splits (more aggressive)

    # Key: neither precision nor recall = "same behavior"
    # precision + recall = 0 with high agreement = good
    # high precision = v2 is more conservative (safer)
    # high recall = v2 is more aggressive (risky)

    # Join rates (within each weaver's simulated namespace)
    # These measure "internal joinability" - how often claims join existing surfaces
    legacy_surface_join_rate: float = 0.0
    v2_surface_join_rate: float = 0.0
    legacy_incident_join_rate: float = 0.0
    v2_incident_join_rate: float = 0.0

    # Bridge immunity
    legacy_bridge_blocked: int = 0
    v2_bridge_blocked: int = 0

    # LLM costs
    total_llm_calls: int = 0
    llm_calls_avoided_by_cache: int = 0
    llm_calls_avoided_by_pattern: int = 0

    # Latency
    legacy_p50_ms: float = 0.0
    legacy_p95_ms: float = 0.0
    v2_p50_ms: float = 0.0
    v2_p95_ms: float = 0.0

    # Fallback distribution - how each weaver derives question_key
    # Legacy
    legacy_singleton_count: int = 0  # unscoped_/page_scope_ fallback
    legacy_pattern_count: int = 0    # Pattern-matched question_key
    legacy_entity_fallback_count: int = 0  # about_X pattern (entity-derived)

    # V2
    v2_singleton_count: int = 0  # Including veto fallbacks
    v2_pattern_count: int = 0    # Pattern-matched (no LLM needed)
    v2_llm_count: int = 0        # LLM-derived question_key (successful)
    v2_veto_count: int = 0       # LLM called but vetoed due to low confidence → singleton


# =============================================================================
# SHADOW WEAVERS - Extract decisions without persisting
# =============================================================================

class LegacyShadowWeaver:
    """Legacy weaver in shadow mode - extract decisions, don't persist."""

    def __init__(self, db_pool: asyncpg.Pool, neo4j: Neo4jService):
        self.db_pool = db_pool
        self.neo4j = neo4j
        self.claim_repo = ClaimRepository(db_pool, neo4j)

        # Import QuestionKeyExtractor from principled_weaver
        from workers.principled_weaver import QuestionKeyExtractor
        self.qk_extractor = QuestionKeyExtractor()

        # Hub entities
        self.hub_entities = {"United States", "China", "European Union", "United Nations",
                            "Asia", "Europe", "North America", "South America", "Africa"}

        # Simulated state (not real - for decision tracking)
        self.scoped_key_to_surface: Dict[Tuple[str, str], str] = {}
        self.surface_anchors: Dict[str, Set[str]] = {}
        self.surface_companions: Dict[str, Set[str]] = {}
        self.incidents: Dict[str, Dict] = {}  # incident_id -> {anchors, companions}
        self.surface_to_incident: Dict[str, str] = {}

    async def simulate_claim(self, claim: Claim) -> ClaimDecisionRecord:
        """Simulate processing a claim - return decision without persisting."""
        start = time.time()
        record = ClaimDecisionRecord(
            claim_id=claim.id,
            claim_text_preview=(claim.text or "")[:80],
            weaver_type="legacy",
        )

        try:
            # Hydrate entities
            claim = await self.claim_repo.hydrate_entities(claim)

            # Extract question_key using legacy method
            question_key = self.qk_extractor.extract(claim)

            # Extract anchors (same logic as principled_weaver)
            entities = self._get_entity_names(claim)
            anchor_entities = self._extract_anchors(claim)
            companion_entities = entities - anchor_entities

            # Compute scope_id
            scope_id = self._compute_scope_id(anchor_entities)

            # Surface decision
            scoped_key = (scope_id, question_key)
            surface_sig = f"{scope_id}:{question_key}"

            is_singleton = question_key.startswith("unscoped_") or question_key.startswith("page_scope_")
            method = "singleton" if is_singleton else "pattern"

            if scoped_key in self.scoped_key_to_surface:
                surface_id = self.scoped_key_to_surface[scoped_key]
                record.surface = SurfaceDecision(
                    scope_id=scope_id,
                    question_key=question_key,
                    surface_signature=surface_sig,
                    is_new=False,
                    existing_surface_id=surface_id,
                    method=method,
                )
                # Update companions for this surface
                self.surface_companions.setdefault(surface_id, set()).update(companion_entities)
            else:
                surface_id = f"sf_shadow_{len(self.scoped_key_to_surface)}"
                self.scoped_key_to_surface[scoped_key] = surface_id
                self.surface_anchors[surface_id] = anchor_entities.copy()
                self.surface_companions[surface_id] = companion_entities.copy()
                record.surface = SurfaceDecision(
                    scope_id=scope_id,
                    question_key=question_key,
                    surface_signature=surface_sig,
                    is_new=True,
                    method=method,
                )

            if is_singleton:
                record.meta_claims.append("singleton_fallback")

            # Incident decision
            record.incident = await self._simulate_incident_routing(
                surface_id, anchor_entities, companion_entities
            )

        except Exception as e:
            record.error = str(e)

        record.processing_time_ms = (time.time() - start) * 1000
        return record

    async def _simulate_incident_routing(
        self,
        surface_id: str,
        anchor_entities: Set[str],
        companion_entities: Set[str],
    ) -> IncidentDecision:
        """Simulate incident routing."""
        # Check if already routed
        if surface_id in self.surface_to_incident:
            inc_id = self.surface_to_incident[surface_id]
            return IncidentDecision(
                incident_signature=inc_id,
                is_new=False,
                existing_incident_id=inc_id,
            )

        # Find compatible incident
        MIN_SHARED_ANCHORS = 2
        COMPANION_OVERLAP_THRESHOLD = 0.15

        best_incident = None
        best_overlap = 0.0
        blocked_by = None

        for inc_id, inc_data in self.incidents.items():
            inc_anchors = inc_data['anchors']
            inc_companions = inc_data['companions']

            shared = anchor_entities & inc_anchors
            if len(shared) < MIN_SHARED_ANCHORS:
                continue

            # Check companion compatibility
            if companion_entities and inc_companions:
                intersection = companion_entities & inc_companions
                union = companion_entities | inc_companions
                jaccard = len(intersection) / len(union) if union else 0.0

                if jaccard < COMPANION_OVERLAP_THRESHOLD:
                    # Bridge blocked
                    blocked_by = next(iter(shared)) if shared else None
                    continue

                if jaccard > best_overlap:
                    best_overlap = jaccard
                    best_incident = inc_id
            else:
                # Underpowered - benefit of doubt
                if best_incident is None:
                    best_incident = inc_id
                    best_overlap = 0.5

        if best_incident:
            self.surface_to_incident[surface_id] = best_incident
            self.incidents[best_incident]['anchors'].update(anchor_entities)
            self.incidents[best_incident]['companions'].update(companion_entities)

            return IncidentDecision(
                incident_signature=best_incident,
                is_new=False,
                existing_incident_id=best_incident,
                shared_anchors=anchor_entities & self.incidents[best_incident]['anchors'],
                companion_overlap=best_overlap,
            )

        # Create new incident
        inc_id = f"in_shadow_{len(self.incidents)}"
        self.incidents[inc_id] = {
            'anchors': anchor_entities.copy(),
            'companions': companion_entities.copy(),
        }
        self.surface_to_incident[surface_id] = inc_id

        return IncidentDecision(
            incident_signature=inc_id,
            is_new=True,
            shared_anchors=anchor_entities,
            bridge_blocked=blocked_by is not None,
            blocking_entity=blocked_by,
        )

    def _get_entity_names(self, claim: Claim) -> Set[str]:
        if not claim.entities:
            return set()
        names = set()
        for entity in claim.entities:
            if hasattr(entity, 'canonical_name'):
                names.add(entity.canonical_name)
            elif hasattr(entity, 'name'):
                names.add(entity.name)
            else:
                names.add(str(entity))
        return names

    def _extract_anchors(self, claim: Claim) -> Set[str]:
        if not claim.entities:
            return set()
        anchors = set()
        for entity in claim.entities:
            name = entity.canonical_name if hasattr(entity, 'canonical_name') else str(entity)
            if hasattr(entity, 'entity_type') and entity.entity_type in ('PERSON', 'ORG'):
                anchors.add(name)
            elif hasattr(entity, 'mention_count') and entity.mention_count and entity.mention_count < 50:
                anchors.add(name)
            else:
                anchors.add(name)
        return anchors

    def _compute_scope_id(self, anchor_entities: Set[str]) -> str:
        scoping = anchor_entities - self.hub_entities
        if not scoping:
            scoping = anchor_entities
        normalized = sorted(a.lower().replace(" ", "").replace("'", "") for a in scoping)
        primary = normalized[:2]
        if not primary:
            return "scope_unscoped"
        return "scope_" + "_".join(primary)


class V2ShadowWeaver:
    """V2 weaver in shadow mode - extract decisions, don't persist."""

    def __init__(self, db_pool: asyncpg.Pool, neo4j: Neo4jService, enable_llm: bool = True):
        self.db_pool = db_pool
        self.neo4j = neo4j
        self.claim_repo = ClaimRepository(db_pool, neo4j)
        self.enable_llm = enable_llm

        # Import LLMAdjudicator from weaver_v2
        if enable_llm:
            from workers.weaver_v2 import LLMAdjudicator
            self.adjudicator = LLMAdjudicator()
        else:
            self.adjudicator = None

        # Hub entities
        self.hub_entities = {"United States", "China", "European Union", "United Nations",
                            "Asia", "Europe", "North America", "South America", "Africa"}

        # Stats
        self.llm_calls = 0
        self.llm_cache_hits = 0
        self.pattern_matches = 0
        self.veto_count = 0  # LLM called but confidence too low → singleton

        # Simulated state
        self.scoped_key_to_surface: Dict[Tuple[str, str], str] = {}
        self.surface_anchors: Dict[str, Set[str]] = {}
        self.surface_companions: Dict[str, Set[str]] = {}
        self.incidents: Dict[str, Dict] = {}
        self.surface_to_incident: Dict[str, str] = {}

    async def simulate_claim(self, claim: Claim) -> ClaimDecisionRecord:
        """Simulate processing a claim with v2 logic."""
        start = time.time()
        record = ClaimDecisionRecord(
            claim_id=claim.id,
            claim_text_preview=(claim.text or "")[:80],
            weaver_type="v2",
        )

        try:
            # Hydrate entities
            claim = await self.claim_repo.hydrate_entities(claim)

            # Extract entities
            entities = self._get_entity_names(claim)
            anchor_entities = self._extract_anchors(claim)
            companion_entities = entities - anchor_entities

            # Compute scope_id
            scope_id = self._compute_scope_id(anchor_entities)

            # Extract question_key with v2 logic (pattern -> LLM fallback)
            question_key, confidence, method = await self._extract_question_key(
                claim, entities, anchor_entities
            )

            record.llm_calls = self.llm_calls

            # Surface decision
            scoped_key = (scope_id, question_key)
            surface_sig = f"{scope_id}:{question_key}"

            if scoped_key in self.scoped_key_to_surface:
                surface_id = self.scoped_key_to_surface[scoped_key]
                record.surface = SurfaceDecision(
                    scope_id=scope_id,
                    question_key=question_key,
                    surface_signature=surface_sig,
                    is_new=False,
                    existing_surface_id=surface_id,
                    confidence=confidence,
                    method=method,
                )
                self.surface_companions.setdefault(surface_id, set()).update(companion_entities)
            else:
                surface_id = f"sf_v2shadow_{len(self.scoped_key_to_surface)}"
                self.scoped_key_to_surface[scoped_key] = surface_id
                self.surface_anchors[surface_id] = anchor_entities.copy()
                self.surface_companions[surface_id] = companion_entities.copy()
                record.surface = SurfaceDecision(
                    scope_id=scope_id,
                    question_key=question_key,
                    surface_signature=surface_sig,
                    is_new=True,
                    confidence=confidence,
                    method=method,
                )

            if method == "singleton":
                record.meta_claims.append("singleton_fallback")
            elif method == "llm":
                record.meta_claims.append("llm_adjudicated")

            # Incident decision (same logic as legacy for fair comparison)
            record.incident = await self._simulate_incident_routing(
                surface_id, anchor_entities, companion_entities
            )

        except Exception as e:
            record.error = str(e)

        record.processing_time_ms = (time.time() - start) * 1000
        return record

    async def _extract_question_key(
        self,
        claim: Claim,
        entities: Set[str],
        anchors: Set[str],
    ) -> Tuple[str, float, str]:
        """Extract question_key with pattern -> LLM fallback."""
        text = (claim.text or "").lower()

        # Pattern matching first
        pattern_key = self._match_patterns(text)
        if pattern_key:
            self.pattern_matches += 1
            return pattern_key, 0.8, "pattern"

        # LLM fallback
        if self.enable_llm and self.adjudicator:
            schema = await self.adjudicator.extract_proposition(
                claim.text or "",
                entities,
                claim.id,
            )
            self.llm_calls += 1

            # LLM veto policy: low confidence -> singleton
            if schema.confidence >= 0.6:
                return schema.proposition_key, schema.confidence, "llm"
            else:
                # Low confidence - fall back to singleton (conservative)
                self.veto_count += 1  # Track veto
                return f"singleton_{claim.id}", 0.1, "singleton"

        # No LLM - singleton
        return f"singleton_{claim.id}", 0.1, "singleton"

    def _match_patterns(self, text: str) -> Optional[str]:
        """Pattern matching for common proposition types."""
        death_patterns = ['kill', 'dead', 'death', 'fatality', 'died', 'perish']
        if any(p in text for p in death_patterns):
            event_type = self._infer_event_type(text)
            return f"{event_type}_death_count"

        injury_patterns = ['injur', 'wound', 'hurt', 'hospitali']
        if any(p in text for p in injury_patterns):
            event_type = self._infer_event_type(text)
            return f"{event_type}_injury_count"

        status_patterns = ['status', 'condition', 'ongoing', 'active', 'resolved']
        if any(p in text for p in status_patterns):
            event_type = self._infer_event_type(text)
            return f"{event_type}_status"

        policy_patterns = ['announc', 'policy', 'legislation', 'bill', 'reform']
        if any(p in text for p in policy_patterns):
            return "policy_announcement"

        return None

    def _infer_event_type(self, text: str) -> str:
        if 'fire' in text or 'blaze' in text:
            return "fire"
        if 'flood' in text:
            return "flood"
        if 'earthquake' in text or 'quake' in text:
            return "earthquake"
        if 'storm' in text or 'typhoon' in text:
            return "storm"
        if 'crash' in text or 'accident' in text:
            return "accident"
        return "incident"

    async def _simulate_incident_routing(
        self,
        surface_id: str,
        anchor_entities: Set[str],
        companion_entities: Set[str],
    ) -> IncidentDecision:
        """Simulate incident routing (same as legacy for comparison)."""
        if surface_id in self.surface_to_incident:
            inc_id = self.surface_to_incident[surface_id]
            return IncidentDecision(
                incident_signature=inc_id,
                is_new=False,
                existing_incident_id=inc_id,
            )

        MIN_SHARED_ANCHORS = 2
        COMPANION_OVERLAP_THRESHOLD = 0.15

        best_incident = None
        best_overlap = 0.0
        blocked_by = None

        for inc_id, inc_data in self.incidents.items():
            inc_anchors = inc_data['anchors']
            inc_companions = inc_data['companions']

            shared = anchor_entities & inc_anchors
            if len(shared) < MIN_SHARED_ANCHORS:
                continue

            if companion_entities and inc_companions:
                intersection = companion_entities & inc_companions
                union = companion_entities | inc_companions
                jaccard = len(intersection) / len(union) if union else 0.0

                if jaccard < COMPANION_OVERLAP_THRESHOLD:
                    blocked_by = next(iter(shared)) if shared else None
                    continue

                if jaccard > best_overlap:
                    best_overlap = jaccard
                    best_incident = inc_id
            else:
                if best_incident is None:
                    best_incident = inc_id
                    best_overlap = 0.5

        if best_incident:
            self.surface_to_incident[surface_id] = best_incident
            self.incidents[best_incident]['anchors'].update(anchor_entities)
            self.incidents[best_incident]['companions'].update(companion_entities)

            return IncidentDecision(
                incident_signature=best_incident,
                is_new=False,
                existing_incident_id=best_incident,
                shared_anchors=anchor_entities & self.incidents[best_incident]['anchors'],
                companion_overlap=best_overlap,
            )

        inc_id = f"in_v2shadow_{len(self.incidents)}"
        self.incidents[inc_id] = {
            'anchors': anchor_entities.copy(),
            'companions': companion_entities.copy(),
        }
        self.surface_to_incident[surface_id] = inc_id

        return IncidentDecision(
            incident_signature=inc_id,
            is_new=True,
            shared_anchors=anchor_entities,
            bridge_blocked=blocked_by is not None,
            blocking_entity=blocked_by,
        )

    def _get_entity_names(self, claim: Claim) -> Set[str]:
        if not claim.entities:
            return set()
        names = set()
        for entity in claim.entities:
            if hasattr(entity, 'canonical_name'):
                names.add(entity.canonical_name)
            elif hasattr(entity, 'name'):
                names.add(entity.name)
            else:
                names.add(str(entity))
        return names

    def _extract_anchors(self, claim: Claim) -> Set[str]:
        if not claim.entities:
            return set()
        anchors = set()
        for entity in claim.entities:
            name = entity.canonical_name if hasattr(entity, 'canonical_name') else str(entity)
            if hasattr(entity, 'entity_type') and entity.entity_type in ('PERSON', 'ORG'):
                anchors.add(name)
            elif hasattr(entity, 'mention_count') and entity.mention_count and entity.mention_count < 50:
                anchors.add(name)
            else:
                anchors.add(name)
        return anchors

    def _compute_scope_id(self, anchor_entities: Set[str]) -> str:
        scoping = anchor_entities - self.hub_entities
        if not scoping:
            scoping = anchor_entities
        normalized = sorted(a.lower().replace(" ", "").replace("'", "") for a in scoping)
        primary = normalized[:2]
        if not primary:
            return "scope_unscoped"
        return "scope_" + "_".join(primary)


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

class ShadowComparisonRunner:
    """Run shadow comparison between legacy and v2 weavers."""

    def __init__(self, db_pool: asyncpg.Pool, neo4j: Neo4jService, enable_llm: bool = True):
        self.db_pool = db_pool
        self.neo4j = neo4j
        self.claim_repo = ClaimRepository(db_pool, neo4j)

        self.legacy_weaver = LegacyShadowWeaver(db_pool, neo4j)
        self.v2_weaver = V2ShadowWeaver(db_pool, neo4j, enable_llm=enable_llm)

        self.legacy_records: List[ClaimDecisionRecord] = []
        self.v2_records: List[ClaimDecisionRecord] = []
        self.diff_records: List[DiffRecord] = []

    async def run_comparison(self, claim_ids: List[str]) -> AggregateMetrics:
        """Run comparison on a batch of claims."""
        logger.info(f"Running shadow comparison on {len(claim_ids)} claims...")

        for i, claim_id in enumerate(claim_ids):
            claim = await self.claim_repo.get_by_id(claim_id)
            if not claim:
                continue

            # Run legacy
            legacy_record = await self.legacy_weaver.simulate_claim(claim)
            self.legacy_records.append(legacy_record)

            # Run v2
            v2_record = await self.v2_weaver.simulate_claim(claim)
            self.v2_records.append(v2_record)

            # Compare
            diff = self._compare_decisions(legacy_record, v2_record)
            self.diff_records.append(diff)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(claim_ids)} claims...")

        # Compute aggregate metrics
        return self._compute_aggregates()

    def _compare_decisions(
        self,
        legacy: ClaimDecisionRecord,
        v2: ClaimDecisionRecord,
    ) -> DiffRecord:
        """Compare legacy and v2 decisions for same claim."""
        diff = DiffRecord(
            claim_id=legacy.claim_id,
            claim_text_preview=legacy.claim_text_preview,
        )

        # Surface comparison
        if legacy.surface and v2.surface:
            legacy_key = f"{legacy.surface.scope_id}:{legacy.surface.question_key}"
            v2_key = f"{v2.surface.scope_id}:{v2.surface.question_key}"

            diff.legacy_surface_key = legacy_key
            diff.v2_surface_key = v2_key
            diff.surface_agrees = (legacy_key == v2_key)

            if not diff.surface_agrees:
                # Determine diff type
                if legacy.surface.is_new and not v2.surface.is_new:
                    diff.surface_diff_type = "v2_joins"
                    diff.v2_more_recall = True
                elif not legacy.surface.is_new and v2.surface.is_new:
                    diff.surface_diff_type = "v2_splits"
                    diff.v2_more_precise = True
                else:
                    diff.surface_diff_type = "key_diff"

        # Incident comparison
        if legacy.incident and v2.incident:
            diff.legacy_incident_sig = legacy.incident.incident_signature
            diff.v2_incident_sig = v2.incident.incident_signature

            # We compare by new/join behavior since signatures differ
            if legacy.incident.is_new == v2.incident.is_new:
                diff.incident_agrees = True
            else:
                diff.incident_agrees = False
                if legacy.incident.is_new and not v2.incident.is_new:
                    diff.incident_diff_type = "v2_joins"
                    diff.v2_more_recall = True
                elif not legacy.incident.is_new and v2.incident.is_new:
                    diff.incident_diff_type = "v2_splits"
                    diff.v2_more_precise = True

        diff.v2_llm_calls = v2.llm_calls
        diff.processing_time_diff_ms = v2.processing_time_ms - legacy.processing_time_ms

        return diff

    def _compute_aggregates(self) -> AggregateMetrics:
        """Compute aggregate metrics from all records."""
        metrics = AggregateMetrics(total_claims=len(self.diff_records))

        if not self.diff_records:
            return metrics

        # Agreement rates (head-to-head: same question_key for same claim?)
        surface_agrees = sum(1 for d in self.diff_records if d.surface_agrees)
        incident_agrees = sum(1 for d in self.diff_records if d.incident_agrees)
        metrics.surface_key_agreement_rate = surface_agrees / len(self.diff_records)
        metrics.incident_membership_agreement_rate = incident_agrees / len(self.diff_records)

        # Precision/Recall cases (head-to-head comparison)
        metrics.v2_precision_cases = sum(1 for d in self.diff_records if d.v2_more_precise)
        metrics.v2_recall_cases = sum(1 for d in self.diff_records if d.v2_more_recall)

        # Join rates (within each weaver's namespace)
        legacy_surface_joins = sum(1 for r in self.legacy_records if r.surface and not r.surface.is_new)
        v2_surface_joins = sum(1 for r in self.v2_records if r.surface and not r.surface.is_new)
        metrics.legacy_surface_join_rate = legacy_surface_joins / len(self.legacy_records) if self.legacy_records else 0
        metrics.v2_surface_join_rate = v2_surface_joins / len(self.v2_records) if self.v2_records else 0

        legacy_incident_joins = sum(1 for r in self.legacy_records if r.incident and not r.incident.is_new)
        v2_incident_joins = sum(1 for r in self.v2_records if r.incident and not r.incident.is_new)
        metrics.legacy_incident_join_rate = legacy_incident_joins / len(self.legacy_records) if self.legacy_records else 0
        metrics.v2_incident_join_rate = v2_incident_joins / len(self.v2_records) if self.v2_records else 0

        # Bridge immunity
        metrics.legacy_bridge_blocked = sum(1 for r in self.legacy_records if r.incident and r.incident.bridge_blocked)
        metrics.v2_bridge_blocked = sum(1 for r in self.v2_records if r.incident and r.incident.bridge_blocked)

        # LLM costs
        metrics.total_llm_calls = self.v2_weaver.llm_calls
        metrics.llm_calls_avoided_by_pattern = self.v2_weaver.pattern_matches

        # Latency
        legacy_times = [r.processing_time_ms for r in self.legacy_records if r.processing_time_ms > 0]
        v2_times = [r.processing_time_ms for r in self.v2_records if r.processing_time_ms > 0]

        if legacy_times:
            legacy_times.sort()
            metrics.legacy_p50_ms = statistics.median(legacy_times)
            metrics.legacy_p95_ms = legacy_times[int(len(legacy_times) * 0.95)] if len(legacy_times) > 1 else legacy_times[0]

        if v2_times:
            v2_times.sort()
            metrics.v2_p50_ms = statistics.median(v2_times)
            metrics.v2_p95_ms = v2_times[int(len(v2_times) * 0.95)] if len(v2_times) > 1 else v2_times[0]

        # Fallback distribution - count by method
        for r in self.legacy_records:
            if r.surface:
                if r.surface.method == "singleton":
                    metrics.legacy_singleton_count += 1
                elif r.surface.method == "pattern":
                    # Check if it's an entity-derived pattern (about_X)
                    if r.surface.question_key.startswith("about_"):
                        metrics.legacy_entity_fallback_count += 1
                    else:
                        metrics.legacy_pattern_count += 1

        for r in self.v2_records:
            if r.surface:
                if r.surface.method == "singleton":
                    metrics.v2_singleton_count += 1
                    # Check if this was a veto (has meta_claim indicator)
                    if "singleton_fallback" in r.meta_claims and hasattr(self.v2_weaver, 'adjudicator'):
                        # If we have LLM enabled but got singleton, it might be a veto
                        # This is approximate - we track vetoes more precisely in V2ShadowWeaver
                        pass
                elif r.surface.method == "pattern":
                    metrics.v2_pattern_count += 1
                elif r.surface.method == "llm":
                    metrics.v2_llm_count += 1

        # Count v2 vetoes from weaver stats
        if hasattr(self.v2_weaver, 'veto_count'):
            metrics.v2_veto_count = self.v2_weaver.veto_count

        return metrics

    def get_disagreements(self) -> List[DiffRecord]:
        """Get all claims where legacy and v2 disagree."""
        return [d for d in self.diff_records if not d.surface_agrees or not d.incident_agrees]

    def print_summary(self, metrics: AggregateMetrics):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("SHADOW COMPARISON SUMMARY")
        print("=" * 60)

        print(f"\nTotal claims: {metrics.total_claims}")

        print(f"\n--- Head-to-Head Agreement ---")
        print(f"Surface key agreement: {metrics.surface_key_agreement_rate:.1%}")
        print(f"Incident membership agreement: {metrics.incident_membership_agreement_rate:.1%}")
        print("(same question_key and same new/join decision for each claim)")

        print(f"\n--- Precision vs Recall (head-to-head) ---")
        print(f"v2 more precise (splits what legacy merges): {metrics.v2_precision_cases}")
        print(f"v2 more recall (merges what legacy splits): {metrics.v2_recall_cases}")
        print("(precision=0, recall=0 with high agreement = identical behavior)")

        print(f"\n--- Internal Join Rates (within each namespace) ---")
        print(f"Legacy surface join: {metrics.legacy_surface_join_rate:.1%}")
        print(f"V2 surface join: {metrics.v2_surface_join_rate:.1%}")
        print(f"Legacy incident join: {metrics.legacy_incident_join_rate:.1%}")
        print(f"V2 incident join: {metrics.v2_incident_join_rate:.1%}")
        print("(measures how often claims join existing surfaces/incidents)")

        print(f"\n--- Bridge Immunity ---")
        print(f"Legacy blocked: {metrics.legacy_bridge_blocked}")
        print(f"V2 blocked: {metrics.v2_bridge_blocked}")

        print(f"\n--- Fallback Distribution (Legacy) ---")
        print(f"  Pattern: {metrics.legacy_pattern_count}")
        print(f"  Entity fallback (about_X): {metrics.legacy_entity_fallback_count}")
        print(f"  Singleton: {metrics.legacy_singleton_count}")

        print(f"\n--- Fallback Distribution (V2) ---")
        print(f"  Pattern: {metrics.v2_pattern_count}")
        print(f"  LLM (accepted): {metrics.v2_llm_count}")
        print(f"  LLM veto → singleton: {metrics.v2_veto_count}")
        print(f"  Singleton (other): {metrics.v2_singleton_count - metrics.v2_veto_count}")
        print(f"  Total singleton: {metrics.v2_singleton_count}")

        print(f"\n--- LLM Costs ---")
        print(f"Total LLM calls: {metrics.total_llm_calls}")
        print(f"Avoided by pattern: {metrics.llm_calls_avoided_by_pattern}")
        if metrics.total_llm_calls + metrics.llm_calls_avoided_by_pattern > 0:
            savings_pct = metrics.llm_calls_avoided_by_pattern / (metrics.total_llm_calls + metrics.llm_calls_avoided_by_pattern)
            print(f"Pattern savings: {savings_pct:.1%}")

        print(f"\n--- Latency ---")
        print(f"Legacy p50: {metrics.legacy_p50_ms:.1f}ms, p95: {metrics.legacy_p95_ms:.1f}ms")
        print(f"V2 p50: {metrics.v2_p50_ms:.1f}ms, p95: {metrics.v2_p95_ms:.1f}ms")

        print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run shadow comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Shadow comparison: legacy vs v2 weaver")
    parser.add_argument("--limit", type=int, default=50, help="Number of claims to compare")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM for v2")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    # Connect
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=2,
        max_size=10
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        # Get sample claims
        results = await neo4j._execute_read("""
            MATCH (c:Claim)
            WHERE c.text IS NOT NULL
            RETURN c.id as id
            ORDER BY c.created_at ASC
            LIMIT $limit
        """, {'limit': args.limit})
        claim_ids = [r['id'] for r in results]

        logger.info(f"Found {len(claim_ids)} claims for comparison")

        # Run comparison
        runner = ShadowComparisonRunner(db_pool, neo4j, enable_llm=not args.no_llm)
        metrics = await runner.run_comparison(claim_ids)

        # Print summary
        runner.print_summary(metrics)

        # Print disagreements
        disagreements = runner.get_disagreements()
        if disagreements:
            print(f"\n--- Disagreements ({len(disagreements)}) ---")
            for d in disagreements[:10]:
                print(f"\n{d.claim_id}: {d.claim_text_preview}...")
                if not d.surface_agrees:
                    print(f"  Surface: legacy={d.legacy_surface_key} vs v2={d.v2_surface_key}")
                    print(f"  Type: {d.surface_diff_type}")
                if not d.incident_agrees:
                    print(f"  Incident: {d.incident_diff_type}")

        # Save to file
        if args.output:
            output = {
                "metrics": asdict(metrics),
                "disagreements": [asdict(d) for d in disagreements],
            }
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
