"""
Event Service - Recursive event formation logic

Implements Event.examine() logic for processing claims into event hierarchy.
Uses LLM for intelligent event naming, classification, and claim reasoning.
"""
import uuid
import logging
import os
import json
from typing import List, Optional, Set, Tuple
from datetime import datetime
import numpy as np
from openai import AsyncOpenAI

from models.event import Event, ClaimDecision, ExaminationResult
from models.claim import Claim
from repositories.event_repository import EventRepository
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from services.update_detector import UpdateDetector
from utils.datetime_utils import neo4j_datetime_to_python

logger = logging.getLogger(__name__)


class EventService:
    """
    Service for recursive event formation and claim examination

    Implements the core logic for:
    - Examining claims against events (MERGE/ADD/DELEGATE/YIELD/REJECT)
    - Creating root events from claims
    - Creating sub-events for novel aspects
    - Calculating event embeddings and coherence
    """

    def __init__(
        self,
        event_repo: EventRepository,
        claim_repo: ClaimRepository,
        entity_repo: EntityRepository,
        openai_client: AsyncOpenAI = None
    ):
        self.event_repo = event_repo
        self.claim_repo = claim_repo
        self.entity_repo = entity_repo

        # LLM client for intelligent reasoning
        self.openai_client = openai_client or AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Update detector for systematic metric evolution detection
        self.update_detector = UpdateDetector()

        # Thresholds (from RECURSIVE_EVENT_DESIGN.md)
        self.DUPLICATE_THRESHOLD = 0.9       # Semantic similarity for duplicates
        self.COHERENCE_THRESHOLD = 0.6       # Min coherence to add claim
        self.ENTITY_OVERLAP_THRESHOLD = 0.5  # Entity overlap for topic match
        self.SUB_EVENT_THRESHOLD = 0.3       # Min overlap for yielding sub-event

        # Multi-signal scoring thresholds (issue #3)
        self.ATTACH_THRESHOLD = 0.7          # High confidence → ATTACH
        self.YIELD_THRESHOLD = 0.4           # Medium confidence → YIELD for sub-event
        self.DELEGATE_THRESHOLD = 0.25       # Low confidence → try siblings
        # Below DELEGATE_THRESHOLD → REJECT

        # Signal weights for ensemble scoring
        self.SIGNAL_WEIGHTS = {
            'entity': 0.20,       # Entity overlap (reduced from sole signal)
            'temporal': 0.20,     # Time proximity to event
            'reference': 0.25,    # Event name/location mentioned (HIGH IMPACT!)
            'semantic': 0.15,     # Embedding similarity
            'spatial': 0.10,      # Location overlap
            'causal': 0.10        # Causal keywords
        }

    async def examine_claims(
        self,
        event: Event,
        new_claims: List[Claim]
    ) -> ExaminationResult:
        """
        Recursively examine new claims and integrate into event structure

        Args:
            event: Event to examine claims against
            new_claims: New claims to process

        Returns:
            ExaminationResult with claims_added, sub_events_created, claims_rejected
        """
        logger.info(f"🔍 Examining {len(new_claims)} claims against event: {event.canonical_name}")

        claims_added = []
        claims_rejected = []
        sub_events_created = []
        yielded_claims = []  # Batch YIELD_SUBEVENT decisions for clustering

        # Fetch existing claims for this event (for duplicate detection)
        existing_claims = await self._get_event_claims(event)

        # Generate embeddings on-demand for all claims (existing + new)
        # This mutates claims in-place and only generates for claims without embeddings
        all_claims = existing_claims + new_claims
        await self._generate_claim_embeddings(all_claims)

        # Load sub-events (for delegation)
        sub_events = await self.event_repo.get_sub_events(event.id) if hasattr(self.event_repo, 'get_sub_events') else []

        # First pass: classify all claims
        for claim in new_claims:
            decision = await self._classify_claim(event, claim, existing_claims, sub_events)

            if decision == ClaimDecision.MERGE:
                # Duplicate of existing claim → corroborate
                logger.debug(f"  ✓ MERGE: {claim.text[:50]}...")
                await self._merge_duplicate(event, claim, existing_claims)
                # Link claim to event in Neo4j graph (via repository)
                await self.event_repo.link_claim(event, claim, relationship_type="SUPPORTS")
                claims_added.append(claim)

            elif decision == ClaimDecision.ADD:
                # Novel but fits this event's topic
                logger.debug(f"  ✓ ADD: {claim.text[:50]}...")
                # Link claim to event in Neo4j graph (via repository)
                await self.event_repo.link_claim(event, claim, relationship_type="SUPPORTS")
                claims_added.append(claim)

            elif decision == ClaimDecision.DELEGATE:
                # A sub-event handles this better
                logger.debug(f"  → DELEGATE: {claim.text[:50]}...")
                sub_event = await self._find_best_sub_event(claim, sub_events)
                if sub_event:
                    # Recursive call
                    result = await self.examine_claims(sub_event, [claim])
                    claims_added.extend(result.claims_added)
                    sub_events_created.extend(result.sub_events_created)
                else:
                    # No suitable sub-event, reject
                    claims_rejected.append(claim)

            elif decision == ClaimDecision.YIELD_SUBEVENT:
                # Novel aspect → batch for clustering
                logger.debug(f"  ⚡ YIELD_SUBEVENT: {claim.text[:50]}...")
                yielded_claims.append(claim)

            elif decision == ClaimDecision.REJECT:
                # Doesn't belong here
                logger.debug(f"  ✗ REJECT: {claim.text[:50]}...")
                claims_rejected.append(claim)

        # Second pass: cluster and create sub-events for yielded claims
        if yielded_claims:
            if len(yielded_claims) == 1:
                # Single claim → create simple sub-event
                sub_event = await self._create_sub_event(event, yielded_claims)
                sub_events_created.append(sub_event)
                claims_added.extend(yielded_claims)
            else:
                # Multiple claims → cluster by theme, create intermediate events
                logger.info(f"🧩 Clustering {len(yielded_claims)} yielded claims by theme...")
                clusters = await self._cluster_claims_by_theme(event, yielded_claims)

                for cluster in clusters:
                    # Create intermediate sub-event for this cluster
                    sub_event = await self._create_sub_event(event, cluster['claims'])
                    sub_events_created.append(sub_event)

                    # Recursively examine claims within the cluster
                    if len(cluster['claims']) > 1:
                        result = await self.examine_claims(sub_event, cluster['claims'])
                        # Nested sub-events are already tracked in result
                        sub_events_created.extend(result.sub_events_created)

                    claims_added.extend(cluster['claims'])

        # Update event metrics if claims were added
        if claims_added:
            await self._update_event_metrics(event, existing_claims + claims_added)

            # Generate narrative from claims
            all_claims = existing_claims + claims_added
            event.summary = await self._generate_event_narrative(event, all_claims)
            logger.info(f"📝 Generated narrative for {event.canonical_name}")

            await self.event_repo.update(event)

        # Propagate metadata from sub-events to parent
        if sub_events_created:
            await self._propagate_from_children(event, sub_events_created)

            # Generate parent narrative from sub-events
            await self._generate_parent_narrative(event, sub_events_created)
            await self.event_repo.update(event)

        logger.info(f"📊 Result: {len(claims_added)} added, {len(sub_events_created)} sub-events, {len(claims_rejected)} rejected")

        return ExaminationResult(
            claims_added=claims_added,
            sub_events_created=sub_events_created,
            claims_rejected=claims_rejected
        )

    async def _compute_relatedness_score(
        self,
        event: Event,
        claim: Claim,
        event_entities: set
    ) -> tuple[float, dict]:
        """
        Multi-signal ensemble scoring (issue #3)

        Returns:
            (total_score, signal_scores) where total_score is weighted sum
        """
        scores = {}
        claim_entities = set(claim.entity_ids)

        # Signal 1: Entity overlap (existing)
        if event_entities and claim_entities:
            scores['entity'] = len(event_entities & claim_entities) / max(len(event_entities), len(claim_entities))
        else:
            scores['entity'] = 0.0

        # Signal 2: Temporal proximity
        if claim.event_time and event.event_start:
            # Calculate days difference
            time_diff = abs((claim.event_time - event.event_start).total_seconds() / 86400)
            # Decay: 1.0 at 0 days, 0.5 at 7 days, 0.1 at 30 days
            scores['temporal'] = max(0.0, 1.0 - (time_diff / 30.0))
        else:
            scores['temporal'] = 0.5  # Unknown time → neutral score

        # Signal 3: Reference detection (CHEAP & HIGH IMPACT!)
        scores['reference'] = 1.0 if self._references_event(event, claim, event_entities) else 0.0

        # Signal 4: Semantic similarity
        if claim.embedding and event.embedding:
            from numpy import dot
            from numpy.linalg import norm
            claim_emb = claim.embedding
            event_emb = event.embedding
            cosine_sim = dot(claim_emb, event_emb) / (norm(claim_emb) * norm(event_emb))
            scores['semantic'] = max(0.0, float(cosine_sim))
        else:
            scores['semantic'] = 0.0

        # Signal 5: Spatial overlap (location mentions)
        # Check if claim text mentions event location
        claim_text_lower = claim.text.lower()
        location_mentions = 0
        for entity_id in event_entities:
            # Get entity from repo to check type and name
            entity = await self.entity_repo.get_by_id(entity_id)
            if entity and entity.entity_type in ['LOCATION', 'GPE']:
                if entity.name.lower() in claim_text_lower:
                    location_mentions += 1
        scores['spatial'] = min(1.0, location_mentions / 2.0)  # Normalize: 2+ locations = 1.0

        # Signal 6: Causal keywords
        causal_keywords = ['because', 'caused', 'led to', 'result', 'due to', 'triggered', 'sparked']
        causal_count = sum(1 for keyword in causal_keywords if keyword in claim_text_lower)
        scores['causal'] = min(1.0, causal_count / 2.0)  # Normalize

        # Weighted ensemble
        total_score = sum(scores[k] * self.SIGNAL_WEIGHTS[k] for k in scores)

        logger.debug(f"📊 Relatedness scores: {scores} → total={total_score:.3f}")

        return total_score, scores

    async def _classify_claim(
        self,
        event: Event,
        claim: Claim,
        existing_claims: List[Claim],
        sub_events: List[Event]
    ) -> ClaimDecision:
        """
        Decide how to handle a new claim using multi-signal scoring (issue #3)

        Logic:
        1. Check for duplicates (semantic similarity > 0.9)
        2. Check if UPDATE of existing metric
        3. Compute multi-signal relatedness score
        4. Classify based on score thresholds:
           - ≥0.7: ATTACH (high confidence)
           - ≥0.4: YIELD_SUBEVENT (medium, needs clustering)
           - ≥0.25: DELEGATE (try sibling events)
           - <0.25: REJECT (unrelated)
        """

        # 1. Check if claim is UPDATE of existing metric (not duplicate)
        update_info = await self._is_metric_update(claim, existing_claims)
        if update_info:
            metric_type, old_value, new_value = update_info
            logger.info(f"📈 Metric UPDATE: {metric_type} {old_value} → {new_value} (will update parent event)")
            return ClaimDecision.ADD  # Add to update the metric

        # 2. Check for duplicates
        for existing_claim in existing_claims:
            if self._calculate_claim_similarity(claim, existing_claim) > self.DUPLICATE_THRESHOLD:
                return ClaimDecision.MERGE

        # 3. Compute multi-signal relatedness score
        event_entities = await self._get_event_entities(event)
        relatedness_score, signal_scores = await self._compute_relatedness_score(event, claim, event_entities)

        logger.info(
            f"📊 Multi-signal score: {relatedness_score:.3f} "
            f"(entity={signal_scores['entity']:.2f}, temporal={signal_scores['temporal']:.2f}, "
            f"reference={signal_scores['reference']:.2f}, semantic={signal_scores['semantic']:.2f})"
        )

        # 4. Classify based on score thresholds
        if relatedness_score >= self.ATTACH_THRESHOLD:
            # High confidence - add directly
            logger.info(f"✅ ATTACH (score={relatedness_score:.3f} ≥ {self.ATTACH_THRESHOLD})")
            return ClaimDecision.ADD

        elif relatedness_score >= self.YIELD_THRESHOLD:
            # Medium confidence - yield for sub-event clustering
            logger.info(f"🌿 YIELD (score={relatedness_score:.3f} ≥ {self.YIELD_THRESHOLD})")
            return ClaimDecision.YIELD_SUBEVENT

        elif relatedness_score >= self.DELEGATE_THRESHOLD:
            # Low confidence - try sibling events first
            if sub_events:
                best_sub_event = await self._find_best_sub_event(claim, sub_events)
                if best_sub_event:
                    sub_match = await self._calculate_match_score(best_sub_event, claim)
                    if sub_match > relatedness_score and sub_match > 0.5:
                        logger.info(f"🔀 DELEGATE to sub-event (sub_score={sub_match:.3f} > {relatedness_score:.3f})")
                        return ClaimDecision.DELEGATE

            # No better sub-event found, still yield as new dimension
            logger.info(f"🌿 YIELD (no better sub-event, score={relatedness_score:.3f})")
            return ClaimDecision.YIELD_SUBEVENT

        else:
            # Very low score - reject
            logger.info(f"❌ REJECT (score={relatedness_score:.3f} < {self.DELEGATE_THRESHOLD})")
            return ClaimDecision.REJECT

    async def create_root_event(self, claims: List[Claim]) -> Event:
        """
        Create new root event from claims

        Args:
            claims: Initial claims for the event

        Returns:
            Created root event
        """
        logger.info(f"🌱 Creating root event from {len(claims)} claims")

        # Generate canonical name using LLM
        canonical_name = await self._generate_event_name(claims)

        # Determine event type using LLM
        event_type = await self._infer_event_type(claims)

        # Calculate temporal bounds
        event_start, event_end = self._calculate_temporal_bounds(claims)

        # Create temporary event for narrative generation (needs temporal bounds + name)
        temp_event = Event(
            id=uuid.uuid4(),
            canonical_name=canonical_name,
            event_type=event_type,
            parent_event_id=None,
            confidence=0.3,
            coherence=0.5,
            event_start=event_start,
            event_end=event_end,
            status='provisional'
        )

        # Calculate initial metrics
        await self._update_event_metrics(temp_event, claims)

        # Generate narrative from claims (BEFORE embedding generation)
        summary = await self._generate_event_narrative(temp_event, claims)
        logger.info(f"📝 Generated initial narrative for root event")

        # Generate event embedding from summary text (semantic representation)
        embedding = await self._generate_event_embedding(summary)
        if embedding:
            logger.info(f"📊 Generated event embedding from summary ({len(embedding)} dims)")

        # Create final event with summary and embedding
        event = Event(
            id=temp_event.id,
            canonical_name=canonical_name,
            event_type=event_type,
            parent_event_id=None,  # Root event
            confidence=temp_event.confidence,
            coherence=temp_event.coherence,
            event_start=event_start,
            event_end=event_end,
            status='provisional',
            summary=summary,
            embedding=embedding
        )

        # Store in repositories
        created_event = await self.event_repo.create(event)

        # Link all source claims to this event
        for claim in claims:
            await self.event_repo.link_claim(created_event, claim, relationship_type="SUPPORTS")
        logger.info(f"🔗 Linked {len(claims)} claims to root event")

        # Build graph structure: Event -> Entity relationships
        await self._link_event_to_entities(created_event, claims)

        logger.info(f"✨ Created root event: {canonical_name} ({created_event.id})")
        return created_event

    async def _create_sub_event(self, parent: Event, claims: List[Claim]) -> Event:
        """
        Create sub-event under parent

        Args:
            parent: Parent event
            claims: Claims for the sub-event

        Returns:
            Created sub-event
        """
        logger.info(f"🌿 Creating sub-event under {parent.canonical_name}")

        # Generate sub-event name
        canonical_name = await self._generate_event_name(claims, parent=parent)

        # Calculate temporal bounds
        event_start, event_end = self._calculate_temporal_bounds(claims)

        # Create temporary sub-event for narrative generation
        temp_sub_event = Event(
            id=uuid.uuid4(),
            canonical_name=canonical_name,
            event_type=parent.event_type,  # Inherit from parent
            parent_event_id=parent.id,
            confidence=0.5,  # Initial confidence
            coherence=0.5,
            event_start=event_start,
            event_end=event_end,
            status='provisional'
        )

        # Calculate metrics
        await self._update_event_metrics(temp_sub_event, claims)

        # Generate narrative from claims (BEFORE embedding generation)
        summary = await self._generate_event_narrative(temp_sub_event, claims)
        logger.info(f"📝 Generated narrative for sub-event")

        # Generate event embedding from summary text (semantic representation)
        embedding = await self._generate_event_embedding(summary)
        if embedding:
            logger.info(f"📊 Generated sub-event embedding from summary ({len(embedding)} dims)")

        # Create final sub-event with summary and embedding
        sub_event = Event(
            id=temp_sub_event.id,
            canonical_name=canonical_name,
            event_type=parent.event_type,  # Inherit from parent
            parent_event_id=parent.id,
            confidence=temp_sub_event.confidence,
            coherence=temp_sub_event.coherence,
            event_start=event_start,
            event_end=event_end,
            status='provisional',
            summary=summary,
            embedding=embedding
        )

        # Store in repositories
        created_sub_event = await self.event_repo.create(sub_event)

        # Create parent-child relationship in Neo4j (via repository)
        await self.event_repo.create_sub_event_relationship(
            parent_id=parent.id,
            child_id=sub_event.id
        )

        # Link all source claims to this sub-event
        for claim in claims:
            await self.event_repo.link_claim(created_sub_event, claim, relationship_type="SUPPORTS")
        logger.info(f"🔗 Linked {len(claims)} claims to sub-event")

        # Build graph structure: Event -> Entity relationships
        await self._link_event_to_entities(created_sub_event, claims)

        logger.info(f"✨ Created sub-event: {canonical_name} under {parent.canonical_name}")
        return created_sub_event

    # Helper methods

    def _calculate_claim_similarity(self, claim1: Claim, claim2: Claim) -> float:
        """
        Calculate semantic similarity between claims using embeddings

        Returns: 0.0 to 1.0
        """
        if not claim1.embedding or not claim2.embedding:
            return 0.0

        # Cosine similarity
        vec1 = np.array(claim1.embedding)
        vec2 = np.array(claim2.embedding)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def _generate_claim_embeddings(self, claims: List[Claim]) -> None:
        """
        Generate embeddings on-demand for claims that don't have them

        Mutates claims in-place by setting claim.embedding
        Uses batch generation for efficiency
        """
        # Find claims without embeddings
        claims_needing_embeddings = [c for c in claims if not c.embedding and c.text]

        if not claims_needing_embeddings:
            return

        logger.debug(f"Generating embeddings for {len(claims_needing_embeddings)} claims on-demand")

        # Batch generate embeddings (OpenAI supports up to 2048 texts per request)
        batch_size = 100
        for i in range(0, len(claims_needing_embeddings), batch_size):
            batch = claims_needing_embeddings[i:i+batch_size]
            texts = [c.text for c in batch]

            try:
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )

                # Assign embeddings back to claims
                for claim, embedding_data in zip(batch, response.data):
                    claim.embedding = embedding_data.embedding

            except Exception as e:
                logger.error(f"Failed to generate embeddings batch: {e}")
                # Set empty embeddings so we don't try again
                for claim in batch:
                    claim.embedding = [0.0] * 1536

    async def _generate_event_embedding(self, summary_text: str) -> Optional[List[float]]:
        """
        Generate embedding for event from its summary text

        Strategy: Create semantic embedding from event summary/narrative
        This captures the coherent meaning of what the event IS,
        rather than averaging individual claim embeddings.

        Similar to gen1 story matching - semantic similarity between
        "what is this event about?" enables proper matching.

        Args:
            summary_text: Event summary/narrative text

        Returns:
            Embedding vector or None if generation fails
        """
        if not summary_text:
            logger.warning("Cannot generate event embedding without summary text")
            return None

        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=summary_text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate event embedding: {e}")
            return None

    async def _get_event_claims(self, event: Event) -> List[Claim]:
        """
        Fetch all claims for an event from Neo4j graph relationships

        Note: Claims are stored in PostgreSQL, but event-claim relationships
        are in Neo4j. This queries the graph to get claim IDs, then hydrates
        from PostgreSQL if needed.
        """
        # Get claims from Neo4j graph (minimal data)
        claims = await self.event_repo.get_event_claims(event.id)

        # For now, we return the minimal claim data from Neo4j
        # If full claim data (embedding, metadata) is needed, hydrate from PostgreSQL:
        # for i, claim in enumerate(claims):
        #     full_claim = await self.claim_repo.get_by_id(claim.id)
        #     if full_claim:
        #         claims[i] = full_claim

        return claims

    async def _get_event_entities(self, event: Event) -> Set[uuid.UUID]:
        """Get all entity IDs associated with an event's claims"""
        claims = await self._get_event_claims(event)

        entity_ids = set()
        for claim in claims:
            entity_ids.update(claim.entity_ids)

        return entity_ids

    def _references_event(self, event: Event, claim: Claim, event_entities: Set[uuid.UUID]) -> bool:
        """
        Check if claim explicitly references the event by name or location

        Universal principle: Explicit reference signals relationship
        even when entity overlap is low (e.g., humanitarian response
        vs investigation - different actors but same event)

        Args:
            event: Event to check against
            claim: Claim to analyze
            event_entities: Pre-computed event entity IDs

        Returns:
            True if claim references event
        """
        claim_text_lower = claim.text.lower()

        # Extract key terms from event name
        # "Wang Fuk Court Fire" → ["wang fuk court", "wang fuk", "fire"]
        event_name_lower = event.canonical_name.lower()

        # Check full name
        if event_name_lower in claim_text_lower:
            logger.info(f"📍 Exact event name match: '{event.canonical_name}'")
            return True

        # Check location name (multi-word locations)
        # Extract location entities from event
        location_terms = []
        if event.location:
            location_terms.append(event.location.lower())

        # Also check for key identifiers in event name
        # e.g., "Wang Fuk Court" even without "Fire"
        event_keywords = [
            word.lower() for word in event.canonical_name.split()
            if len(word) >= 3 and word.lower() not in ['fire', 'the', 'and', 'incident']
        ]

        # Check for multi-word location phrases (at least 2 words)
        for i in range(len(event_keywords) - 1):
            two_word_phrase = f"{event_keywords[i]} {event_keywords[i+1]}"
            if two_word_phrase in claim_text_lower:
                logger.info(f"📍 Location phrase match: '{two_word_phrase}'")
                return True

        # Check for location terms
        for location in location_terms:
            if location in claim_text_lower:
                logger.info(f"📍 Location match: '{location}'")
                return True

        # Check for contextual references like "the fire", "the incident"
        contextual_refs = ["the fire", "the incident", "the blaze", "the disaster"]
        has_contextual_ref = any(ref in claim_text_lower for ref in contextual_refs)

        if has_contextual_ref:
            # Prior behavior: only accept if at least one shared entity.
            if len(set(claim.entity_ids) & event_entities) > 0:
                logger.info(f"📍 Contextual reference + shared entity")
                return True

            # New: allow generic references when temporally aligned (e.g., "the fire" within 48h of event start).
            if claim.event_time and event.event_start:
                days_from_start = abs((claim.event_time - event.event_start).total_seconds()) / 86400
                if days_from_start <= 2.0:
                    logger.info(f"📍 Contextual reference within temporal window ({days_from_start:.1f}d)")
                    return True

        return False

    async def _is_metric_update(self, claim: Claim, existing_claims: List[Claim]) -> Optional[tuple]:
        """
        Check if claim is an UPDATE of an existing metric (e.g., death toll increasing)

        Uses UpdateDetector for systematic pattern detection instead of hardcoded regexes.

        Args:
            claim: New claim to check
            existing_claims: Existing claims in the event

        Returns:
            (metric_type, old_value, new_value) if update detected, None otherwise
        """
        # Detect topic for new claim
        topic_key = self.update_detector.detect_topic_key(claim)
        if not topic_key:
            return None

        new_value = self.update_detector.extract_numeric_value(claim, topic_key)
        if new_value is None:
            return None

        # Check if any existing claim has same topic with different value
        for existing in existing_claims:
            existing_topic = self.update_detector.detect_topic_key(existing)
            if existing_topic == topic_key:
                old_value = self.update_detector.extract_numeric_value(existing, topic_key)
                if old_value is not None and old_value != new_value:
                    logger.debug(f"📈 Found metric update: {topic_key} {old_value} → {new_value}")
                    return (topic_key, old_value, new_value)

        return None

    async def _calculate_semantic_similarity(self, event: Event, claim: Claim) -> float:
        """
        Calculate semantic similarity between event and claim using embeddings

        Args:
            event: Event to compare against
            claim: Claim to analyze

        Returns:
            Cosine similarity score between 0 and 1
        """
        if not event.embedding or not claim.embedding:
            return 0.0

        import numpy as np
        vec1 = np.array(event.embedding)
        vec2 = np.array(claim.embedding)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            similarity = float(dot_product / (norm1 * norm2))
            # Normalize from [-1, 1] to [0, 1]
            return (similarity + 1) / 2

        return 0.0

    async def _generate_event_narrative(self, event: Event, claims: List[Claim]) -> str:
        """
        Generate narrative summary from event claims using LLM

        Args:
            event: Event to generate narrative for
            claims: All claims supporting this event

        Returns:
            Narrative summary (2-3 sentences)
        """
        if not claims:
            return f"{event.canonical_name} - No details available."

        # Prepare claims for LLM
        claim_texts = [f"- {claim.text}" for claim in claims[:10]]  # Limit to 10 for context
        claims_str = "\n".join(claim_texts)

        prompt = f"""Generate a concise narrative summary (2-3 sentences) for this event:

Event Name: {event.canonical_name}
Event Type: {event.event_type}

Claims:
{claims_str}

Write a clear, factual narrative that synthesizes these claims into a coherent story.
IMPORTANT: Include temporal information (WHEN things happened) if available in the claims.
Focus on: what happened, when it happened, and key impacts."""

        response = await self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.3
        )

        narrative = response.choices[0].message.content.strip()
        logger.debug(f"📖 Generated narrative: {narrative[:100]}...")
        return narrative

    async def _generate_parent_narrative(self, parent: Event, sub_events: List[Event]):
        """
        Generate parent narrative from sub-event narratives

        This creates a high-level story from all sub-events to check coherence.
        Orders sub-events chronologically to show event progression.

        Args:
            parent: Parent event
            sub_events: List of sub-events created
        """
        # Get all sub-events (including existing ones)
        all_sub_events = await self.event_repo.get_sub_events(parent.id)

        if not all_sub_events:
            return

        # Sort sub-events by earliest_time for chronological ordering
        # Events without time go last
        def get_sort_key(event):
            if event.event_start:
                start_py = neo4j_datetime_to_python(event.event_start)
                return (0, start_py) if start_py else (1, datetime.max)
            return (1, datetime.max)

        sorted_sub_events = sorted(all_sub_events, key=get_sort_key)

        # Collect sub-event narratives with temporal info
        sub_narratives = []
        for sub in sorted_sub_events:
            if sub.summary:
                # Add temporal context if available
                time_str = ""
                if sub.event_start:
                    start_py = neo4j_datetime_to_python(sub.event_start)
                    if start_py:
                        time_str = f" (Around {start_py.strftime('%B %d, %Y')})"

                sub_narratives.append(f"**{sub.canonical_name}**{time_str}: {sub.summary}")

        if not sub_narratives:
            logger.debug("No sub-event narratives to synthesize")
            return

        narratives_str = "\n\n".join(sub_narratives)

        # Get parent event's actual start date for the prompt
        parent_start_str = "Unknown"
        if parent.event_start:
            parent_start_py = neo4j_datetime_to_python(parent.event_start)
            if parent_start_py:
                parent_start_str = parent_start_py.strftime('%B %d, %Y')

        prompt = f"""Synthesize a high-level narrative (2-3 paragraphs) for this complex event from its sub-events:

Event: {parent.canonical_name}
Type: {parent.event_type}
Event Start Date: {parent_start_str}

Sub-Events (in chronological order):
{narratives_str}

Create a coherent story that:
1. Follows the CHRONOLOGICAL PROGRESSION - start with what happened first (the incident/breakout)
2. Shows how sub-events relate to each other temporally
3. Emphasizes WHEN things happened (use temporal markers: "initially", "following this", "in the aftermath", etc.)
4. Identifies the key phases/dimensions and their sequence
5. USE THE EVENT START DATE PROVIDED ABOVE as the date when the main incident occurred

Write in clear, journalistic style with strong temporal flow. Do not invent dates - use the Event Start Date provided."""

        response = await self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.3
        )

        parent_narrative = response.choices[0].message.content.strip()
        parent.summary = parent_narrative

        logger.info(f"📚 Generated parent narrative for {parent.canonical_name}")
        logger.debug(f"   Narrative: {parent_narrative[:150]}...")

    async def _propagate_from_children(
        self,
        parent: Event,
        children: List[Event]
    ) -> None:
        """
        Propagate metadata from sub-events upward to parent

        Updates:
        - event_end: max of all descendant end times
        - coherence: measure of clustering quality (tight vs dispersed)
        - confidence: boost if children are high-confidence
        - canonical_name: enrich if sub-events reveal new dimensions

        Args:
            parent: Parent event to update
            children: New sub-events created
        """
        if not children:
            return

        logger.debug(f"⬆️  Propagating metadata from {len(children)} children to {parent.canonical_name}")

        # 1. Update event_end = max(all descendant end times)
        max_end = parent.event_end
        for child in children:
            if child.event_end:
                child_end_py = neo4j_datetime_to_python(child.event_end)
                max_end_py = neo4j_datetime_to_python(max_end) if max_end else None

                if child_end_py and (not max_end_py or child_end_py > max_end_py):
                    max_end = child.event_end

        if max_end != parent.event_end:
            logger.info(f"⬆️  Updated parent event_end: {parent.event_end} → {max_end}")
            parent.event_end = max_end

        # 2. Calculate coherence from clustering quality
        # High coherence = children are similar (tight cluster)
        # Low coherence = children are diverse (dispersed cluster)
        if len(children) > 1:
            # Use child embeddings to measure dispersion
            child_embeddings = [c.embedding for c in children if c.embedding]
            if len(child_embeddings) >= 2:
                embeddings_array = np.array(child_embeddings)

                # Calculate pairwise similarities
                similarities = []
                for i in range(len(child_embeddings)):
                    for j in range(i+1, len(child_embeddings)):
                        vec1 = embeddings_array[i]
                        vec2 = embeddings_array[j]
                        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        similarities.append(sim)

                avg_similarity = float(np.mean(similarities))

                # High similarity = high coherence (children are tightly related)
                # Update parent coherence as weighted average
                new_coherence = 0.7 * parent.coherence + 0.3 * avg_similarity

                logger.info(f"⬆️  Updated parent coherence: {parent.coherence:.2f} → {new_coherence:.2f} (child similarity: {avg_similarity:.2f})")
                parent.coherence = new_coherence

        # 3. Boost confidence if children are high-confidence
        child_confidences = [c.confidence for c in children]
        if child_confidences:
            avg_child_confidence = sum(child_confidences) / len(child_confidences)
            if avg_child_confidence > parent.confidence:
                new_confidence = 0.8 * parent.confidence + 0.2 * avg_child_confidence
                logger.info(f"⬆️  Boosted parent confidence: {parent.confidence:.2f} → {new_confidence:.2f}")
                parent.confidence = new_confidence

        # 4. Update parent in repository
        await self.event_repo.update(parent)

    async def _cluster_claims_by_theme(
        self,
        parent_event: Event,
        claims: List[Claim]
    ) -> List[dict]:
        """
        Cluster claims by semantic theme using embeddings + LLM

        Universal clustering approach:
        1. Use embedding similarity to find natural groupings
        2. Use LLM to name each cluster theme
        3. Return clusters with theme name and claims

        Args:
            parent_event: Parent event for context
            claims: Claims to cluster

        Returns:
            List of dicts: [{'theme': 'Investigation Phase', 'claims': [...]}, ...]
        """
        if len(claims) <= 1:
            return [{'theme': 'Sub-event', 'claims': claims}]

        # Build similarity matrix from embeddings
        embeddings = []
        valid_claims = []
        for claim in claims:
            if claim.embedding:
                embeddings.append(np.array(claim.embedding))
                valid_claims.append(claim)

        if len(valid_claims) <= 1:
            # No embeddings available, treat as single cluster
            return [{'theme': 'Related Events', 'claims': claims}]

        embeddings = np.array(embeddings)

        # Simple hierarchical clustering using cosine similarity
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # Compute pairwise cosine distances
        distances = pdist(embeddings, metric='cosine')

        # Hierarchical clustering
        linkage_matrix = linkage(distances, method='average')

        # Cut tree using distance threshold (let natural clusters emerge)
        # Distance threshold of 0.3 in cosine distance space
        # (closer = more similar, let semantically similar claims stay together)
        distance_threshold = 0.3
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

        logger.info(f"📊 Natural clustering: {len(valid_claims)} claims → {len(set(cluster_labels))} clusters (distance threshold={distance_threshold})")

        # Group claims by cluster
        clusters = {}
        for claim, label in zip(valid_claims, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(claim)

        # Use LLM to name each cluster
        result_clusters = []
        for label, cluster_claims in clusters.items():
            theme = await self._name_cluster_theme(parent_event, cluster_claims)
            result_clusters.append({
                'theme': theme,
                'claims': cluster_claims
            })

        logger.info(f"📊 Clustered {len(valid_claims)} claims into {len(result_clusters)} themes: {[c['theme'] for c in result_clusters]}")
        return result_clusters

    async def _name_cluster_theme(
        self,
        parent_event: Event,
        claims: List[Claim]
    ) -> str:
        """
        Use LLM to identify the theme of a cluster of claims

        Args:
            parent_event: Parent event for context
            claims: Claims in the cluster

        Returns:
            Theme name (e.g., 'Investigation Phase', 'Safety Findings')
        """
        claim_texts = [c.text[:200] for c in claims[:5]]  # Sample first 5 claims

        prompt = f"""You are analyzing a cluster of related claims about an event.

Parent Event: {parent_event.canonical_name} ({parent_event.event_type})

Claims in this cluster:
{chr(10).join(f"- {text}" for text in claim_texts)}

What is the common theme or aspect that unifies these claims?

Consider:
- Investigation/legal proceedings
- Safety findings/root cause analysis
- Regulatory response/consequences
- Political/social dimensions
- Temporal phases (before, during, after)

Provide a concise theme name (2-5 words) that captures the essence of this cluster.
Respond with ONLY the theme name."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )

            theme = response.choices[0].message.content.strip()
            logger.debug(f"🏷️  Cluster theme: {theme}")
            return theme

        except Exception as e:
            logger.warning(f"Failed to name cluster theme with LLM: {e}")
            return "Related Events"

    async def _is_temporal_aftermath(
        self,
        event: Event,
        claim: Claim,
        existing_claims: List[Claim]
    ) -> bool:
        """
        Detect if claim is temporal aftermath/investigation of the event

        Key insight from human analysis:
        - Investigation/aftermath happens AFTER the incident (3-7 days typical)
        - Different entities (investigators vs firefighters)
        - But semantically references the original event

        Example:
        - Fire incident: Nov 26 (firefighters, casualties)
        - Investigation: Nov 29-Dec 2 (officials, safety inspectors)
        - Investigation references "the fire" even with different actors

        Returns: True if this is aftermath/investigation phase
        """
        # 1. Temporal check: Claim happens AFTER event ended
        if not claim.event_time or not event.event_end:
            return False

        # Defensive conversion for datetime arithmetic
        event_end_py = neo4j_datetime_to_python(event.event_end)
        claim_time_py = neo4j_datetime_to_python(claim.event_time)

        if not event_end_py or not claim_time_py:
            logger.warning(f"Could not convert datetimes: event_end={type(event.event_end)}, claim_time={type(claim.event_time)}")
            return False

        # Aftermath typically 2-14 days after incident
        days_after = (claim_time_py - event_end_py).days
        if days_after < 2 or days_after > 14:
            return False

        logger.debug(f"Claim is {days_after} days after event - checking semantic reference...")

        # 2. Semantic check: Does claim reference the event?
        # Use LLM to detect aftermath language
        prompt = f"""Analyze if this claim discusses the aftermath or investigation of an event.

Event: {event.canonical_name} ({event.event_type})
Event occurred: {neo4j_datetime_to_python(event.event_start).strftime('%B %d, %Y') if event.event_start and neo4j_datetime_to_python(event.event_start) else 'unknown'}

Existing claims about the event (sample):
{chr(10).join([f"- {c.text[:100]}" for c in existing_claims[:3]])}

New claim ({neo4j_datetime_to_python(claim.event_time).strftime('%B %d, %Y') if claim.event_time and neo4j_datetime_to_python(claim.event_time) else 'unknown'}):
"{claim.text}"

Question: Is this new claim discussing the aftermath, investigation, or consequences of the event described above?

Consider:
- Does it mention investigation, accountability, officials, safety reviews?
- Does it reference "the {event.event_type.lower()}" or similar?
- Does it discuss responses, failures, or consequences?

Answer ONLY "yes" or "no"."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=5
            )

            answer = response.choices[0].message.content.strip().lower()
            is_aftermath = "yes" in answer

            logger.info(f"Temporal aftermath check: {is_aftermath} (claim: '{claim.text[:80]}...')")
            return is_aftermath

        except Exception as e:
            logger.warning(f"LLM aftermath check failed: {e}")
            return False

    async def _calculate_coherence_with_claim(
        self,
        event: Event,
        claim: Claim,
        existing_claims: List[Claim]
    ) -> float:
        """
        Calculate how well a new claim fits with existing claims

        Returns: 0.0 to 1.0
        """
        if not existing_claims or not claim.embedding:
            return 0.5  # Neutral

        # Average similarity with existing claims
        similarities = []
        for existing_claim in existing_claims:
            sim = self._calculate_claim_similarity(claim, existing_claim)
            similarities.append(sim)

        return float(np.mean(similarities))

    async def _find_best_sub_event(
        self,
        claim: Claim,
        sub_events: List[Event]
    ) -> Optional[Event]:
        """
        Find the best sub-event to delegate a claim to

        Returns: Best matching sub-event or None
        """
        if not sub_events:
            return None

        best_event = None
        best_score = 0.0

        for sub_event in sub_events:
            score = await self._calculate_match_score(sub_event, claim)
            if score > best_score:
                best_score = score
                best_event = sub_event

        return best_event if best_score > 0.5 else None

    async def _calculate_match_score(self, event: Event, claim: Claim) -> float:
        """
        Calculate how well a claim matches an event

        Returns: 0.0 to 1.0
        """
        # Entity overlap
        event_entities = await self._get_event_entities(event)
        claim_entities = set(claim.entity_ids)

        if not event_entities or not claim_entities:
            entity_overlap = 0.0
        else:
            entity_overlap = len(event_entities & claim_entities) / max(len(event_entities), len(claim_entities))

        # Semantic similarity (if event has embedding)
        semantic_sim = 0.0
        if event.embedding and claim.embedding:
            vec_event = np.array(event.embedding)
            vec_claim = np.array(claim.embedding)
            dot_product = np.dot(vec_event, vec_claim)
            norm_event = np.linalg.norm(vec_event)
            norm_claim = np.linalg.norm(vec_claim)
            if norm_event > 0 and norm_claim > 0:
                semantic_sim = dot_product / (norm_event * norm_claim)

        # Combined score
        return 0.6 * entity_overlap + 0.4 * semantic_sim

    async def _merge_duplicate(
        self,
        event: Event,
        claim: Claim,
        existing_claims: List[Claim]
    ):
        """
        Handle duplicate claim (corroborate existing claim)

        For now, just log it. Can enhance with confidence boosting later.
        """
        # Find most similar claim
        best_match = None
        best_similarity = 0.0

        for existing_claim in existing_claims:
            sim = self._calculate_claim_similarity(claim, existing_claim)
            if sim > best_similarity:
                best_similarity = sim
                best_match = existing_claim

        logger.debug(f"    Duplicate of: {best_match.text[:50] if best_match else 'unknown'}... (similarity: {best_similarity:.2f})")

        # TODO: Boost confidence of existing claim
        # TODO: Track corroboration in metadata

    async def _update_event_metrics(self, event: Event, claims: List[Claim]):
        """
        Update event confidence and coherence based on claims

        Confidence: Based on number and quality of claims
        Coherence: Average similarity between all claim pairs
        """
        if not claims:
            return

        # Update claim count
        event.claims_count = len(claims)

        # Calculate confidence (increases with more high-quality claims)
        avg_claim_confidence = np.mean([c.confidence for c in claims])
        claim_count_factor = min(len(claims) / 10.0, 1.0)  # Saturates at 10 claims
        event.confidence = 0.3 + 0.4 * claim_count_factor + 0.3 * avg_claim_confidence
        event.confidence = min(event.confidence, 1.0)

        # Calculate coherence (pairwise similarity)
        if len(claims) >= 2:
            similarities = []
            claims_with_embeddings = [c for c in claims if c.embedding]

            for i, claim1 in enumerate(claims_with_embeddings):
                for claim2 in claims_with_embeddings[i+1:]:
                    sim = self._calculate_claim_similarity(claim1, claim2)
                    similarities.append(sim)

            if similarities:
                event.coherence = float(np.mean(similarities))

        # Update status based on confidence and claim count
        event.update_status()

        logger.debug(f"  📊 Updated metrics: confidence={event.confidence:.2f}, coherence={event.coherence:.2f}, status={event.status}")

    def _calculate_temporal_bounds(self, claims: List[Claim]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Calculate earliest and latest event times from claims"""
        event_times = [c.event_time for c in claims if c.event_time]

        if not event_times:
            return None, None

        return min(event_times), max(event_times)

    async def _generate_event_name(self, claims: List[Claim], parent: Optional[Event] = None) -> str:
        """
        Generate canonical event name using LLM

        Args:
            claims: Claims to generate name from
            parent: Parent event (for sub-events)

        Returns:
            Canonical event name (e.g., "2025 Hong Kong Tai Po Fire")
        """
        if not claims:
            return "Unnamed Event"

        # Prepare claim texts (limit to avoid token overflow)
        claim_texts = [c.text for c in claims[:20]]  # Max 20 claims

        # Get entities for context
        all_entities = set()
        entity_names = []
        for claim in claims[:10]:
            if claim.entities:
                for entity in claim.entities:
                    if entity.canonical_name not in all_entities:
                        all_entities.add(entity.canonical_name)
                        entity_names.append(f"{entity.canonical_name} ({entity.entity_type})")

        # Get temporal context
        temporal_info = ""
        if claims and claims[0].event_time:
            event_time = claims[0].event_time
            temporal_info = f"\nTemporal context: {event_time.strftime('%B %d, %Y')}"

        if parent:
            # Sub-event naming
            prompt = f"""You are a news analyst identifying a distinct aspect or phase within a larger event: "{parent.canonical_name}"

Claims about this aspect:
{chr(10).join(f"- {text}" for text in claim_texts[:10])}

Key entities: {', '.join(entity_names[:10]) if entity_names else 'None'}{temporal_info}

What is the most natural, concise name for this distinct aspect? Think about:
- What makes this different from the main event?
- What is the central focus of these claims?
- How would journalists refer to this aspect?

Examples of good sub-event names: "Investigation and Arrests", "Public Response and Condolences", "Rescue Operations", "Aftermath and Rebuilding"

Respond with ONLY the canonical name (3-7 words), no explanation."""

        else:
            # Root event naming
            prompt = f"""You are a news analyst naming an event. Generate the most natural canonical name that journalists and historians would use.

Claims about this event:
{chr(10).join(f"- {text}" for text in claim_texts)}

Key entities: {', '.join(entity_names) if entity_names else 'None'}{temporal_info}

What is the most natural, memorable canonical name for this event? Consider:
- What would people call this event in conversation?
- What would a Wikipedia article title be?
- What captures the essence without being too verbose?
- Should the year be included, or is the event self-identifying?

Examples of good event names:
- "Grenfell Tower Fire" (location + type)
- "January 6 Capitol Attack" (date + place + type)
- "Fukushima Disaster" (location alone when unambiguous)
- "2011 Tōhoku Earthquake and Tsunami" (year + location when needed)

Respond with ONLY the canonical name, no explanation."""

        try:
            logger.debug(f"📝 Event naming prompt:\n{prompt}")

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )

            name = response.choices[0].message.content.strip()
            # Remove quotes if LLM wrapped the name
            name = name.strip('"\'')

            logger.info(f"🏷️  Generated event name: {name}")
            return name

        except Exception as e:
            logger.error(f"Failed to generate event name with LLM: {e}")
            # Fallback: Use entity-based naming
            if entity_names:
                return f"Event involving {entity_names[0].split('(')[0].strip()}"
            return f"Event: {claims[0].text[:60]}..."

    async def _link_event_to_entities(self, event: Event, claims: List[Claim]):
        """
        Create Event->INVOLVES->Entity relationships in Neo4j

        Args:
            event: Event to link
            claims: Claims that belong to this event
        """
        # Collect all unique entity IDs from claims
        entity_ids = set()
        for claim in claims:
            entity_ids.update(claim.entity_ids)

        if not entity_ids:
            logger.debug(f"No entities to link for event {event.canonical_name}")
            return

        # Create INVOLVES relationships in Neo4j (via repository)
        await self.event_repo.link_event_to_entities(event.id, entity_ids)

        logger.info(f"🔗 Linked event {event.canonical_name} to {len(entity_ids)} entities")

    async def _infer_event_type(self, claims: List[Claim]) -> str:
        """
        Infer event type from claims using LLM

        Returns: Event type (FIRE, PROTEST, SHOOTING, ELECTION, etc.)
        """
        if not claims:
            return "INCIDENT"

        # Sample claim texts
        claim_texts = [c.text for c in claims[:10]]

        prompt = f"""You are analyzing a news event. Based on these claims, what is the primary event type?

Claims:
{chr(10).join(f"- {text}" for text in claim_texts)}

Consider the core nature of what happened:
- FIRE: Building fires, wildfires, explosions
- SHOOTING: Gun violence, mass shootings
- PROTEST: Demonstrations, rallies, civil unrest
- ELECTION: Elections, voting, political campaigns
- CONFLICT: Wars, military operations, armed conflicts
- DISASTER: Natural disasters, earthquakes, floods, hurricanes
- ACCIDENT: Transportation accidents, industrial accidents
- SUMMIT: International meetings, conferences, diplomatic events
- LEGAL: Court cases, trials, legal proceedings
- INCIDENT: General incidents (default if none of the above fit)

What is the single best classification? Respond with ONLY one word from the list above."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )

            event_type = response.choices[0].message.content.strip().upper()

            # Validate
            valid_types = ["FIRE", "SHOOTING", "PROTEST", "ELECTION", "CONFLICT",
                          "DISASTER", "ACCIDENT", "SUMMIT", "LEGAL", "INCIDENT"]
            if event_type not in valid_types:
                logger.warning(f"LLM returned invalid event type: {event_type}, defaulting to INCIDENT")
                event_type = "INCIDENT"

            logger.info(f"📋 Classified event type: {event_type}")
            return event_type

        except Exception as e:
            logger.error(f"Failed to classify event type with LLM: {e}")
            return "INCIDENT"
