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

        # Thresholds (from RECURSIVE_EVENT_DESIGN.md)
        self.DUPLICATE_THRESHOLD = 0.9       # Semantic similarity for duplicates
        self.COHERENCE_THRESHOLD = 0.6       # Min coherence to add claim
        self.ENTITY_OVERLAP_THRESHOLD = 0.5  # Entity overlap for topic match
        self.SUB_EVENT_THRESHOLD = 0.3       # Min overlap for yielding sub-event

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

        # Fetch existing claims for this event (for duplicate detection)
        existing_claims = await self._get_event_claims(event)

        # Load sub-events (for delegation)
        sub_events = await self.event_repo.get_sub_events(event.id) if hasattr(self.event_repo, 'get_sub_events') else []

        for claim in new_claims:
            decision = await self._classify_claim(event, claim, existing_claims, sub_events)

            if decision == ClaimDecision.MERGE:
                # Duplicate of existing claim → corroborate
                logger.debug(f"  ✓ MERGE: {claim.text[:50]}...")
                await self._merge_duplicate(event, claim, existing_claims)
                claims_added.append(claim)

            elif decision == ClaimDecision.ADD:
                # Novel but fits this event's topic
                logger.debug(f"  ✓ ADD: {claim.text[:50]}...")
                event.claim_ids.append(claim.id)
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
                # Novel aspect → create sub-event
                logger.debug(f"  ⚡ YIELD_SUBEVENT: {claim.text[:50]}...")
                sub_event = await self._create_sub_event(event, [claim])
                sub_events_created.append(sub_event)
                claims_added.append(claim)

            elif decision == ClaimDecision.REJECT:
                # Doesn't belong here
                logger.debug(f"  ✗ REJECT: {claim.text[:50]}...")
                claims_rejected.append(claim)

        # Update event metrics if claims were added
        if claims_added:
            await self._update_event_metrics(event, existing_claims + claims_added)
            await self.event_repo.update(event)

        logger.info(f"📊 Result: {len(claims_added)} added, {len(sub_events_created)} sub-events, {len(claims_rejected)} rejected")

        return ExaminationResult(
            claims_added=claims_added,
            sub_events_created=sub_events_created,
            claims_rejected=claims_rejected
        )

    async def _classify_claim(
        self,
        event: Event,
        claim: Claim,
        existing_claims: List[Claim],
        sub_events: List[Event]
    ) -> ClaimDecision:
        """
        Decide how to handle a new claim

        Logic (temporal-aware):
        1. Check for duplicates (semantic similarity > 0.9)
        2. Check if temporal aftermath/investigation (happens AFTER event + references it)
        3. Check if it fits my topic (entity overlap > 0.5, coherence > 0.6)
        4. Check if sub-event handles it better
        5. Check if novel aspect (entity overlap 0.3-0.5)
        6. Otherwise reject
        """

        # 1. Check for duplicates
        for existing_claim in existing_claims:
            if self._calculate_claim_similarity(claim, existing_claim) > self.DUPLICATE_THRESHOLD:
                return ClaimDecision.MERGE

        # 2. NEW: Check if this is temporal aftermath/investigation
        # Key insight: Investigation/aftermath happens AFTER the incident but references it
        if await self._is_temporal_aftermath(event, claim, existing_claims):
            logger.info(f"🕐 Claim is temporal aftermath of {event.canonical_name}")
            return ClaimDecision.YIELD_SUBEVENT

        # Get entities for overlap calculation
        event_entities = await self._get_event_entities(event)
        claim_entities = set(claim.entity_ids)

        if not event_entities or not claim_entities:
            # No entities to compare - use semantic similarity as fallback
            entity_overlap = 0.0
        else:
            entity_overlap = len(event_entities & claim_entities) / max(len(event_entities), len(claim_entities))

        # 3. Check if it fits my topic
        if entity_overlap > self.ENTITY_OVERLAP_THRESHOLD:
            # Check semantic coherence with existing claims
            coherence = await self._calculate_coherence_with_claim(event, claim, existing_claims)
            if coherence > self.COHERENCE_THRESHOLD:
                return ClaimDecision.ADD

        # 4. Check if sub-event handles it better
        if sub_events:
            best_sub_event = await self._find_best_sub_event(claim, sub_events)
            if best_sub_event:
                sub_match = await self._calculate_match_score(best_sub_event, claim)
                my_match = entity_overlap  # Simplified match score
                if sub_match > my_match and sub_match > 0.5:
                    return ClaimDecision.DELEGATE

        # 5. Is it novel but related? (shares context but different aspect)
        if entity_overlap > self.SUB_EVENT_THRESHOLD and entity_overlap < self.ENTITY_OVERLAP_THRESHOLD:
            # Related but distinct aspect
            return ClaimDecision.YIELD_SUBEVENT

        # 6. Unrelated
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

        # Generate event embedding
        embedding = await self._generate_event_embedding(claims)

        # Create event
        event = Event(
            id=uuid.uuid4(),
            canonical_name=canonical_name,
            event_type=event_type,
            parent_event_id=None,  # Root event
            claim_ids=[c.id for c in claims],
            confidence=0.3,  # Initial confidence
            coherence=0.5,   # Initial coherence
            event_start=event_start,
            event_end=event_end,
            status='provisional',
            embedding=embedding
        )

        # Calculate initial metrics
        await self._update_event_metrics(event, claims)

        # Store in repositories
        created_event = await self.event_repo.create(event)

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

        # Generate embedding
        embedding = await self._generate_event_embedding(claims)

        # Create sub-event
        sub_event = Event(
            id=uuid.uuid4(),
            canonical_name=canonical_name,
            event_type=parent.event_type,  # Inherit from parent
            parent_event_id=parent.id,
            claim_ids=[c.id for c in claims],
            confidence=0.5,  # Initial confidence
            coherence=0.5,
            event_start=event_start,
            event_end=event_end,
            status='provisional',
            embedding=embedding
        )

        # Calculate metrics
        await self._update_event_metrics(sub_event, claims)

        # Store in repositories
        created_sub_event = await self.event_repo.create(sub_event)

        # Create parent-child relationship in Neo4j
        await self.event_repo.neo4j.create_event_relationship(
            parent_id=str(parent.id),
            child_id=str(sub_event.id),
            relationship_type="CONTAINS"
        )

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

    async def _generate_event_embedding(self, claims: List[Claim]) -> Optional[List[float]]:
        """
        Generate embedding for event from its claims

        Strategy: Average claim embeddings (simple but effective)
        Can be enhanced with LLM-based generation later
        """
        claim_embeddings = [c.embedding for c in claims if c.embedding]

        if not claim_embeddings:
            return None

        # Average embeddings
        avg_embedding = np.mean(claim_embeddings, axis=0)
        return avg_embedding.tolist()

    async def _get_event_claims(self, event: Event) -> List[Claim]:
        """Fetch all claims for an event"""
        if not event.claim_ids:
            return []

        claims = []
        for claim_id in event.claim_ids:
            claim = await self.claim_repo.get_by_id(claim_id)
            if claim:
                claims.append(claim)

        return claims

    async def _get_event_entities(self, event: Event) -> Set[uuid.UUID]:
        """Get all entity IDs associated with an event's claims"""
        claims = await self._get_event_claims(event)

        entity_ids = set()
        for claim in claims:
            entity_ids.update(claim.entity_ids)

        return entity_ids

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

        # Create INVOLVES relationships in Neo4j
        for entity_id in entity_ids:
            await self.event_repo.neo4j._execute_write("""
                MATCH (e:Event {id: $event_id})
                MATCH (entity:Entity {id: $entity_id})
                MERGE (e)-[r:INVOLVES]->(entity)
                ON CREATE SET r.created_at = datetime()
            """, {
                'event_id': str(event.id),
                'entity_id': str(entity_id)
            })

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
