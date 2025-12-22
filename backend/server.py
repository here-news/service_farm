#!/usr/bin/env python3
"""
Mini Demo Server for Fractal Event System

A standalone FastAPI server that demonstrates:
- Events emerging from streaming claims
- Live updates via SSE
- Event detail pages with narratives
- Slow-motion replay for visualization

Run:
    docker exec herenews-app python /app/test_eu/demo/server.py
"""

import os
import json
import asyncio
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
import httpx
import psycopg2
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

SIM_THRESHOLD = 0.70
LLM_THRESHOLD = 0.55
EVENT_MERGE_THRESHOLD = 0.60
RELATIONSHIP_THRESHOLD = 0.45  # Threshold for checking relationships between peer EUs


class RelationType(str, Enum):
    SAME = "same"              # Should merge (high similarity)
    CONTAINS = "contains"      # A contains B as sub-aspect
    CONTAINED_BY = "contained_by"  # B contains A
    SIBLING = "sibling"        # Both aspects of larger topic
    CAUSES = "causes"          # A caused B
    RELATES = "relates"        # Associated but distinct


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"


# ============== Data Structures ==============

class EventState(str, Enum):
    LIVE = "LIVE"
    WARM = "WARM"
    STABLE = "STABLE"
    DORMANT = "DORMANT"


@dataclass
class StreamClaim:
    id: str
    text: str
    page_id: str
    embedding: List[float]
    source_name: str = ""


@dataclass
class EmergentEU:
    id: str
    level: int
    embedding: List[float]
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    state: EventState = EventState.LIVE
    narrative: Optional[str] = None
    canonical_name: Optional[str] = None  # Stable name, converges over time
    byline: Optional[str] = None          # Current state, updates with new info
    headline: Optional[str] = None        # Legacy, maps to canonical_name

    # Metrics
    internal_corr: int = 0
    internal_contra: int = 0

    def size(self) -> int:
        return len(self.claim_ids)

    def mass(self) -> float:
        return self.size() * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * len(self.page_ids))

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def tension(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_contra / total if total > 0 else 0.0

    def update_embedding(self, new_embedding: List[float]):
        n = len(self.claim_ids)
        if n == 1:
            self.embedding = new_embedding
        else:
            self.embedding = [
                (self.embedding[i] * (n - 1) + new_embedding[i]) / n
                for i in range(len(new_embedding))
            ]

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'level': self.level,
            'size': self.size(),
            'mass': round(self.mass(), 2),
            'coherence': round(self.coherence(), 2),
            'tension': round(self.tension(), 2),
            'state': self.state.value,
            'canonical_name': self.canonical_name or f"Event {self.id}",
            'byline': self.byline or "",
            'headline': self.canonical_name or f"Event {self.id}",  # Legacy compatibility
            'narrative': self.narrative,
            'claim_count': len(self.claim_ids),
            'source_count': len(self.page_ids),
            'sub_event_count': len(self.children),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'sample_claims': self.texts[:3] if self.texts else []
        }


@dataclass
class EntityOrganism:
    """
    Entity as Epistemic Organism - parallel to Events.

    Entities (People, Organizations, Locations) are organisms that:
    - Consume claims that mention them (food)
    - Grow across multiple events (symbiosis)
    - Can become qualia-capable at threshold
    - Have their own lifecycle (warm → dormant → revivable)
    """
    id: str
    name: str
    entity_type: EntityType
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)  # Claim texts mentioning this entity
    event_ids: Set[str] = field(default_factory=set)  # Events this entity appears in
    page_ids: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    state: EventState = EventState.LIVE
    profile: Optional[str] = None  # Generated profile/description

    # Metrics
    mention_count: int = 0

    def mass(self) -> float:
        """Entity mass based on mentions and event spread"""
        return self.mention_count * 0.1 * (1 + 0.2 * len(self.event_ids))

    def is_qualia_capable(self) -> bool:
        """
        Entity becomes qualia-capable when:
        - > 10 mentions (enough substance)
        - > 3 events (cross-event presence)
        - > 2 sources (multiple perspectives)
        """
        return (
            self.mention_count > 10 and
            len(self.event_ids) > 3 and
            len(self.page_ids) > 2
        )

    def feed(self, claim_id: str, claim_text: str, event_id: str, page_id: str):
        """Entity absorbs a claim (food sharing)"""
        self.claim_ids.append(claim_id)
        self.texts.append(claim_text)
        self.event_ids.add(event_id)
        self.page_ids.add(page_id)
        self.mention_count += 1
        self.last_activity = datetime.now()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'entity_type': self.entity_type.value,
            'mention_count': self.mention_count,
            'mass': round(self.mass(), 2),
            'event_count': len(self.event_ids),
            'source_count': len(self.page_ids),
            'claim_count': len(self.claim_ids),
            'is_qualia_capable': self.is_qualia_capable(),
            'state': self.state.value,
            'profile': self.profile,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'sample_claims': self.texts[:5] if self.texts else [],
            'event_ids': list(self.event_ids)[:10]  # Top 10 events
        }


# ============== Demo Engine ==============

class DemoEngine:
    """
    Manages the demo state and streaming.

    UNIFIED HIERARCHY:
    - All EUs live in a single collection (self.eus)
    - Level determines position in hierarchy:
      - Level 1: Sub-events (claim clusters)
      - Level 2: Events (sub-event clusters)
      - Level 3+: Frames (event clusters, can nest infinitely)
    - parent_id links children to parents at any level
    - This is the fractal design: same structure at all levels
    """

    def __init__(self):
        # Single unified collection - frames are just level 3+ EUs
        self.eus: Dict[str, EmergentEU] = {}

        # Entity organisms - parallel to events
        self.entities: Dict[str, EntityOrganism] = {}
        self.entity_name_to_id: Dict[str, str] = {}  # normalized name -> entity id

        # Claim routing cache
        self.claim_to_eu: Dict[str, str] = {}  # claim_id -> EU id

        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.claims: List[StreamClaim] = []
        self.current_index = 0
        self.is_streaming = False
        self.stream_delay = 1.0  # seconds between claims

        self.stats = {
            'claims_processed': 0,
            'eus_by_level': {1: 0, 2: 0, 3: 0},  # Track EU counts by level
            'entities_created': 0,
            'entity_feeds': 0,  # Total entity food absorptions
        }

    def get_eus_by_level(self, level: int) -> List[EmergentEU]:
        """Get all EUs at a specific level"""
        return [eu for eu in self.eus.values() if eu.level == level]

    @property
    def sub_events(self) -> Dict[str, EmergentEU]:
        """Compatibility: get level 1 EUs"""
        return {k: v for k, v in self.eus.items() if v.level == 1}

    @property
    def events(self) -> Dict[str, EmergentEU]:
        """Get level 2+ EUs (user-facing events)"""
        return {k: v for k, v in self.eus.items() if v.level >= 2}

    async def load_claims(self):
        """Load claims and embeddings from database"""
        # Load graph snapshot
        import sys
        sys.path.insert(0, '/app/test_eu')
        from load_graph import load_snapshot

        snapshot = load_snapshot()

        # Load embeddings
        conn = psycopg2.connect(
            host=PG_HOST, database=PG_DB,
            user=PG_USER, password=PG_PASS
        )
        cur = conn.cursor()
        cur.execute("SELECT claim_id, embedding FROM core.claim_embeddings")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        embeddings = {}
        for claim_id, emb in rows:
            if isinstance(emb, str):
                emb = [float(x) for x in emb.strip('[]').split(',')]
            embeddings[claim_id] = list(emb)

        # Build claims list
        for claim in snapshot.claims.values():
            if claim.id in embeddings and claim.text:
                page = snapshot.pages.get(claim.page_id, None)
                source_name = page.url.split('/')[2] if page and page.url else "Unknown"

                self.claims.append(StreamClaim(
                    id=claim.id,
                    text=claim.text,
                    page_id=claim.page_id or "",
                    embedding=embeddings[claim.id],
                    source_name=source_name
                ))

        print(f"Loaded {len(self.claims)} claims for demo")

    async def emit_event(self, event_type: str, data: dict):
        """Emit event to SSE stream"""
        await self.event_queue.put({
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    async def classify_relationship(self, event_a: EmergentEU, event_b: EmergentEU, entity_sim: float = 0.0) -> RelationType:
        """
        Use LLM to classify relationship between two events.
        Mass determines priority (which we check first), but semantics determines relationship.
        """
        desc_a = " ".join(event_a.texts[:3])[:300]
        desc_b = " ".join(event_b.texts[:3])[:300]

        # Add entity overlap hint if significant
        entity_hint = ""
        if entity_sim >= 0.3:
            entities_a = self.extract_key_entities(event_a)
            entities_b = self.extract_key_entities(event_b)
            shared = entities_a & entities_b
            if shared:
                entity_hint = f"\nNote: Both events share key entities: {', '.join(shared)}"

        prompt = f"""Classify the relationship between these two news events.

Event A: {desc_a}

Event B: {desc_b}
{entity_hint}

**Key question: Would a reader expect to see both events grouped under ONE news story?**

Options:
1. SAME - Identical event, should merge (e.g., duplicate reports)
2. CONTAINS - A fully encompasses B (e.g., "Election results" contains "Vote count in Florida")
3. CONTAINED_BY - B fully encompasses A
4. SIBLING - Part of the SAME NEWS STORY (e.g., "Fire kills 36" + "Politicians send condolences" = same story)
   - Reactions, responses, developments about the same underlying event are SIBLINGS
   - Different angles on the same incident are SIBLINGS
5. CAUSES - A directly caused B (explicit causal chain)
6. RELATES - Tangentially related but DIFFERENT stories (e.g., "Fire in HK" + "Fire safety regulations debate")

If events share key entities (location, people, incident), they are likely SIBLING unless clearly different stories.

Return ONLY: SAME, CONTAINS, CONTAINED_BY, SIBLING, CAUSES, or RELATES"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0
                    },
                    timeout=30
                )
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip().upper()

                if "SAME" in answer:
                    return RelationType.SAME
                elif "CONTAINED_BY" in answer:
                    return RelationType.CONTAINED_BY
                elif "CONTAINS" in answer:
                    return RelationType.CONTAINS
                elif "SIBLING" in answer:
                    return RelationType.SIBLING
                elif "CAUSES" in answer:
                    return RelationType.CAUSES
                else:
                    return RelationType.RELATES
        except Exception as e:
            print(f"Relationship classification error: {e}")
            return RelationType.RELATES

    async def create_parent_eu(self, child_a: EmergentEU, child_b: EmergentEU) -> EmergentEU:
        """
        Create a parent EU at level+1 for sibling children.

        Works at any level:
        - Level 2 siblings -> Create level 3 parent
        - Level 3 siblings -> Create level 4 parent
        - etc. (infinitely fractal)
        """
        parent_level = child_a.level + 1
        parent_id = f"eu_{parent_level}_{len(self.get_eus_by_level(parent_level))}"

        # Combine embeddings weighted by mass
        total_mass = child_a.mass() + child_b.mass()
        weight_a = child_a.mass() / total_mass
        weight_b = child_b.mass() / total_mass

        combined_embedding = [
            weight_a * child_a.embedding[i] + weight_b * child_b.embedding[i]
            for i in range(len(child_a.embedding))
        ]

        parent = EmergentEU(
            id=parent_id,
            level=parent_level,
            embedding=combined_embedding,
            claim_ids=child_a.claim_ids + child_b.claim_ids,
            texts=child_a.texts[:2] + child_b.texts[:2],
            page_ids=child_a.page_ids | child_b.page_ids,
            children=[child_a.id, child_b.id],
            internal_corr=child_a.internal_corr + child_b.internal_corr,
            internal_contra=child_a.internal_contra + child_b.internal_contra
        )

        # Generate canonical name and byline for parent
        parent.canonical_name = await self.generate_canonical_name(parent)
        parent.byline = await self.generate_byline(parent)
        parent.headline = parent.canonical_name  # Legacy

        # Add to unified collection
        self.eus[parent_id] = parent
        child_a.parent_id = parent_id
        child_b.parent_id = parent_id

        # Update stats
        self.stats['eus_by_level'][parent_level] = self.stats['eus_by_level'].get(parent_level, 0) + 1

        return parent

    async def generate_parent_headline(self, child_a: EmergentEU, child_b: EmergentEU) -> str:
        """Generate headline for parent EU covering sibling children"""
        # Higher mass child gets priority in framing
        if child_a.mass() >= child_b.mass():
            primary, secondary = child_a, child_b
        else:
            primary, secondary = child_b, child_a

        prompt = f"""Generate a concise headline (max 8 words) for a news topic covering both:

Primary: {primary.headline or primary.texts[0][:80]}
Secondary: {secondary.headline or secondary.texts[0][:80]}

The headline should be broader, encompassing both aspects.

Headline:"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 25,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                result = response.json()
                return result['choices'][0]['message']['content'].strip().strip('"')
        except Exception as e:
            print(f"Parent headline error: {e}")
            return f"Coverage: {primary.headline}"

    async def process_claim(self, claim: StreamClaim):
        """Process a single claim - route to appropriate EU in unified hierarchy"""
        self.stats['claims_processed'] += 1

        # MASS-PRIORITY ROUTING: Sort level-1 EUs by mass (check massive ones FIRST)
        # Mass determines priority, semantics determines relationship
        level1_eus = sorted(
            self.get_eus_by_level(1),
            key=lambda s: s.mass(),
            reverse=True
        )

        # Find best matching sub-event (checking massive ones first)
        best_eu = None
        best_sim = 0.0
        for eu in level1_eus:
            sim = self.cosine_similarity(claim.embedding, eu.embedding)
            if sim > best_sim:
                best_sim = sim
                best_eu = eu

        if best_eu and best_sim >= SIM_THRESHOLD:
            # Absorb into existing sub-event
            best_eu.claim_ids.append(claim.id)
            best_eu.texts.append(claim.text)
            best_eu.page_ids.add(claim.page_id)
            best_eu.update_embedding(claim.embedding)
            best_eu.last_activity = datetime.now()
            best_eu.internal_corr += 1
            self.claim_to_eu[claim.id] = best_eu.id

            await self.emit_event('claim_absorbed', {
                'claim_id': claim.id,
                'claim_text': claim.text[:100],
                'source': claim.source_name,
                'eu_id': best_eu.id,
                'eu_level': best_eu.level,
                'eu_size': best_eu.size(),
                'similarity': round(best_sim, 2)
            })

            # Propagate up the hierarchy - update all ancestors
            await self._propagate_claim_up(best_eu, claim)

            # FOOD SHARING: Feed entities from this claim
            # Claims nourish both events AND entities (doc 71 ecosystem)
            await self.feed_entities(claim, best_eu.id)
        else:
            # Create new level-1 EU (sub-event)
            eu_id = f"eu_1_{len(self.get_eus_by_level(1))}"
            eu = EmergentEU(
                id=eu_id,
                level=1,
                embedding=claim.embedding.copy(),
                claim_ids=[claim.id],
                texts=[claim.text],
                page_ids={claim.page_id},
                internal_corr=1
            )
            self.eus[eu_id] = eu
            self.claim_to_eu[claim.id] = eu_id
            self.stats['eus_by_level'][1] = self.stats['eus_by_level'].get(1, 0) + 1

            await self.emit_event('eu_created', {
                'eu_id': eu_id,
                'level': 1,
                'claim_text': claim.text[:150],
                'source': claim.source_name,
                'size': 1
            })

            # FOOD SHARING: Feed entities from new claim too
            await self.feed_entities(claim, eu_id)

        # Try to merge EUs upward in hierarchy
        await self.try_merge()

    async def _propagate_claim_up(self, eu: EmergentEU, claim: StreamClaim):
        """Propagate a claim absorption up the hierarchy to all ancestors"""
        current = eu
        while current.parent_id:
            parent = self.eus.get(current.parent_id)
            if not parent:
                break

            parent.claim_ids.append(claim.id)
            parent.texts.append(claim.text)
            parent.page_ids.add(claim.page_id)
            parent.last_activity = datetime.now()
            parent.internal_corr += 1

            # Update byline periodically (every 10 claims) for level 2+ events
            # Byline reflects current state, so it should update as new info arrives
            if parent.level >= 2 and len(parent.claim_ids) % 10 == 0:
                parent.byline = await self.generate_byline(parent)

            await self.emit_event('eu_updated', {
                'eu_id': parent.id,
                'level': parent.level,
                'eu': parent.to_dict()
            })

            current = parent

    async def try_merge(self):
        """
        Try to merge EUs upward in hierarchy.

        Unified approach:
        - Level 1 (sub-events) with size >= 3 and no parent -> try to merge into level 2
        - Level 2 (events) -> check relationships with other level 2 EUs
        - Sibling relationships create level 3 (frames)
        - Can continue infinitely: sibling frames -> level 4, etc.
        """
        # Find orphan level-1 EUs ready to merge (size >= 3, no parent)
        orphan_subs = [
            eu for eu in self.get_eus_by_level(1)
            if eu.parent_id is None and eu.size() >= 3
        ]

        for sub in orphan_subs:
            # MASS-PRIORITY: Sort level-2 EUs by mass (check massive ones first)
            sorted_events = sorted(
                self.get_eus_by_level(2),
                key=lambda e: e.mass(),
                reverse=True
            )

            # Find best matching event (checking massive ones first)
            best_event = None
            best_sim = 0.0
            for event in sorted_events:
                sim = self.cosine_similarity(sub.embedding, event.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_event = event

            if best_event and best_sim >= EVENT_MERGE_THRESHOLD:
                # Add to existing event (level-2 EU)
                best_event.children.append(sub.id)
                best_event.claim_ids.extend(sub.claim_ids)
                best_event.texts.extend(sub.texts)
                best_event.page_ids.update(sub.page_ids)
                best_event.internal_corr += sub.internal_corr
                best_event.internal_contra += sub.internal_contra
                sub.parent_id = best_event.id
                best_event.last_activity = datetime.now()

                # Update event embedding
                n = len(best_event.children)
                best_event.embedding = [
                    (best_event.embedding[i] * (n - 1) + sub.embedding[i]) / n
                    for i in range(len(best_event.embedding))
                ]

                await self.emit_event('eu_merged', {
                    'child_id': sub.id,
                    'parent_id': best_event.id,
                    'parent_level': best_event.level,
                    'parent': best_event.to_dict()
                })
            else:
                # Create new level-2 EU (event)
                event_id = f"eu_2_{len(self.get_eus_by_level(2))}"
                event = EmergentEU(
                    id=event_id,
                    level=2,
                    embedding=sub.embedding.copy(),
                    claim_ids=sub.claim_ids.copy(),
                    texts=sub.texts.copy(),
                    page_ids=sub.page_ids.copy(),
                    children=[sub.id],
                    internal_corr=sub.internal_corr,
                    internal_contra=sub.internal_contra
                )
                self.eus[event_id] = event
                sub.parent_id = event_id
                self.stats['eus_by_level'][2] = self.stats['eus_by_level'].get(2, 0) + 1

                # Generate canonical name and byline
                event.canonical_name = await self.generate_canonical_name(event)
                event.byline = await self.generate_byline(event)
                event.headline = event.canonical_name  # Legacy compatibility

                await self.emit_event('eu_created', {
                    'eu_id': event_id,
                    'level': 2,
                    'eu': event.to_dict()
                })

                # After creating new level-2 EU, check for relationships with peers
                await self.try_peer_relationships(event)

    def extract_key_entities(self, event: EmergentEU) -> Set[str]:
        """Extract key entities from event claims for entity-based matching"""
        entities = set()
        text = " ".join(event.texts[:5]).lower()

        # Location entities
        locations = ["hong kong", "tai po", "utah", "china", "venezuela", "ukraine"]
        for loc in locations:
            if loc in text:
                entities.add(loc)

        # Event type entities
        event_types = ["fire", "assassination", "killed", "arrested", "sentenced", "trial"]
        for et in event_types:
            if et in text:
                entities.add(et)

        # Key names (simplified - in production use NER)
        names = ["jimmy lai", "charlie kirk", "lee jae myung", "do kwon", "king charles"]
        for name in names:
            if name in text:
                entities.add(name)

        return entities

    def entity_overlap(self, event_a: EmergentEU, event_b: EmergentEU) -> float:
        """Calculate entity overlap between two events"""
        entities_a = self.extract_key_entities(event_a)
        entities_b = self.extract_key_entities(event_b)

        if not entities_a or not entities_b:
            return 0.0

        intersection = entities_a & entities_b
        union = entities_a | entities_b
        return len(intersection) / len(union) if union else 0.0

    async def try_peer_relationships(self, new_eu: EmergentEU):
        """
        Check if new EU should be linked to existing peers via semantic relationships.

        Generalized for any level:
        - Level 2 peers: Check other level-2 EUs, create level-3 frame if siblings
        - Level 3+ peers: Could create level-4+, infinitely fractal
        """
        peer_level = new_eu.level

        # Sort peers by mass (check massive ones first)
        peers = sorted(
            [eu for eu in self.get_eus_by_level(peer_level) if eu.id != new_eu.id],
            key=lambda e: e.mass(),
            reverse=True
        )

        for other in peers:
            # Skip if already share a parent
            if new_eu.parent_id and other.parent_id and new_eu.parent_id == other.parent_id:
                continue

            sim = self.cosine_similarity(new_eu.embedding, other.embedding)
            entity_sim = self.entity_overlap(new_eu, other)

            # Use EITHER embedding similarity OR entity overlap as gate
            # This catches "same story, different angle" cases
            should_check = sim >= RELATIONSHIP_THRESHOLD or entity_sim >= 0.3

            if should_check:
                if sim < RELATIONSHIP_THRESHOLD and entity_sim >= 0.3:
                    print(f"Entity match triggered: {new_eu.id} ~ {other.id} (entities: {entity_sim:.2f}, embed: {sim:.2f})")

                # Use semantic classification to determine relationship type
                # Pass entity_sim to help LLM understand they share entities
                rel_type = await self.classify_relationship(other, new_eu, entity_sim)  # other first (higher mass priority)
                print(f"LLM classified: {other.id} ~ {new_eu.id} = {rel_type.value}")

                if rel_type == RelationType.SAME:
                    # Should merge - edge case since we just created it
                    print(f"Same event detected: {new_eu.id} ~ {other.id}")
                    pass

                elif rel_type == RelationType.CONTAINS:
                    # Other contains new EU (make new EU a child of other)
                    new_eu.parent_id = other.id
                    if new_eu.id not in other.children:
                        other.children.append(new_eu.id)
                    await self.emit_event('containment', {
                        'parent_id': other.id,
                        'child_id': new_eu.id,
                        'parent': other.to_dict(),
                        'child': new_eu.to_dict()
                    })
                    print(f"Containment: {other.headline} contains {new_eu.headline}")
                    break

                elif rel_type == RelationType.CONTAINED_BY:
                    # New EU contains other (make other a child of new EU)
                    other.parent_id = new_eu.id
                    if other.id not in new_eu.children:
                        new_eu.children.append(other.id)
                    await self.emit_event('containment', {
                        'parent_id': new_eu.id,
                        'child_id': other.id,
                        'parent': new_eu.to_dict(),
                        'child': other.to_dict()
                    })
                    print(f"Containment: {new_eu.headline} contains {other.headline}")
                    break

                elif rel_type == RelationType.SIBLING:
                    # Create parent at next level for both siblings
                    # Check if other already has a parent at next level
                    if other.parent_id and self.eus.get(other.parent_id):
                        # Add new EU to existing parent
                        parent = self.eus[other.parent_id]
                        parent.children.append(new_eu.id)
                        parent.claim_ids.extend(new_eu.claim_ids)
                        parent.page_ids.update(new_eu.page_ids)
                        new_eu.parent_id = parent.id

                        await self.emit_event('sibling_added', {
                            'parent_id': parent.id,
                            'child_id': new_eu.id,
                            'parent_level': parent.level,
                            'parent': parent.to_dict(),
                            'child': new_eu.to_dict()
                        })
                        print(f"Sibling added to parent: {new_eu.headline} -> {parent.headline}")
                    else:
                        # Create new parent at level+1
                        parent = await self.create_parent_eu(other, new_eu)
                        await self.emit_event('eu_created', {
                            'eu_id': parent.id,
                            'level': parent.level,
                            'eu': parent.to_dict(),
                            'children': [other.to_dict(), new_eu.to_dict()]
                        })
                        print(f"Parent EU created: {parent.headline} (level {parent.level}, children: {other.headline}, {new_eu.headline})")
                    break

                elif rel_type == RelationType.CAUSES:
                    await self.emit_event('causal_link', {
                        'cause_id': other.id,
                        'effect_id': new_eu.id,
                        'cause': other.to_dict(),
                        'effect': new_eu.to_dict()
                    })
                    print(f"Causal: {other.headline} -> {new_eu.headline}")
                    # Don't break - could have multiple relationships

                else:
                    # RELATES - just note the association
                    await self.emit_event('related', {
                        'eu_a': other.to_dict(),
                        'eu_b': new_eu.to_dict(),
                        'similarity': round(sim, 2)
                    })
                    # Don't break - could have multiple

    async def generate_canonical_name(self, event: EmergentEU) -> str:
        """Generate stable canonical name for event - no volatile metrics"""
        sample_claims = event.texts[:5]
        claims_text = "\n".join(f"- {c[:100]}" for c in sample_claims)

        prompt = f"""Generate a stable canonical name (max 8 words) for this news event.

IMPORTANT: Do NOT include numbers that might change (death tolls, injuries, missing counts).
Focus on: What happened, Where, and key entities involved.

Good examples:
- "Hong Kong Tai Po High-Rise Fire"
- "South Korean President Impeachment"
- "Charlie Kirk Utah Rally Shooting"

Bad examples (include volatile numbers):
- "Fire Kills 36 in Hong Kong" (death toll may change)
- "279 Missing in Building Fire" (missing count may change)

Claims:
{claims_text}

Canonical name:"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 30,
                        "temperature": 0.3  # Lower temp for stability
                    },
                    timeout=30
                )
                result = response.json()
                return result['choices'][0]['message']['content'].strip().strip('"')
        except Exception as e:
            print(f"Canonical name generation error: {e}")
            return f"Event {event.id}"

    async def generate_byline(self, event: EmergentEU) -> str:
        """Generate current-state byline - includes latest metrics"""
        # Use most recent claims for byline (not oldest)
        recent_claims = event.texts[-5:] if len(event.texts) > 5 else event.texts
        claims_text = "\n".join(f"- {c[:100]}" for c in recent_claims)

        prompt = f"""Generate a brief byline (max 12 words) summarizing the CURRENT state of this event.

Include the latest numbers/metrics if relevant (death toll, injuries, status).
This will be updated as new information arrives.

Recent claims:
{claims_text}

Byline:"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 30,
                        "temperature": 0.5
                    },
                    timeout=30
                )
                result = response.json()
                return result['choices'][0]['message']['content'].strip().strip('"')
        except Exception as e:
            print(f"Byline generation error: {e}")
            return ""

    async def generate_headline(self, event: EmergentEU) -> str:
        """Legacy - now generates canonical name"""
        return await self.generate_canonical_name(event)

    async def generate_narrative(self, event: EmergentEU) -> str:
        """Generate narrative for event using LLM"""
        sample_claims = event.texts[:10]
        claims_text = "\n".join(f"- {c}" for c in sample_claims)

        prompt = f"""Write a brief news summary (2-3 paragraphs) based on these claims:

{claims_text}

Include:
- What happened (main facts)
- Key details (who, where, when)
- Current status

Summary:"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 300,
                        "temperature": 0.7
                    },
                    timeout=60
                )
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Narrative generation error: {e}")
            return "Narrative generation failed."

    # ============== Entity Organism Methods ==============

    async def extract_entities_from_claim(self, claim_text: str) -> List[dict]:
        """
        Extract entities from a claim using LLM.
        Returns list of {name, type} dicts.
        """
        prompt = f"""Extract named entities from this news claim. Return JSON array.

Claim: {claim_text[:500]}

Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT

Return format (JSON array only, no markdown):
[{{"name": "Jimmy Lai", "type": "PERSON"}}, {{"name": "Hong Kong", "type": "LOCATION"}}]

If no entities found, return: []
Entities:"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0
                    },
                    timeout=30
                )
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()

                # Clean up potential markdown
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()

                entities = json.loads(content)
                return entities if isinstance(entities, list) else []
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication"""
        return name.lower().strip()

    def get_or_create_entity(self, name: str, entity_type: str) -> EntityOrganism:
        """Get existing entity or create new one"""
        normalized = self.normalize_entity_name(name)

        if normalized in self.entity_name_to_id:
            return self.entities[self.entity_name_to_id[normalized]]

        # Create new entity
        entity_id = f"ent_{len(self.entities)}"
        try:
            etype = EntityType(entity_type)
        except ValueError:
            etype = EntityType.CONCEPT

        entity = EntityOrganism(
            id=entity_id,
            name=name,
            entity_type=etype
        )
        self.entities[entity_id] = entity
        self.entity_name_to_id[normalized] = entity_id
        self.stats['entities_created'] += 1

        return entity

    async def feed_entities(self, claim: StreamClaim, event_id: str):
        """
        Extract entities from claim and feed them (food sharing).
        This implements the doc 71 concept: claims nourish multiple organisms.
        """
        entities_data = await self.extract_entities_from_claim(claim.text)

        for ent_data in entities_data:
            name = ent_data.get('name', '')
            etype = ent_data.get('type', 'CONCEPT')

            if not name or len(name) < 2:
                continue

            entity = self.get_or_create_entity(name, etype)
            entity.feed(claim.id, claim.text, event_id, claim.page_id)
            self.stats['entity_feeds'] += 1

            # Emit entity update event
            await self.emit_event('entity_fed', {
                'entity_id': entity.id,
                'entity_name': entity.name,
                'entity_type': entity.entity_type.value,
                'claim_id': claim.id,
                'event_id': event_id,
                'mention_count': entity.mention_count,
                'is_qualia_capable': entity.is_qualia_capable()
            })

            # Log qualia threshold crossing
            if entity.mention_count == 11 and entity.is_qualia_capable():
                print(f"🧠 Entity became qualia-capable: {entity.name}")
                await self.emit_event('entity_qualia', {
                    'entity_id': entity.id,
                    'entity_name': entity.name,
                    'entity': entity.to_dict()
                })

    async def generate_entity_profile(self, entity: EntityOrganism) -> str:
        """Generate profile for entity based on claims mentioning it"""
        sample_claims = entity.texts[:10]
        claims_text = "\n".join(f"- {c[:150]}" for c in sample_claims)

        prompt = f"""Based on these news claims, write a brief profile (2-3 sentences) for {entity.name}.

Claims mentioning this entity:
{claims_text}

Profile:"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Entity profile error: {e}")
            return f"{entity.name} - {entity.entity_type.value}"

    async def start_streaming(self, delay: float = 1.0):
        """Start streaming claims"""
        self.stream_delay = delay
        self.is_streaming = True

        while self.is_streaming and self.current_index < len(self.claims):
            claim = self.claims[self.current_index]
            await self.process_claim(claim)
            self.current_index += 1

            await self.emit_event('progress', {
                'current': self.current_index,
                'total': len(self.claims),
                'percent': round(100 * self.current_index / len(self.claims), 1)
            })

            await asyncio.sleep(self.stream_delay)

        self.is_streaming = False

    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False

    def reset(self):
        """Reset demo state"""
        self.eus.clear()
        self.entities.clear()
        self.entity_name_to_id.clear()
        self.claim_to_eu.clear()
        self.current_index = 0
        self.is_streaming = False
        self.stats = {
            'claims_processed': 0,
            'eus_by_level': {1: 0, 2: 0, 3: 0},
            'entities_created': 0,
            'entity_feeds': 0,
        }


# ============== Global State ==============

engine = DemoEngine()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading claims...")
    await engine.load_claims()
    print("Demo server ready!")
    yield
    # Shutdown
    engine.stop_streaming()


app = FastAPI(title="Fractal Event Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def homepage():
    """Serve the homepage (ecosystem view)"""
    return HTMLResponse(content=open('/app/test_eu/demo/homepage.html').read())


@app.get("/event")
async def event_page():
    """Serve the single event page (product view)"""
    return HTMLResponse(content=open('/app/test_eu/demo/event-single.html').read())


@app.get("/demo")
async def demo_page():
    """Serve the demo/admin page (shows all events streaming)"""
    return HTMLResponse(content=open('/app/test_eu/demo/index.html').read())


@app.get("/api/eus")
async def get_all_eus():
    """
    Get all EUs organized by level.
    This is the unified view of the fractal hierarchy.
    """
    # Group EUs by level
    by_level = {}
    for eu in engine.eus.values():
        if eu.level not in by_level:
            by_level[eu.level] = []
        by_level[eu.level].append(eu)

    # Sort each level by mass
    result = {}
    for level, eus in by_level.items():
        sorted_eus = sorted(eus, key=lambda e: e.mass(), reverse=True)
        result[f"level_{level}"] = [e.to_dict() for e in sorted_eus]

    return {
        'eus': result,
        'stats': engine.stats,
        'total_eus': len(engine.eus)
    }


@app.get("/api/events")
async def get_events():
    """
    Get all events (level 2+) sorted by mass.
    Higher-level events (level 3+) are just parent events that group related events.
    """
    all_events = sorted(
        [eu for eu in engine.eus.values() if eu.level >= 2],
        key=lambda e: e.mass(),
        reverse=True
    )
    return {
        'events': [e.to_dict() for e in all_events],
        'stats': engine.stats
    }


@app.get("/api/eus/{eu_id}")
async def get_eu(eu_id: str):
    """Get any EU by ID with its full hierarchy"""
    if eu_id not in engine.eus:
        return {"error": "EU not found"}

    eu = engine.eus[eu_id]

    # Generate narrative if level >= 2 and not cached
    if eu.level >= 2 and not eu.narrative:
        eu.narrative = await engine.generate_narrative(eu)

    # Get children recursively
    def get_children_tree(eu_id: str, depth: int = 0) -> dict:
        if eu_id not in engine.eus or depth > 5:
            return None
        child = engine.eus[eu_id]
        return {
            **child.to_dict(),
            'children_tree': [
                get_children_tree(c_id, depth + 1)
                for c_id in child.children
                if c_id in engine.eus
            ]
        }

    children_tree = [
        get_children_tree(child_id)
        for child_id in eu.children
        if child_id in engine.eus
    ]

    # Get parent chain
    parent_chain = []
    current = eu
    while current.parent_id and current.parent_id in engine.eus:
        parent = engine.eus[current.parent_id]
        parent_chain.append(parent.to_dict())
        current = parent

    return {
        'eu': eu.to_dict(),
        'narrative': eu.narrative,
        'children_tree': children_tree,
        'parent_chain': parent_chain,
        'all_claims': eu.texts
    }


@app.get("/api/events/{event_id}")
async def get_event(event_id: str):
    """Get single event with full details - backward compatible"""
    # Try to find by exact ID or by old format
    eu = engine.eus.get(event_id)
    if not eu:
        # Try old format event_X -> eu_2_X
        for eu_id, candidate in engine.eus.items():
            if candidate.level == 2 and (eu_id == event_id or eu_id.replace('eu_2_', 'event_') == event_id):
                eu = candidate
                break

    if not eu:
        return {"error": "Event not found"}

    # Generate narrative if not cached
    if not eu.narrative:
        eu.narrative = await engine.generate_narrative(eu)

    # Get sub-events (children)
    sub_events = [
        engine.eus[sub_id].to_dict()
        for sub_id in eu.children
        if sub_id in engine.eus
    ]

    return {
        'event': eu.to_dict(),
        'narrative': eu.narrative,
        'sub_events': sub_events,
        'all_claims': eu.texts
    }


# ============== Entity Organism Endpoints ==============

@app.get("/entity")
async def entity_page():
    """Serve the entity page (entity organism view)"""
    return HTMLResponse(content=open('/app/test_eu/demo/entity-page.html').read())


@app.get("/api/entities")
async def get_entities():
    """
    Get all entities sorted by mass.
    Shows entity organisms that have grown from claims.
    """
    all_entities = sorted(
        engine.entities.values(),
        key=lambda e: e.mass(),
        reverse=True
    )

    # Group by type
    by_type = {}
    for entity in all_entities:
        etype = entity.entity_type.value
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(entity.to_dict())

    return {
        'entities': [e.to_dict() for e in all_entities],
        'by_type': by_type,
        'total': len(all_entities),
        'qualia_capable': len([e for e in all_entities if e.is_qualia_capable()]),
        'stats': {
            'entities_created': engine.stats.get('entities_created', 0),
            'entity_feeds': engine.stats.get('entity_feeds', 0),
        }
    }


@app.get("/api/entities/{entity_id}")
async def get_entity(entity_id: str):
    """
    Get single entity with full details.
    Shows entity-event symbiosis: which events this entity appears in.
    """
    entity = engine.entities.get(entity_id)
    if not entity:
        return {"error": "Entity not found"}

    # Generate profile if not cached and entity is qualia-capable
    if not entity.profile and entity.is_qualia_capable():
        entity.profile = await engine.generate_entity_profile(entity)

    # Get events this entity appears in
    related_events = []
    for event_id in entity.event_ids:
        if event_id in engine.eus:
            eu = engine.eus[event_id]
            # Find parent event (level 2+) if this is a sub-event
            if eu.level == 1 and eu.parent_id and eu.parent_id in engine.eus:
                parent = engine.eus[eu.parent_id]
                related_events.append({
                    **parent.to_dict(),
                    'via_sub_event': eu.id
                })
            elif eu.level >= 2:
                related_events.append(eu.to_dict())

    # Deduplicate events (same entity might appear in multiple sub-events of same event)
    seen_events = set()
    unique_events = []
    for event in related_events:
        if event['id'] not in seen_events:
            seen_events.add(event['id'])
            unique_events.append(event)

    return {
        'entity': entity.to_dict(),
        'profile': entity.profile,
        'related_events': unique_events,
        'all_claims': entity.texts,
        'symbiosis': {
            'event_count': len(unique_events),
            'claim_count': len(entity.claim_ids),
            'source_count': len(entity.page_ids),
            'mass': round(entity.mass(), 2),
            'is_qualia_capable': entity.is_qualia_capable()
        }
    }


@app.get("/api/stream")
async def event_stream(request: Request):
    """SSE endpoint for live updates"""
    async def generate():
        while True:
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.wait_for(
                    engine.event_queue.get(),
                    timeout=30.0
                )
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/start")
async def start_stream(delay: float = 0.5):
    """Start streaming claims"""
    if not engine.is_streaming:
        asyncio.create_task(engine.start_streaming(delay))
    return {"status": "started", "delay": delay}


@app.post("/api/stop")
async def stop_stream():
    """Stop streaming"""
    engine.stop_streaming()
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset():
    """Reset demo"""
    engine.reset()
    return {"status": "reset"}


@app.get("/api/status")
async def status():
    """Get demo status"""
    qualia_entities = [e for e in engine.entities.values() if e.is_qualia_capable()]
    return {
        'is_streaming': engine.is_streaming,
        'progress': {
            'current': engine.current_index,
            'total': len(engine.claims),
            'percent': round(100 * engine.current_index / len(engine.claims), 1) if engine.claims else 0
        },
        'stats': engine.stats,
        'total_eus': len(engine.eus),
        'eus_by_level': {
            'level_1': len(engine.get_eus_by_level(1)),
            'level_2': len(engine.get_eus_by_level(2)),
            'level_3': len(engine.get_eus_by_level(3)),
        },
        'sub_event_count': len(engine.get_eus_by_level(1)),
        'event_count': len([eu for eu in engine.eus.values() if eu.level >= 2]),
        # Entity organism stats
        'entity_count': len(engine.entities),
        'qualia_capable_entities': len(qualia_entities),
        'entity_types': {
            'PERSON': len([e for e in engine.entities.values() if e.entity_type == EntityType.PERSON]),
            'ORGANIZATION': len([e for e in engine.entities.values() if e.entity_type == EntityType.ORGANIZATION]),
            'LOCATION': len([e for e in engine.entities.values() if e.entity_type == EntityType.LOCATION]),
            'CONCEPT': len([e for e in engine.entities.values() if e.entity_type == EntityType.CONCEPT]),
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
