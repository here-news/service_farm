"""
Load graph data from Neo4j for EU experiments.

READ-ONLY: Does not modify any data in Neo4j.

Run inside container:
    docker exec herenews-app python /app/test_eu/load_graph.py
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, '/app/backend')

from neo4j import AsyncGraphDatabase


@dataclass
class ClaimData:
    """Raw claim data from graph"""
    id: str
    text: str
    confidence: Optional[float] = None
    event_time: Optional[str] = None
    # Relationships
    page_id: Optional[str] = None
    event_id: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    corroborates_ids: List[str] = field(default_factory=list)
    contradicts_ids: List[str] = field(default_factory=list)
    updates_ids: List[str] = field(default_factory=list)
    corroborated_by_ids: List[str] = field(default_factory=list)
    contradicted_by_ids: List[str] = field(default_factory=list)


@dataclass
class EntityData:
    """Raw entity data from graph"""
    id: str
    canonical_name: str
    entity_type: str
    wikidata_qid: Optional[str] = None
    mention_count: int = 0


@dataclass
class EventData:
    """Raw event data from graph"""
    id: str
    canonical_name: str
    event_type: Optional[str] = None
    coherence: Optional[float] = None
    status: Optional[str] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)


@dataclass
class PageData:
    """Raw page data from graph"""
    id: str
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    claim_ids: List[str] = field(default_factory=list)


@dataclass
class GraphSnapshot:
    """Complete snapshot of graph state for experiments"""
    claims: Dict[str, ClaimData] = field(default_factory=dict)
    entities: Dict[str, EntityData] = field(default_factory=dict)
    events: Dict[str, EventData] = field(default_factory=dict)
    pages: Dict[str, PageData] = field(default_factory=dict)
    timestamp: str = ""

    def summary(self) -> str:
        return (
            f"GraphSnapshot @ {self.timestamp}\n"
            f"  Claims: {len(self.claims)}\n"
            f"  Entities: {len(self.entities)}\n"
            f"  Events: {len(self.events)}\n"
            f"  Pages: {len(self.pages)}"
        )


async def load_graph() -> GraphSnapshot:
    """
    Load complete graph snapshot from Neo4j.
    READ-ONLY operation.
    """
    uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    snapshot = GraphSnapshot(timestamp=datetime.utcnow().isoformat())

    async with driver.session() as session:
        # Load claims with all relationships
        print("Loading claims...")
        result = await session.run("""
            MATCH (c:Claim)
            OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
            OPTIONAL MATCH (e:Event)-[:INTAKES]->(c)
            OPTIONAL MATCH (c)-[:MENTIONS]->(en:Entity)
            OPTIONAL MATCH (c)-[:CORROBORATES]->(corr:Claim)
            OPTIONAL MATCH (c)-[:CONTRADICTS]->(contra:Claim)
            OPTIONAL MATCH (c)-[:UPDATES]->(upd:Claim)
            OPTIONAL MATCH (corr_by:Claim)-[:CORROBORATES]->(c)
            OPTIONAL MATCH (contra_by:Claim)-[:CONTRADICTS]->(c)
            RETURN c.id as id, c.text as text, c.confidence as confidence,
                   c.event_time as event_time,
                   p.id as page_id, e.id as event_id,
                   collect(DISTINCT en.id) as entity_ids,
                   collect(DISTINCT corr.id) as corroborates_ids,
                   collect(DISTINCT contra.id) as contradicts_ids,
                   collect(DISTINCT upd.id) as updates_ids,
                   collect(DISTINCT corr_by.id) as corroborated_by_ids,
                   collect(DISTINCT contra_by.id) as contradicted_by_ids
        """)
        async for record in result:
            claim = ClaimData(
                id=record['id'],
                text=record['text'] or "",
                confidence=record['confidence'],
                event_time=str(record['event_time']) if record['event_time'] else None,
                page_id=record['page_id'],
                event_id=record['event_id'],
                entity_ids=[e for e in record['entity_ids'] if e],
                corroborates_ids=[c for c in record['corroborates_ids'] if c],
                contradicts_ids=[c for c in record['contradicts_ids'] if c],
                updates_ids=[u for u in record['updates_ids'] if u],
                corroborated_by_ids=[c for c in record['corroborated_by_ids'] if c],
                contradicted_by_ids=[c for c in record['contradicted_by_ids'] if c],
            )
            snapshot.claims[claim.id] = claim
        print(f"  Loaded {len(snapshot.claims)} claims")

        # Load entities with mention counts
        print("Loading entities...")
        result = await session.run("""
            MATCH (en:Entity)
            OPTIONAL MATCH (c:Claim)-[:MENTIONS]->(en)
            RETURN en.id as id, en.canonical_name as name,
                   en.entity_type as type, en.wikidata_qid as qid,
                   count(c) as mention_count
        """)
        async for record in result:
            entity = EntityData(
                id=record['id'],
                canonical_name=record['name'] or "",
                entity_type=record['type'] or "UNKNOWN",
                wikidata_qid=record['qid'],
                mention_count=record['mention_count']
            )
            snapshot.entities[entity.id] = entity
        print(f"  Loaded {len(snapshot.entities)} entities")

        # Load events with relationships
        print("Loading events...")
        result = await session.run("""
            MATCH (e:Event)
            OPTIONAL MATCH (e)-[:INTAKES]->(c:Claim)
            OPTIONAL MATCH (e)-[:INVOLVES]->(en:Entity)
            OPTIONAL MATCH (parent:Event)-[:CONTAINS]->(e)
            OPTIONAL MATCH (e)-[:CONTAINS]->(child:Event)
            RETURN e.id as id, e.canonical_name as name,
                   e.event_type as type, e.coherence as coherence,
                   e.status as status,
                   parent.id as parent_id,
                   collect(DISTINCT child.id) as child_ids,
                   collect(DISTINCT c.id) as claim_ids,
                   collect(DISTINCT en.id) as entity_ids
        """)
        async for record in result:
            event = EventData(
                id=record['id'],
                canonical_name=record['name'] or "",
                event_type=record['type'],
                coherence=record['coherence'],
                status=record['status'],
                parent_id=record['parent_id'],
                child_ids=[c for c in record['child_ids'] if c],
                claim_ids=[c for c in record['claim_ids'] if c],
                entity_ids=[e for e in record['entity_ids'] if e],
            )
            snapshot.events[event.id] = event
        print(f"  Loaded {len(snapshot.events)} events")

        # Load pages
        print("Loading pages...")
        result = await session.run("""
            MATCH (p:Page)
            OPTIONAL MATCH (p)-[:EMITS]->(c:Claim)
            RETURN p.id as id, p.url as url, p.title as title,
                   p.domain as domain,
                   collect(DISTINCT c.id) as claim_ids
        """)
        async for record in result:
            page = PageData(
                id=record['id'],
                url=record['url'] or "",
                title=record['title'],
                domain=record['domain'],
                claim_ids=[c for c in record['claim_ids'] if c],
            )
            snapshot.pages[page.id] = page
        print(f"  Loaded {len(snapshot.pages)} pages")

    await driver.close()
    return snapshot


def save_snapshot(snapshot: GraphSnapshot, path: str = "/app/test_eu/results/snapshot.json"):
    """Save snapshot to JSON for offline analysis"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    data = {
        'timestamp': snapshot.timestamp,
        'claims': {k: asdict(v) for k, v in snapshot.claims.items()},
        'entities': {k: asdict(v) for k, v in snapshot.entities.items()},
        'events': {k: asdict(v) for k, v in snapshot.events.items()},
        'pages': {k: asdict(v) for k, v in snapshot.pages.items()},
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved snapshot to {path}")


def load_snapshot(path: str = "/app/test_eu/results/snapshot.json") -> GraphSnapshot:
    """Load snapshot from JSON"""
    with open(path, 'r') as f:
        data = json.load(f)

    snapshot = GraphSnapshot(timestamp=data['timestamp'])

    for k, v in data['claims'].items():
        snapshot.claims[k] = ClaimData(**v)
    for k, v in data['entities'].items():
        snapshot.entities[k] = EntityData(**v)
    for k, v in data['events'].items():
        snapshot.events[k] = EventData(**v)
    for k, v in data['pages'].items():
        snapshot.pages[k] = PageData(**v)

    return snapshot


async def main():
    print("=" * 60)
    print("EU Experiment: Loading Graph Snapshot")
    print("=" * 60)

    snapshot = await load_graph()
    print()
    print(snapshot.summary())
    print()

    save_snapshot(snapshot)

    # Quick stats
    print("\n" + "=" * 60)
    print("Quick Analysis")
    print("=" * 60)

    # Claims with most corroboration
    by_corroboration = sorted(
        snapshot.claims.values(),
        key=lambda c: len(c.corroborated_by_ids),
        reverse=True
    )[:5]
    print("\nTop 5 most corroborated claims:")
    for c in by_corroboration:
        print(f"  [{len(c.corroborated_by_ids)} corr] {c.text[:80]}...")

    # Claims with contradictions
    contradicted = [c for c in snapshot.claims.values() if c.contradicted_by_ids]
    print(f"\nClaims with contradictions: {len(contradicted)}")
    for c in contradicted[:3]:
        print(f"  [{len(c.contradicted_by_ids)} contra] {c.text[:80]}...")

    # Entities by mention count
    top_entities = sorted(
        snapshot.entities.values(),
        key=lambda e: e.mention_count,
        reverse=True
    )[:10]
    print("\nTop 10 entities by mention count:")
    for e in top_entities:
        print(f"  [{e.mention_count}] {e.canonical_name} ({e.entity_type})")

    # Events by claim count
    print("\nEvents by claim count:")
    for e in sorted(snapshot.events.values(), key=lambda x: len(x.claim_ids), reverse=True):
        parent_str = f" (child of {e.parent_id})" if e.parent_id else ""
        print(f"  [{len(e.claim_ids)} claims] {e.canonical_name}{parent_str}")


if __name__ == "__main__":
    asyncio.run(main())
