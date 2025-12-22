#!/usr/bin/env python3
"""
Experiment: Migration Dry Run

Test mapping existing Event data to new EU (EventfulUnit) structure.
Validates data compatibility before the shift.

Questions to answer:
1. Can we map Event → EU (level 2) cleanly?
2. Can we generate sub-events (level 1) from existing claims?
3. What data is lost/gained in migration?
4. How do existing metrics map to new metrics?
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/app')

from lib.db.pg import get_pg_pool
from lib.db.neo4j import get_neo4j_driver
from lib.embedding import get_embedding_batch
from test_eu.fractal_cluster import hierarchical_clustering_llm


@dataclass
class ExistingEvent:
    """Current Event structure from Neo4j"""
    id: str
    canonical_name: str
    coherence: float
    claim_count: int
    page_count: int
    created_at: datetime
    claims: List[Dict] = field(default_factory=list)
    pages: List[str] = field(default_factory=list)


@dataclass
class ProposedEU:
    """New EventfulUnit structure"""
    id: str
    level: int  # 0=claim, 1=sub-event, 2=event
    embedding: List[float]

    # Hierarchy
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    # Content
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: List[str] = field(default_factory=list)

    # Metrics (computed)
    internal_corr: int = 0
    internal_contra: int = 0

    def mass(self) -> float:
        size = len(self.claim_ids) if self.level <= 1 else sum(1 for _ in self.children)
        return size * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * len(self.page_ids))

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def tension(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_contra / total if total > 0 else 0.0

    def state(self) -> str:
        return "ACTIVE" if self.tension() > 0.1 else "STABLE"


class MigrationDryRun:
    """Simulates migration from Event → EU structure"""

    def __init__(self, pg_pool, neo4j_driver):
        self.pg = pg_pool
        self.neo4j = neo4j_driver

    async def load_existing_event(self, event_id: str) -> ExistingEvent:
        """Load an existing event with all its data"""
        async with self.neo4j.session() as session:
            # Get event
            result = await session.run("""
                MATCH (e:Event {id: $event_id})
                OPTIONAL MATCH (e)<-[:BELONGS_TO]-(c:Claim)
                OPTIONAL MATCH (c)<-[:CONTAINS]-(p:Page)
                RETURN e.id as id,
                       e.canonical_name as name,
                       e.coherence as coherence,
                       collect(DISTINCT {id: c.id, text: c.text, page_id: p.id}) as claims,
                       collect(DISTINCT p.id) as pages
            """, event_id=event_id)

            record = await result.single()
            if not record:
                raise ValueError(f"Event {event_id} not found")

            claims = [c for c in record['claims'] if c['id'] is not None]
            pages = [p for p in record['pages'] if p is not None]

            return ExistingEvent(
                id=record['id'],
                canonical_name=record['name'] or "Unnamed Event",
                coherence=record['coherence'] or 0.0,
                claim_count=len(claims),
                page_count=len(pages),
                created_at=datetime.now(),
                claims=claims,
                pages=pages
            )

    async def migrate_event_to_eu(self, event: ExistingEvent) -> Tuple[ProposedEU, List[ProposedEU]]:
        """
        Migrate a single Event to EU structure.
        Returns: (event_eu, sub_event_eus)
        """
        print(f"\n📦 Migrating: {event.canonical_name}")
        print(f"   Claims: {event.claim_count}, Pages: {event.page_count}")

        if not event.claims:
            print("   ⚠️ No claims to migrate")
            return None, []

        # Get embeddings for all claims
        claim_texts = [c['text'] for c in event.claims if c.get('text')]
        if not claim_texts:
            print("   ⚠️ No claim texts")
            return None, []

        print(f"   🔢 Getting embeddings for {len(claim_texts)} claims...")
        embeddings = await get_embedding_batch(claim_texts)

        # Create claim-level EUs (level 0)
        claim_eus = []
        for i, (claim, emb) in enumerate(zip(event.claims, embeddings)):
            if not claim.get('text'):
                continue
            claim_eu = ProposedEU(
                id=f"eu_claim_{claim['id']}",
                level=0,
                embedding=emb,
                claim_ids=[claim['id']],
                texts=[claim['text']],
                page_ids=[claim['page_id']] if claim.get('page_id') else []
            )
            claim_eus.append(claim_eu)

        print(f"   📝 Created {len(claim_eus)} claim EUs (level 0)")

        # Cluster claims into sub-events (level 1)
        print(f"   🔗 Clustering into sub-events...")
        sub_events = await self._cluster_into_sub_events(claim_eus, embeddings)
        print(f"   📦 Created {len(sub_events)} sub-event EUs (level 1)")

        # Create event EU (level 2)
        event_embedding = self._compute_centroid([eu.embedding for eu in sub_events])

        # Get corr/contra counts from existing event
        corr, contra = await self._get_topology_counts(event.id)

        event_eu = ProposedEU(
            id=f"eu_event_{event.id}",
            level=2,
            embedding=event_embedding,
            children=[eu.id for eu in sub_events],
            claim_ids=[c['id'] for c in event.claims if c.get('id')],
            texts=[],  # Level 2 doesn't store texts directly
            page_ids=event.pages,
            internal_corr=corr,
            internal_contra=contra
        )

        # Set parent references
        for sub in sub_events:
            sub.parent_id = event_eu.id

        return event_eu, sub_events

    async def _cluster_into_sub_events(
        self,
        claim_eus: List[ProposedEU],
        embeddings: List[List[float]]
    ) -> List[ProposedEU]:
        """Cluster claim EUs into sub-event EUs"""
        if len(claim_eus) < 3:
            # Too few claims - single sub-event
            centroid = self._compute_centroid(embeddings)
            sub_eu = ProposedEU(
                id=f"eu_sub_{claim_eus[0].claim_ids[0][:8]}",
                level=1,
                embedding=centroid,
                children=[eu.id for eu in claim_eus],
                claim_ids=[cid for eu in claim_eus for cid in eu.claim_ids],
                texts=[t for eu in claim_eus for t in eu.texts],
                page_ids=list(set(pid for eu in claim_eus for pid in eu.page_ids))
            )
            return [sub_eu]

        # Use our validated clustering
        texts = [eu.texts[0] for eu in claim_eus]
        clusters = await hierarchical_clustering_llm(
            texts,
            embeddings,
            sim_threshold=0.70,
            llm_threshold=0.55
        )

        sub_eus = []
        for i, cluster in enumerate(clusters):
            cluster_eus = [claim_eus[idx] for idx in cluster['indices']]
            cluster_embeddings = [embeddings[idx] for idx in cluster['indices']]
            centroid = self._compute_centroid(cluster_embeddings)

            sub_eu = ProposedEU(
                id=f"eu_sub_{i}_{cluster_eus[0].claim_ids[0][:8]}",
                level=1,
                embedding=centroid,
                children=[eu.id for eu in cluster_eus],
                claim_ids=[cid for eu in cluster_eus for cid in eu.claim_ids],
                texts=[t for eu in cluster_eus for t in eu.texts],
                page_ids=list(set(pid for eu in cluster_eus for pid in eu.page_ids))
            )
            sub_eus.append(sub_eu)

        return sub_eus

    def _compute_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """Compute centroid of embeddings"""
        if not embeddings:
            return [0.0] * 1536
        n = len(embeddings)
        dim = len(embeddings[0])
        centroid = [sum(emb[i] for emb in embeddings) / n for i in range(dim)]
        return centroid

    async def _get_topology_counts(self, event_id: str) -> Tuple[int, int]:
        """Get corroboration/contradiction counts from existing topology"""
        async with self.neo4j.session() as session:
            result = await session.run("""
                MATCH (e:Event {id: $event_id})<-[:BELONGS_TO]-(c1:Claim)
                MATCH (e)<-[:BELONGS_TO]-(c2:Claim)
                WHERE c1.id < c2.id
                OPTIONAL MATCH (c1)-[r:RELATES_TO]-(c2)
                RETURN
                    sum(CASE WHEN r.relation_type = 'corroboration' THEN 1 ELSE 0 END) as corr,
                    sum(CASE WHEN r.relation_type = 'contradiction' THEN 1 ELSE 0 END) as contra
            """, event_id=event_id)
            record = await result.single()
            return record['corr'] or 0, record['contra'] or 0

    def compare_metrics(self, old: ExistingEvent, new_event: ProposedEU, sub_events: List[ProposedEU]) -> Dict:
        """Compare old and new metrics"""
        return {
            'event_id': old.id,
            'event_name': old.canonical_name,
            'old_metrics': {
                'coherence': old.coherence,
                'claim_count': old.claim_count,
                'page_count': old.page_count
            },
            'new_metrics': {
                'coherence': new_event.coherence(),
                'mass': new_event.mass(),
                'tension': new_event.tension(),
                'state': new_event.state(),
                'claim_count': len(new_event.claim_ids),
                'page_count': len(new_event.page_ids),
                'sub_event_count': len(sub_events)
            },
            'structure': {
                'level_0_count': sum(len(sub.children) for sub in sub_events),
                'level_1_count': len(sub_events),
                'level_2_count': 1,
                'avg_claims_per_sub': len(new_event.claim_ids) / len(sub_events) if sub_events else 0
            },
            'compatibility': {
                'claim_preserved': len(new_event.claim_ids) == old.claim_count,
                'page_preserved': len(new_event.page_ids) == old.page_count,
                'coherence_similar': abs(new_event.coherence() - old.coherence) < 0.1
            }
        }


async def main():
    print("=" * 70)
    print("MIGRATION DRY RUN: Event → EU Structure")
    print("=" * 70)

    pg_pool = await get_pg_pool()
    neo4j_driver = get_neo4j_driver()

    migrator = MigrationDryRun(pg_pool, neo4j_driver)

    # Get sample events to migrate
    async with neo4j_driver.session() as session:
        result = await session.run("""
            MATCH (e:Event)<-[:BELONGS_TO]-(c:Claim)
            WITH e, count(c) as claim_count
            WHERE claim_count >= 10
            RETURN e.id as id, e.canonical_name as name, claim_count
            ORDER BY claim_count DESC
            LIMIT 5
        """)
        events_to_migrate = [record async for record in result]

    print(f"\n📋 Found {len(events_to_migrate)} events to migrate")

    results = []
    for record in events_to_migrate:
        try:
            # Load existing event
            existing = await migrator.load_existing_event(record['id'])

            # Migrate to EU structure
            event_eu, sub_events = await migrator.migrate_event_to_eu(existing)

            if event_eu:
                # Compare metrics
                comparison = migrator.compare_metrics(existing, event_eu, sub_events)
                results.append(comparison)

                # Print summary
                print(f"\n   ✅ Migration successful:")
                print(f"      Old coherence: {comparison['old_metrics']['coherence']:.2f}")
                print(f"      New coherence: {comparison['new_metrics']['coherence']:.2f}")
                print(f"      New mass: {comparison['new_metrics']['mass']:.2f}")
                print(f"      New state: {comparison['new_metrics']['state']}")
                print(f"      Sub-events: {comparison['structure']['level_1_count']}")
                print(f"      Claims preserved: {comparison['compatibility']['claim_preserved']}")

        except Exception as e:
            print(f"   ❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)

    if results:
        preserved = sum(1 for r in results if r['compatibility']['claim_preserved'])
        coherence_similar = sum(1 for r in results if r['compatibility']['coherence_similar'])

        print(f"\n📊 Results:")
        print(f"   Events migrated: {len(results)}")
        print(f"   Claims preserved: {preserved}/{len(results)} ({100*preserved/len(results):.0f}%)")
        print(f"   Coherence similar: {coherence_similar}/{len(results)} ({100*coherence_similar/len(results):.0f}%)")

        avg_sub_events = sum(r['structure']['level_1_count'] for r in results) / len(results)
        print(f"   Avg sub-events per event: {avg_sub_events:.1f}")

        avg_claims_per_sub = sum(r['structure']['avg_claims_per_sub'] for r in results) / len(results)
        print(f"   Avg claims per sub-event: {avg_claims_per_sub:.1f}")

        # New metrics summary
        print(f"\n📈 New Metrics Distribution:")
        masses = [r['new_metrics']['mass'] for r in results]
        tensions = [r['new_metrics']['tension'] for r in results]
        states = [r['new_metrics']['state'] for r in results]

        print(f"   Mass range: {min(masses):.2f} - {max(masses):.2f}")
        print(f"   Tension range: {min(tensions):.2f} - {max(tensions):.2f}")
        print(f"   States: {sum(1 for s in states if s == 'STABLE')} STABLE, {sum(1 for s in states if s == 'ACTIVE')} ACTIVE")

    # Save detailed results
    with open('/app/test_eu/migration_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to migration_results.json")

    await pg_pool.close()
    await neo4j_driver.close()


if __name__ == "__main__":
    asyncio.run(main())
