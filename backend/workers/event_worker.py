"""
Event Worker - Multi-Pass Clustering for Event Formation

Implements recursive multi-pass clustering to form events from claims.

Algorithm (TODO.md Phase 4):
- Pass 1: Tight temporal clusters (2-day window + 2+ entity overlap)
- Pass 2: Bridge temporal gaps (14-day relaxed + 4+ entity overlap)
- Pass 3: Transitive merging (graph DFS for connected components)

Design principles:
- Agnostic: Let data speak through entity/temporal signals
- Incremental: New claims trigger re-clustering on affected components
- Conservative: Require minimum entity overlap to avoid spurious merges
"""
import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import asyncpg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Clustering parameters (from TODO.md)
PASS1_TIME_WINDOW_DAYS = 2  # Tight temporal window
PASS1_MIN_ENTITY_OVERLAP = 2  # Minimum shared entities

PASS2_TIME_WINDOW_DAYS = 14  # Relaxed temporal window
PASS2_MIN_ENTITY_OVERLAP = 4  # Higher entity requirement


class EventWorker:
    """
    Worker that forms events from claims using multi-pass clustering
    """

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id

    async def start(self):
        """Start worker loop"""
        logger.info(f"📊 event-worker-{self.worker_id} started (multi-pass clustering)")

        try:
            while True:
                try:
                    # For now, just poll for new claims and trigger clustering
                    # TODO: Make this event-driven via job queue
                    await asyncio.sleep(10)  # Poll every 10 seconds

                    await self.run_clustering_cycle()

                except Exception as e:
                    logger.error(f"❌ Clustering cycle error: {e}", exc_info=True)
                    await asyncio.sleep(5)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

    async def run_clustering_cycle(self):
        """
        Run a full 3-pass clustering cycle on all claims

        This is a simple initial implementation that re-clusters everything.
        Later: make incremental (only re-cluster affected components)
        """
        async with self.db_pool.acquire() as conn:
            # Fetch all claims with their entities
            claims = await self._fetch_claims_with_entities(conn)

            if not claims:
                logger.debug("No claims to cluster")
                return

            logger.info(f"🔍 Clustering {len(claims)} claims...")

            # Pass 1: Tight temporal clustering
            clusters_pass1 = await self._cluster_pass1(claims)
            logger.info(f"✅ Pass 1: {len(clusters_pass1)} tight clusters")

            # Pass 2: Bridge temporal gaps
            clusters_pass2 = await self._cluster_pass2(clusters_pass1, claims)
            logger.info(f"✅ Pass 2: {len(clusters_pass2)} bridged clusters")

            # Pass 3: Transitive merging
            final_clusters = await self._cluster_pass3(clusters_pass2)
            logger.info(f"✅ Pass 3: {len(final_clusters)} final events")

            # Materialize events
            await self._materialize_events(conn, final_clusters, claims)

    async def _fetch_claims_with_entities(self, conn: asyncpg.Connection) -> List[Dict]:
        """
        Fetch all claims with their associated entities

        Returns list of dicts with claim metadata + entity list
        """
        rows = await conn.fetch("""
            SELECT
                c.id,
                c.page_id,
                c.text,
                c.event_time,
                c.confidence,
                c.created_at,
                COALESCE(
                    array_agg(ce.entity_id) FILTER (WHERE ce.entity_id IS NOT NULL),
                    ARRAY[]::uuid[]
                ) as entity_ids
            FROM core.claims c
            LEFT JOIN core.claim_entities ce ON c.id = ce.claim_id
            WHERE c.event_time IS NOT NULL  -- Only cluster claims with temporal info
            GROUP BY c.id
            ORDER BY c.event_time
        """)

        claims = []
        for row in rows:
            claims.append({
                'id': row['id'],
                'page_id': row['page_id'],
                'text': row['text'],
                'event_time': row['event_time'],
                'confidence': row['confidence'],
                'created_at': row['created_at'],
                'entity_ids': set(row['entity_ids'])  # Convert to set for fast overlap
            })

        return claims

    async def _cluster_pass1(self, claims: List[Dict]) -> List[Set[uuid.UUID]]:
        """
        Pass 1: Tight temporal clustering (2-day window + 2+ entity overlap)

        Returns list of claim ID sets (each set is a cluster)
        """
        clusters = []

        # Sort claims by event_time
        sorted_claims = sorted(claims, key=lambda c: c['event_time'])

        # Sliding window approach
        for i, claim in enumerate(sorted_claims):
            cluster = {claim['id']}
            claim_entities = claim['entity_ids']

            # Look ahead within time window
            for j in range(i + 1, len(sorted_claims)):
                other_claim = sorted_claims[j]

                # Check time window
                time_diff = (other_claim['event_time'] - claim['event_time']).days
                if time_diff > PASS1_TIME_WINDOW_DAYS:
                    break  # Beyond window

                # Check entity overlap
                shared_entities = claim_entities & other_claim['entity_ids']
                if len(shared_entities) >= PASS1_MIN_ENTITY_OVERLAP:
                    cluster.add(other_claim['id'])

            if len(cluster) > 1:
                clusters.append(cluster)

        # Merge overlapping clusters
        return self._merge_overlapping_clusters(clusters)

    async def _cluster_pass2(
        self, pass1_clusters: List[Set[uuid.UUID]], claims: List[Dict]
    ) -> List[Set[uuid.UUID]]:
        """
        Pass 2: Bridge temporal gaps (14-day window + 4+ entity overlap)

        Takes Pass 1 clusters and attempts to merge them using relaxed constraints
        """
        # Build claim index for fast lookup
        claim_index = {c['id']: c for c in claims}

        # Get aggregate info for each cluster
        cluster_info = []
        for cluster in pass1_clusters:
            cluster_claims = [claim_index[cid] for cid in cluster]

            # Aggregate temporal bounds
            min_time = min(c['event_time'] for c in cluster_claims)
            max_time = max(c['event_time'] for c in cluster_claims)

            # Aggregate entities (union)
            all_entities = set()
            for c in cluster_claims:
                all_entities.update(c['entity_ids'])

            cluster_info.append({
                'claim_ids': cluster,
                'min_time': min_time,
                'max_time': max_time,
                'entities': all_entities
            })

        # Try to merge clusters
        merged = []
        for i, cluster_a in enumerate(cluster_info):
            merged_cluster = cluster_a['claim_ids'].copy()

            for j, cluster_b in enumerate(cluster_info):
                if i >= j:  # Skip self and already compared
                    continue

                # Check temporal proximity (14-day window between cluster bounds)
                time_gap = min(
                    abs((cluster_a['min_time'] - cluster_b['max_time']).days),
                    abs((cluster_b['min_time'] - cluster_a['max_time']).days)
                )

                if time_gap > PASS2_TIME_WINDOW_DAYS:
                    continue

                # Check entity overlap (4+ shared entities)
                shared_entities = cluster_a['entities'] & cluster_b['entities']
                if len(shared_entities) >= PASS2_MIN_ENTITY_OVERLAP:
                    merged_cluster.update(cluster_b['claim_ids'])

            merged.append(merged_cluster)

        return self._merge_overlapping_clusters(merged)

    async def _cluster_pass3(self, pass2_clusters: List[Set[uuid.UUID]]) -> List[Set[uuid.UUID]]:
        """
        Pass 3: Transitive merging via graph DFS

        Build a graph where clusters are nodes, edges when they share claims
        Then extract connected components
        """
        # Build adjacency graph
        graph = defaultdict(set)

        for i, cluster_a in enumerate(pass2_clusters):
            for j, cluster_b in enumerate(pass2_clusters):
                if i >= j:
                    continue

                # If clusters share any claims, they should be merged
                if cluster_a & cluster_b:
                    graph[i].add(j)
                    graph[j].add(i)

        # DFS to find connected components
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(len(pass2_clusters)):
            if i not in visited:
                component = set()
                dfs(i, component)

                # Merge all clusters in this component
                merged = set()
                for cluster_idx in component:
                    merged.update(pass2_clusters[cluster_idx])

                components.append(merged)

        return components

    def _merge_overlapping_clusters(self, clusters: List[Set[uuid.UUID]]) -> List[Set[uuid.UUID]]:
        """
        Merge clusters that share any claims (transitive closure)
        """
        if not clusters:
            return []

        # Build union-find structure
        merged = []
        visited = [False] * len(clusters)

        for i in range(len(clusters)):
            if visited[i]:
                continue

            # Start new merged cluster
            current = clusters[i].copy()
            visited[i] = True

            # Keep merging until no more overlaps found
            changed = True
            while changed:
                changed = False
                for j in range(len(clusters)):
                    if visited[j]:
                        continue

                    if current & clusters[j]:  # Overlap found
                        current.update(clusters[j])
                        visited[j] = True
                        changed = True

            merged.append(current)

        return merged

    async def _materialize_events(
        self, conn: asyncpg.Connection, clusters: List[Set[uuid.UUID]],
        claims: List[Dict]
    ):
        """
        Create/update event records from final clusters
        """
        claim_index = {c['id']: c for c in claims}

        for cluster_claims in clusters:
            if not cluster_claims:
                continue

            # Aggregate cluster metadata
            cluster_claim_objs = [claim_index[cid] for cid in cluster_claims]

            # Temporal bounds
            event_start = min(c['event_time'] for c in cluster_claim_objs)
            event_end = max(c['event_time'] for c in cluster_claim_objs)

            # Aggregate entities
            all_entities = set()
            for c in cluster_claim_objs:
                all_entities.update(c['entity_ids'])

            # Average confidence
            avg_confidence = sum(c['confidence'] for c in cluster_claim_objs) / len(cluster_claim_objs)

            # Get page IDs
            page_ids = {c['page_id'] for c in cluster_claim_objs if c['page_id']}

            # Create event (simple placeholder title for now)
            event_title = f"Event ({event_start.strftime('%Y-%m-%d')})"

            logger.info(
                f"💾 Creating event: {len(cluster_claims)} claims, "
                f"{len(all_entities)} entities, {len(page_ids)} pages"
            )

            # Insert event
            event_id = await conn.fetchval("""
                INSERT INTO core.events (
                    title, summary, event_start, event_end,
                    confidence, claims_count, pages_count,
                    event_scale, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                event_title,
                f"{len(cluster_claims)} claims from {len(page_ids)} sources",
                event_start,
                event_end,
                avg_confidence,
                len(cluster_claims),
                len(page_ids),
                'micro',  # Default scale for now
                'active'
            )

            # Link entities to event
            for entity_id in all_entities:
                await conn.execute("""
                    INSERT INTO core.event_entities (event_id, entity_id, relevance_score)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (event_id, entity_id) DO NOTHING
                """, event_id, entity_id, 0.5)

            # Link pages to event
            for page_id in page_ids:
                await conn.execute("""
                    INSERT INTO core.page_events (page_id, event_id, relevance_score)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (page_id, event_id) DO NOTHING
                """, page_id, event_id, 0.5)


async def main():
    """Main worker entry point"""
    worker_id = int(os.getenv('WORKER_ID', '1'))

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    worker = EventWorker(db_pool, job_queue, worker_id=worker_id)
    logger.info(f"📊 Starting Event worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
