"""
Entity Merge Worker - Consolidate Duplicate Entities

Runs after Wikidata enrichment to merge entities that share the same QID.
This handles cases where semantic extraction creates variants like:
- "Police" and "Hong Kong Police" (both → Q25859)
- "Fire Department" and "Fire Services Department" (both → Q1595073)

Algorithm:
1. Find all entities with duplicate Wikidata QIDs
2. For each group, pick the canonical entity (most mentions + best name)
3. Merge all references (claim_entities, event_entities)
4. Sum mention counts
5. Delete duplicates

Triggers:
- After each Wikidata enrichment batch
- Periodic cleanup (can run as cron job)
"""
import asyncio
import logging
import os
import uuid
from typing import List, Dict, Tuple
from datetime import datetime

import asyncpg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityMergeWorker:
    """
    Worker that merges duplicate entities sharing the same Wikidata QID
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        job_queue: JobQueue,
        neo4j_service: Neo4jService,
        worker_id: int = 1
    ):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.neo4j = neo4j_service
        self.worker_id = worker_id

    async def start(self):
        """Start worker loop"""
        logger.info(f"🔗 entity-merge-worker-{self.worker_id} started")

        try:
            while True:
                try:
                    # Listen for merge jobs (triggered after Wikidata enrichment)
                    job = await self.job_queue.dequeue('entity_merge', timeout=30)

                    if job:
                        await self.run_merge_pass()
                    else:
                        # No job in queue - run periodic merge anyway
                        logger.info("⏰ Running periodic merge check...")
                        await self.run_merge_pass()
                        await asyncio.sleep(300)  # Check every 5 minutes

                except Exception as e:
                    logger.error(f"❌ Merge worker error: {e}", exc_info=True)
                    await asyncio.sleep(5)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

    async def run_merge_pass(self):
        """
        Find and merge all duplicate entities using Neo4j
        """
        # Find all QIDs with multiple entities from Neo4j
        duplicates = await self.neo4j.find_duplicate_entities()

        if not duplicates:
            logger.info("✓ No duplicates found")
            return

        logger.info(f"🔍 Found {len(duplicates)} QIDs with duplicates")

        merged_count = 0
        for dup_group in duplicates:
            qid = dup_group['wikidata_qid']
            entity_ids = dup_group['entity_ids']
            names = dup_group['names']
            mention_counts = dup_group['mention_counts']

            # Pick canonical entity (most mentions, or longest name if tie)
            canonical_idx = self._pick_canonical(names, mention_counts)
            canonical_id = entity_ids[canonical_idx]
            canonical_name = names[canonical_idx]

            # IDs to merge into canonical
            duplicate_ids = [eid for i, eid in enumerate(entity_ids) if i != canonical_idx]

            logger.info(
                f"🔗 Merging {qid}: {len(duplicate_ids)} variants → '{canonical_name}'"
            )
            for i, (name, count) in enumerate(zip(names, mention_counts)):
                if i != canonical_idx:
                    logger.info(f"   ← '{name}' ({count} mentions)")

            # Perform merge in Neo4j
            total_mentions = sum(mention_counts)
            await self.neo4j.merge_entities(
                canonical_id=canonical_id,
                duplicate_ids=duplicate_ids,
                total_mentions=total_mentions
            )

            logger.info(f"✅ Merged → {canonical_name} (total: {total_mentions} mentions)")
            merged_count += len(duplicate_ids)

        logger.info(f"✓ Merge pass complete: {merged_count} entities merged")

    def _pick_canonical(self, names: List[str], mention_counts: List[int]) -> int:
        """
        Pick the best canonical entity from duplicates

        Priority:
        1. Most mentions
        2. Longest name (usually more specific)
        3. First alphabetically
        """
        # Find max mentions
        max_mentions = max(mention_counts)

        # Filter to entities with max mentions
        candidates = [
            (i, name, count)
            for i, (name, count) in enumerate(zip(names, mention_counts))
            if count == max_mentions
        ]

        # If tie, pick longest name
        if len(candidates) > 1:
            candidates.sort(key=lambda x: (len(x[1]), x[1]), reverse=True)

        return candidates[0][0]


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

    # Initialize Neo4j service
    neo4j_service = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'password')
    )
    await neo4j_service.connect()

    worker = EntityMergeWorker(db_pool, job_queue, neo4j_service, worker_id=worker_id)
    logger.info(f"🔗 Starting entity merge worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()
        await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(main())
