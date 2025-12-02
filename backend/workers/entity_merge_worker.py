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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityMergeWorker:
    """
    Worker that merges duplicate entities sharing the same Wikidata QID
    """

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
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
        Find and merge all duplicate entities
        """
        async with self.db_pool.acquire() as conn:
            # Find all QIDs with multiple entities
            duplicates = await conn.fetch("""
                SELECT
                    wikidata_qid,
                    array_agg(id) as entity_ids,
                    array_agg(canonical_name) as names,
                    array_agg(mention_count) as mention_counts
                FROM core.entities
                WHERE wikidata_qid IS NOT NULL
                  AND status IN ('enriched', 'checked')
                GROUP BY wikidata_qid
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
            """)

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

                # Perform merge
                total_mentions = await self._merge_entities(
                    conn, canonical_id, duplicate_ids, names, mention_counts
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

    async def _merge_entities(
        self,
        conn: asyncpg.Connection,
        canonical_id: uuid.UUID,
        duplicate_ids: List[uuid.UUID],
        names: List[str],
        mention_counts: List[int]
    ) -> int:
        """
        Merge duplicate entities into canonical entity

        Steps:
        1. Update claim_entities references
        2. Update event_entities references (avoid duplicates)
        3. Sum mention counts
        4. Delete duplicate entities

        Returns total mention count
        """
        async with conn.transaction():
            # Step 1: Update claim_entities
            await conn.execute("""
                UPDATE core.claim_entities
                SET entity_id = $1
                WHERE entity_id = ANY($2::uuid[])
            """, canonical_id, duplicate_ids)

            # Step 2: Update event_entities (handle duplicates)
            # First, insert canonical entity for all events that had duplicates
            await conn.execute("""
                INSERT INTO core.event_entities (event_id, entity_id)
                SELECT DISTINCT event_id, $1::uuid
                FROM core.event_entities
                WHERE entity_id = ANY($2::uuid[])
                ON CONFLICT (event_id, entity_id) DO NOTHING
            """, canonical_id, duplicate_ids)

            # Then delete old references
            await conn.execute("""
                DELETE FROM core.event_entities
                WHERE entity_id = ANY($1::uuid[])
            """, duplicate_ids)

            # Step 3: Sum mention counts
            total_mentions = sum(mention_counts)
            await conn.execute("""
                UPDATE core.entities
                SET mention_count = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, canonical_id, total_mentions)

            # Step 4: Delete duplicates
            await conn.execute("""
                DELETE FROM core.entities
                WHERE id = ANY($1::uuid[])
            """, duplicate_ids)

            return total_mentions


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

    worker = EntityMergeWorker(db_pool, job_queue, worker_id=worker_id)
    logger.info(f"🔗 Starting entity merge worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
