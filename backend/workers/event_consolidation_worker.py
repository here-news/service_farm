"""
Event Consolidation Worker - Periodic Bayesian Re-evaluation

Runs periodically to:
1. Re-score all event pairs with accumulated evidence
2. Merge events that should now be attached (posterior > threshold)
3. Update confidence based on corroboration
4. Promote event status (provisional → emerging → stable)

This implements the Bayesian posterior update: as we gather more pages and claims,
we re-evaluate whether events should be merged.
"""
import asyncio
import logging
import os
import uuid
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta

import asyncpg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from workers.event_attachment import EventAttachmentScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventConsolidationWorker:
    """
    Periodic re-evaluation of events with Bayesian posterior updates
    """

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.scorer = EventAttachmentScorer()

    async def start(self):
        """Start periodic consolidation loop"""
        logger.info(f"🔄 event-consolidation-worker-{self.worker_id} started")

        try:
            while True:
                try:
                    await self.run_consolidation_pass()
                    await asyncio.sleep(600)  # Run every 10 minutes

                except Exception as e:
                    logger.error(f"❌ Consolidation worker error: {e}", exc_info=True)
                    await asyncio.sleep(30)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

    async def run_consolidation_pass(self):
        """
        Main consolidation pass:
        1. Find all provisional/emerging events
        2. Re-score pairs to find merge candidates
        3. Merge events with improved scores
        4. Update status based on accumulated evidence
        """
        async with self.db_pool.acquire() as conn:
            # Get all active events (provisional, emerging, stable)
            events = await conn.fetch("""
                SELECT
                    e.id, e.title, e.status, e.event_start, e.event_end,
                    e.confidence, e.claims_count, e.pages_count, e.event_scale,
                    e.embedding,
                    e.created_at, e.updated_at
                FROM core.events e
                WHERE e.status IN ('provisional', 'emerging', 'stable')
                  AND e.created_at > NOW() - INTERVAL '7 days'
                ORDER BY e.pages_count DESC, e.claims_count DESC
            """)

            if len(events) < 2:
                logger.info("✓ Less than 2 events, nothing to consolidate")
                return

            logger.info(f"🔍 Evaluating {len(events)} events for consolidation")

            # Track merges performed
            merges_performed = 0

            # Compare events pairwise (only within similar time windows)
            for i, event_a in enumerate(events):
                for event_b in events[i+1:]:
                    # Skip if too far apart temporally (> 3 days)
                    if event_a['event_start'] and event_b['event_start']:
                        time_diff = abs((event_a['event_start'] - event_b['event_start']).days)
                        if time_diff > 3:
                            continue

                    # Re-score with current data
                    score_result = await self._should_merge_events(conn, event_a, event_b)

                    # Debug logging
                    logger.info(
                        f"📊 Score: {score_result['total_score']:.3f} | "
                        f"'{event_b['title'][:40]}...' vs '{event_a['title'][:40]}...' | "
                        f"Decision: {score_result['decision']}"
                    )

                    if score_result['should_merge']:
                        logger.info(
                            f"🔗 Merging events: "
                            f"'{event_b['title'][:50]}' → '{event_a['title'][:50]}'"
                        )
                        await self._merge_events(conn, target=event_a['id'], source=event_b['id'])
                        merges_performed += 1

            # Update event status based on accumulated evidence
            await self._update_event_status(conn)

            if merges_performed > 0:
                logger.info(f"✅ Consolidation pass complete: {merges_performed} events merged")
            else:
                logger.info("✓ Consolidation pass complete: no merges needed")

    async def _should_merge_events(
        self,
        conn: asyncpg.Connection,
        event_a: Dict,
        event_b: Dict
    ) -> Dict:
        """
        Re-score two events to see if they should be merged

        Returns dict with score_result and should_merge decision
        """
        # Get entities for both events
        entities_a = await conn.fetch("""
            SELECT e.canonical_name
            FROM core.entities e
            JOIN core.event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = $1
        """, event_a['id'])

        entities_b = await conn.fetch("""
            SELECT e.canonical_name
            FROM core.entities e
            JOIN core.event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = $1
        """, event_b['id'])

        entity_set_a = {e['canonical_name'] for e in entities_a}
        entity_set_b = {e['canonical_name'] for e in entities_b}

        # Get claims for event_b (the one we're testing)
        claims_b = await conn.fetch("""
            SELECT c.text, c.event_time, c.modality, c.confidence
            FROM core.claims c
            JOIN core.page_events pe ON c.page_id = pe.page_id
            WHERE pe.event_id = $1
            LIMIT 10
        """, event_b['id'])

        # Convert to dicts
        page_claims = [dict(c) for c in claims_b]

        # Score using attachment scorer
        score_result = self.scorer.score_page_to_event(
            page_embedding=self._parse_embedding(event_b['embedding']),
            page_entities=entity_set_b,
            page_time=event_b['event_start'],
            page_claims=page_claims,
            event=dict(event_a),
            event_entities=entity_set_a,
            event_claims=[]  # Not needed for basic scoring
        )

        # Return full result for debugging
        return {
            **score_result,
            'should_merge': score_result['decision'] == 'attach'
        }

    def _parse_embedding(self, emb) -> List[float]:
        """Parse embedding from pgvector format"""
        if not emb:
            return []
        if isinstance(emb, list):
            return emb
        # Parse pgvector string format "[0.1,0.2,...]"
        emb_str = str(emb).strip('[]')
        return [float(x) for x in emb_str.split(',') if x.strip()]

    async def _merge_events(
        self,
        conn: asyncpg.Connection,
        target: uuid.UUID,
        source: uuid.UUID
    ):
        """
        Merge source event into target event

        1. Move all page_events references
        2. Move all entity references
        3. Update target claims_count, pages_count
        4. Mark source as archived
        """
        async with conn.transaction():
            # Move page_events (insert new references, avoid duplicates)
            await conn.execute("""
                INSERT INTO core.page_events (page_id, event_id)
                SELECT page_id, $1
                FROM core.page_events
                WHERE event_id = $2
                ON CONFLICT (page_id, event_id) DO NOTHING
            """, target, source)

            # Delete old page_events references
            await conn.execute("""
                DELETE FROM core.page_events
                WHERE event_id = $1
            """, source)

            # Move entity references
            await conn.execute("""
                INSERT INTO core.event_entities (event_id, entity_id)
                SELECT $1, entity_id
                FROM core.event_entities
                WHERE event_id = $2
                ON CONFLICT (event_id, entity_id) DO NOTHING
            """, target, source)

            # Delete old entity references
            await conn.execute("""
                DELETE FROM core.event_entities
                WHERE event_id = $1
            """, source)

            # Update target event counts and confidence
            await conn.execute("""
                UPDATE core.events
                SET
                    pages_count = (SELECT COUNT(DISTINCT page_id) FROM core.page_events WHERE event_id = $1),
                    claims_count = (SELECT COUNT(DISTINCT c.id)
                                    FROM core.claims c
                                    JOIN core.page_events pe ON c.page_id = pe.page_id
                                    WHERE pe.event_id = $1),
                    confidence = LEAST(confidence + 0.15, 0.95),
                    updated_at = NOW()
                WHERE id = $1
            """, target)

            # Archive source event
            await conn.execute("""
                UPDATE core.events
                SET status = 'archived',
                    updated_at = NOW()
                WHERE id = $1
            """, source)

    async def _update_event_status(self, conn: asyncpg.Connection):
        """
        Update event status based on accumulated evidence:
        - provisional → emerging (2+ pages)
        - emerging → stable (5+ pages)
        """
        await conn.execute("""
            UPDATE core.events
            SET status = CASE
                WHEN status = 'provisional' AND pages_count >= 2 THEN 'emerging'
                WHEN status = 'emerging' AND pages_count >= 5 THEN 'stable'
                ELSE status
            END,
            updated_at = NOW()
            WHERE status IN ('provisional', 'emerging')
              AND (
                  (status = 'provisional' AND pages_count >= 2) OR
                  (status = 'emerging' AND pages_count >= 5)
              )
        """)


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

    worker = EventConsolidationWorker(db_pool, job_queue, worker_id=worker_id)
    logger.info(f"🔄 Starting event consolidation worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
