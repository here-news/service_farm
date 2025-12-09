"""
Event Worker - Recursive Event Formation

Simple worker that uses EventService for recursive event formation.

Architecture:
- Listens to event_queue for page_ids
- Fetches claims using ClaimRepository
- Hydrates entities for claims
- Uses EventService to form events (create_root_event or examine_claims)
- EventService handles all LLM-based naming, matching, and recursive logic
"""
import asyncio
import json
import logging
import os
import uuid
from typing import Optional, List

import asyncpg
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from services.event_service import EventService
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from repositories.event_repository import EventRepository
from repositories.page_repository import PageRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventWorker:
    """
    Simple event worker using EventService for recursive event formation
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        neo4j_service: Neo4jService,
        job_queue: JobQueue,
        worker_id: int = 1
    ):
        self.db_pool = db_pool
        self.neo4j = neo4j_service
        self.job_queue = job_queue
        self.worker_id = worker_id

        # Initialize repositories and service
        self.claim_repo = ClaimRepository(db_pool, neo4j_service)
        self.entity_repo = EntityRepository(db_pool, neo4j_service)
        self.event_repo = EventRepository(db_pool, neo4j_service)
        self.event_service = EventService(
            event_repo=self.event_repo,
            claim_repo=self.claim_repo,
            entity_repo=self.entity_repo
        )

    async def start(self):
        """Start worker loop"""
        logger.info(f"📊 event-worker-{self.worker_id} started")

        try:
            while True:
                try:
                    # Listen to job queue for new pages
                    job = await self.job_queue.dequeue('queue:event:high', timeout=5)

                    if job:
                        page_id = job['page_id']
                        await self.process_page(page_id)

                except Exception as e:
                    logger.error(f"❌ Event worker error: {e}", exc_info=True)
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

    async def process_page(self, page_id: uuid.UUID):
        """
        Process page and create/attach to event using EventService

        Steps:
        1. Fetch claims for page
        2. Hydrate entities for claims
        3. Collect all entity IDs
        4. Find candidate events (if any)
        5. If good match: examine_claims() to attach or create sub-events
        6. If no match: create_root_event()
        """
        logger.info(f"📄 Processing page {page_id}")

        # 1. Fetch claims
        claims = await self.claim_repo.get_by_page(page_id)

        if not claims:
            logger.warning(f"⚠️  No claims found for page {page_id}")
            return

        logger.info(f"📝 Found {len(claims)} claims")

        # 2. Hydrate entities for claims
        for claim in claims:
            await self.claim_repo.hydrate_entities(claim)

        # Count entities
        all_entity_ids = set()
        for claim in claims:
            all_entity_ids.update(claim.entity_ids)

        logger.info(f"🎯 Claims reference {len(all_entity_ids)} unique entities")

        # 3. Generate embedding from claims for semantic similarity matching
        page_embedding = await self._generate_page_embedding_from_claims(claims)
        if page_embedding:
            logger.info(f"📊 Generated page embedding from {len(claims)} claims")
        else:
            logger.warning(f"⚠️  Failed to generate page embedding")

        # 4. Find candidate events (using entities + embeddings + time)
        reference_time = claims[0].event_time if claims and claims[0].event_time else None

        candidates = await self.event_repo.find_candidates(
            entity_ids=all_entity_ids,
            reference_time=reference_time,
            time_window_days=7,
            page_embedding=page_embedding
        )

        # 5. Decision: attach to existing event or create new
        if candidates:
            best_event, best_score = candidates[0]
            logger.info(f"🔍 Best candidate: {best_event.canonical_name} (score: {best_score:.2f})")

            # Threshold 0.25 to allow related articles through - borderline cases (e.g. 0.29)
            # should still be examined rather than creating duplicate root events
            if best_score > 0.25:
                # Potential match - examine claims to determine relationship
                logger.info(f"🎯 Examining claims against existing event...")
                result = await self.event_service.examine_claims(best_event, claims)

                logger.info(f"📊 Examination Results:")
                logger.info(f"   ✅ Claims added: {len(result.claims_added)}")
                logger.info(f"   🌿 Sub-events created: {len(result.sub_events_created)}")
                logger.info(f"   ❌ Claims rejected: {len(result.claims_rejected)}")

                if result.sub_events_created:
                    for sub_event in result.sub_events_created:
                        logger.info(f"   🌿 Created sub-event: {sub_event.canonical_name}")
            else:
                # Very low match score - likely unrelated
                logger.info(f"⚠️  Match score {best_score:.2f} < 0.25, creating new root event")
                event = await self.event_service.create_root_event(claims)
                logger.info(f"✨ Created root event: {event.canonical_name}")
        else:
            # No candidates - create new root event
            logger.info(f"📝 No candidate events found, creating new root event")
            event = await self.event_service.create_root_event(claims)
            logger.info(f"✨ Created root event: {event.canonical_name}")

    async def _generate_page_embedding_from_claims(self, claims: List) -> Optional[List[float]]:
        """
        Generate embedding for page from its claims

        Strategy: Combine claim texts into coherent description of page content,
        then generate semantic embedding. This represents "what is this page about?"

        Similar to event embedding generation - semantic representation enables
        matching between new pages and existing events.

        Args:
            claims: List of claims from the page

        Returns:
            Embedding vector or None if generation fails
        """
        if not claims:
            return None

        # Combine claim texts (limit to ~8k chars to avoid token limits)
        claim_texts = [c.text for c in claims if c.text]
        if not claim_texts:
            return None

        # Concatenate claims with newlines (truncate if too long)
        combined_text = "\n".join(claim_texts)
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."

        try:
            # Use OpenAI embedding API (same as EventService)
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate page embedding: {e}")
            return None


async def main():
    """Main entry point"""
    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=10
    )

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    # Connect to Redis job queue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Start worker
    worker = EventWorker(
        db_pool=db_pool,
        neo4j_service=neo4j,
        job_queue=job_queue,
        worker_id=1
    )

    try:
        await worker.start()
    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
