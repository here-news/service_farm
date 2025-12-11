"""
Event Worker - LiveEvent Pool Manager

Layered Architecture:

Layer 1: EventWorker
    - Monitors queue:event:high for knowledge_complete signal (page_id)
    - Fetches all claims for the page (one article = one context)
    - Fetches pre-computed page embedding from PostgreSQL
    - Passes all claims + embedding to pool

Layer 2: LiveEventPool
    - Routes page claims to appropriate living event
    - Uses page embedding + entities + time for multi-signal matching
    - Activates event (loads from storage if hibernated)
    - Manages pool of active events
    - Runs periodic metabolism cycles

Layer 3: LiveEvent
    - Examines all page claims via own metabolism
    - Decides: accept / reject / create sub-event
    - Updates internal state
    - Regenerates narrative when needed
    - Hibernates when dormant (24h+ idle)

Flow: KnowledgeWorker → queue:event:high → EventWorker → LiveEventPool → LiveEvent
"""
import asyncio
import json
import logging
import os
from typing import Optional, List

import asyncpg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from services.event_service import EventService
from services.live_event_pool import LiveEventPool
from services.claim_topology import ClaimTopologyService
from repositories.claim_repository import ClaimRepository
from repositories.entity_repository import EntityRepository
from repositories.event_repository import EventRepository
from repositories.page_repository import PageRepository

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventWorker:
    """
    Event worker that manages a pool of living event organisms.

    Each worker maintains an in-memory pool of active events that:
    - Bootstrap from new claims
    - Hydrate state from storage
    - Execute metabolism (examine claims, update narratives)
    - Hibernate when dormant (24h+ idle)

    Scale workers via docker-compose --scale for parallel processing.
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

        # Initialize repositories
        self.page_repo = PageRepository(db_pool, neo4j_service)
        self.claim_repo = ClaimRepository(db_pool, neo4j_service)
        self.entity_repo = EntityRepository(db_pool, neo4j_service)
        self.event_repo = EventRepository(db_pool, neo4j_service)
        self.event_service = EventService(
            event_repo=self.event_repo,
            claim_repo=self.claim_repo,
            entity_repo=self.entity_repo
        )

        # Initialize ClaimTopologyService for Bayesian plausibility analysis
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.topology_service = ClaimTopologyService(openai_client)

        # Initialize LiveEvent pool with topology service
        self.pool = LiveEventPool(
            event_service=self.event_service,
            claim_repo=self.claim_repo,
            event_repo=self.event_repo,
            entity_repo=self.entity_repo,
            topology_service=self.topology_service
        )

    async def start(self):
        """
        Start worker loop.

        Listens to two queues:
        1. queue:event:high - knowledge_complete signal (page_id) for new claims
        2. queue:event:command - commands for living events (e.g., /retopologize)

        Commands are processed with higher priority (checked first, non-blocking).
        """
        logger.info(f"📊 event-worker-{self.worker_id} started")

        # Start metabolism cycle in background
        asyncio.create_task(self._metabolism_loop())

        try:
            while True:
                try:
                    # Check command queue first (non-blocking, higher priority)
                    cmd = await self.job_queue.dequeue_nonblocking(
                        self.job_queue.COMMAND_QUEUE
                    )
                    if cmd:
                        await self.process_command(cmd)
                        continue  # Check for more commands before processing pages

                    # Listen for knowledge_complete signal with page_id
                    job = await self.job_queue.dequeue('queue:event:high', timeout=5)

                    if job:
                        page_id = job['page_id']
                        await self.process_page(page_id)

                except Exception as e:
                    logger.error(f"❌ Event worker error: {e}", exc_info=True)
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")

    async def process_command(self, cmd: dict):
        """
        Process a command sent to a living event.

        Commands are MCP-like paths that instruct events to perform actions.

        Supported commands:
        - /retopologize: Re-run Bayesian topology analysis and regenerate narrative
        - /regenerate: Regenerate narrative (without full topology re-analysis)
        - /hibernate: Force hibernate the event
        - /status: Log current event status

        Args:
            cmd: Command dict with 'event_id', 'command', 'params'
        """
        event_id = cmd.get('event_id')
        command = cmd.get('command', '').strip()
        params = cmd.get('params', {})

        logger.info(f"📨 Command received: {command} for {event_id}")

        if not event_id or not command:
            logger.warning(f"⚠️ Invalid command: missing event_id or command")
            return

        # Route to pool for handling
        result = await self.pool.handle_command(event_id, command, params)

        if result.get('success'):
            logger.info(f"✅ Command {command} completed for {event_id}")
        else:
            logger.warning(f"⚠️ Command {command} failed: {result.get('error')}")

    async def process_page(self, page_id: str):
        """
        Process page - all claims together as one context.

        Layer 1: Prepare page context and pass to pool

        Steps:
        1. Fetch all claims for page
        2. Hydrate entities for each claim
        3. Fetch pre-computed page embedding (generated by KnowledgeWorker)
        4. Pass all claims + embedding to pool (Layer 2 routes to appropriate event)
        """
        logger.info(f"📄 Processing page {page_id}")

        # 1. Fetch all claims for this page
        claims = await self.claim_repo.get_by_page(page_id)

        if not claims:
            logger.warning(f"⚠️  No claims found for page {page_id}")
            return

        logger.info(f"📝 Found {len(claims)} claims")

        # 2. Hydrate entities for all claims
        for claim in claims:
            await self.claim_repo.hydrate_entities(claim)

        # Count unique entities
        all_entity_ids = set()
        for claim in claims:
            all_entity_ids.update(claim.entity_ids)

        logger.info(f"🎯 Claims reference {len(all_entity_ids)} unique entities")

        # 3. Fetch page embedding (pre-computed by KnowledgeWorker)
        page_embedding = await self._fetch_page_embedding(page_id)

        if page_embedding:
            logger.info(f"📊 Fetched page embedding")
        else:
            logger.warning(f"⚠️  No page embedding found - will match without semantic signal")

        # 4. Pass to pool - Layer 2 routes to appropriate event
        # All claims go together (same article = same context)
        await self.pool.route_page_claims(claims, page_embedding)

    async def _fetch_page_embedding(self, page_id: str) -> Optional[List[float]]:
        """
        Fetch pre-computed page embedding via PageRepository.

        Embedding was generated by KnowledgeWorker during STAGE 4e.
        Uses domain model access pattern - no direct DB queries.
        """
        return await self.page_repo.get_embedding(page_id)

    async def _metabolism_loop(self):
        """
        Periodic metabolism cycle for living events.

        Runs every hour to:
        - Regenerate narratives for active events
        - Hibernate dormant events
        """
        logger.info(f"🔄 Metabolism loop started (runs every 1h)")

        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                logger.info(f"⏰ Running metabolism cycle...")
                await self.pool.metabolism_cycle()
                logger.info(f"✅ Metabolism cycle complete")

            except Exception as e:
                logger.error(f"❌ Metabolism loop error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 min before retry


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
