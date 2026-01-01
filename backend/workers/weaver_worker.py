"""
Weaver Worker - Continuous claim processing into surfaces

Architecture:
- Consumes claims from Redis queue (claims:pending)
- Finds candidate surfaces via pgvector + Neo4j
- Links claims to surfaces using identity rules
- Creates new surfaces when no match found
- Emits proto-inquiries when tensions detected

Flow: KnowledgeWorker → queue:claims:pending → WeaverWorker → Surfaces → ProtoInquiries

Key principles:
- Support affects prioritization, not acceptance thresholds
- Identity requires identity evidence (not just high support)
- Surfaces are connected components of identity edges
"""
import asyncio
import json
import logging
import os
import sys
from typing import Optional, List, Set, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from repositories.claim_repository import ClaimRepository
from repositories.surface_repository import SurfaceRepository
from models.domain.claim import Claim
from models.domain.surface import Surface
from utils.id_generator import generate_id

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LinkResult:
    """Result of linking a claim to a surface."""
    claim_id: str
    surface_id: str
    is_new_surface: bool
    similarity: float


class WeaverWorker:
    """
    Weaver worker that continuously processes claims into surfaces.

    Each claim is:
    1. Compared against candidate surfaces (by embedding + anchor)
    2. Linked to best matching surface (if above threshold)
    3. Or creates a new surface (if no match)

    The worker maintains no in-memory state - all state is in DB.
    This allows horizontal scaling with multiple workers.
    """

    QUEUE_NAME = "claims:pending"
    PROCESSED_QUEUE = "claims:processed"
    FAILED_QUEUE = "claims:failed"

    # Identity linking parameters
    SIMILARITY_THRESHOLD = 0.75  # Strict - don't lower based on support
    ANCHOR_BONUS = 0.1  # Bonus for shared anchor entities
    TIME_WINDOW_DAYS = 14

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

        # Repositories
        self.claim_repo = ClaimRepository(db_pool, neo4j_service)
        self.surface_repo = SurfaceRepository(db_pool, neo4j_service)

        # Stats
        self.claims_processed = 0
        self.surfaces_created = 0
        self.surfaces_updated = 0

        self.running = False

    async def run(self):
        """Main processing loop."""
        self.running = True
        logger.info(f"🧬 Weaver Worker {self.worker_id} starting...")

        while self.running:
            try:
                # Pull claim from queue (blocking with timeout)
                item = await self.job_queue.dequeue(
                    self.QUEUE_NAME,
                    timeout=5.0
                )

                if item is None:
                    continue

                # Parse claim ID or claim data
                claim_id = self._parse_queue_item(item)
                if not claim_id:
                    continue

                # Process claim
                result = await self.process_claim(claim_id)

                if result:
                    await self.job_queue.enqueue(
                        self.PROCESSED_QUEUE,
                        json.dumps({"claim_id": claim_id, "surface_id": result.surface_id})
                    )
                    self.claims_processed += 1

                    if self.claims_processed % 100 == 0:
                        logger.info(
                            f"📊 Progress: {self.claims_processed} claims, "
                            f"{self.surfaces_created} new surfaces, "
                            f"{self.surfaces_updated} updated"
                        )

            except Exception as e:
                logger.error(f"❌ Error processing claim: {e}", exc_info=True)
                if 'claim_id' in locals():
                    await self.job_queue.enqueue(
                        self.FAILED_QUEUE,
                        json.dumps({"claim_id": claim_id, "error": str(e)})
                    )

    def _parse_queue_item(self, item: bytes) -> Optional[str]:
        """Parse queue item to get claim ID."""
        try:
            data = json.loads(item)
            if isinstance(data, dict):
                return data.get('claim_id') or data.get('id')
            elif isinstance(data, str):
                return data
        except json.JSONDecodeError:
            # Plain string claim ID
            return item.decode() if isinstance(item, bytes) else str(item)
        return None

    async def process_claim(self, claim_id: str) -> Optional[LinkResult]:
        """
        Process a single claim into the surface lattice.

        Steps:
        1. Load claim with embedding and entities
        2. Find candidate surfaces
        3. Compute similarities
        4. Link to best match or create new surface
        5. Update surface properties

        Args:
            claim_id: Claim ID to process

        Returns:
            LinkResult with surface info
        """
        # 1. Load claim
        claim = await self.claim_repo.get_by_id(claim_id)
        if not claim:
            logger.warning(f"⚠️ Claim {claim_id} not found")
            return None

        # Load embedding
        claim.embedding = await self.claim_repo.get_embedding(claim_id)
        if not claim.embedding:
            logger.warning(f"⚠️ Claim {claim_id} has no embedding, skipping")
            return None

        # Load entities
        claim = await self.claim_repo.hydrate_entities(claim)

        # Extract anchor entities (high-IDF entities)
        claim_anchors = self._extract_anchors(claim)

        # 2. Find candidate surfaces
        candidates = await self.surface_repo.find_candidates_for_claim(
            claim,
            time_window_days=self.TIME_WINDOW_DAYS,
            limit=20
        )

        # 3. Find best matching surface
        best_surface = None
        best_similarity = 0.0

        if candidates:
            surfaces = await self.surface_repo.get_by_ids(candidates)

            for surface in surfaces:
                # Prioritize by support (check high-support first)
                # But don't lower threshold based on support
                similarity = await self._compute_similarity(claim, surface)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_surface = surface

        # 4. Link or create
        if best_surface and best_similarity >= self.SIMILARITY_THRESHOLD:
            # Link to existing surface
            best_surface.add_claim(claim)
            await self._update_surface_centroid(best_surface, claim)
            await self.surface_repo.save(best_surface)
            self.surfaces_updated += 1

            return LinkResult(
                claim_id=claim_id,
                surface_id=best_surface.id,
                is_new_surface=False,
                similarity=best_similarity
            )
        else:
            # Create new surface
            new_surface = self._create_surface_from_claim(claim, claim_anchors)
            await self.surface_repo.save(new_surface)
            self.surfaces_created += 1

            return LinkResult(
                claim_id=claim_id,
                surface_id=new_surface.id,
                is_new_surface=True,
                similarity=0.0
            )

    def _extract_anchors(self, claim: Claim) -> Set[str]:
        """Extract anchor entities from claim."""
        # For now, use all entities as potential anchors
        # TODO: Filter by IDF (high-IDF = discriminative = anchor)
        if claim.entities:
            return {e.canonical_name for e in claim.entities}
        return set()

    async def _compute_similarity(
        self,
        claim: Claim,
        surface: Surface
    ) -> float:
        """
        Compute similarity between claim and surface.

        Combines:
        - Embedding cosine similarity
        - Anchor entity overlap bonus

        Note: Support does NOT affect threshold.
        """
        similarity = 0.0

        # Embedding similarity (if surface has centroid)
        if claim.embedding and surface.centroid:
            # Cosine similarity
            claim_vec = np.array(claim.embedding)
            surface_vec = np.array(surface.centroid)

            dot_product = np.dot(claim_vec, surface_vec)
            norm_product = np.linalg.norm(claim_vec) * np.linalg.norm(surface_vec)

            if norm_product > 0:
                similarity = dot_product / norm_product

        # Anchor overlap bonus
        claim_anchors = self._extract_anchors(claim)
        if claim_anchors and surface.anchor_entities:
            overlap = len(claim_anchors & surface.anchor_entities)
            if overlap > 0:
                similarity += self.ANCHOR_BONUS

        return similarity

    def _create_surface_from_claim(
        self,
        claim: Claim,
        anchors: Set[str]
    ) -> Surface:
        """Create a new surface from a single claim."""
        # Use page_id as source identifier (tracks diversity of sources)
        sources = {claim.page_id} if claim.page_id else set()

        # Parse event_time if string
        event_time = claim.event_time
        if isinstance(event_time, str):
            from dateutil.parser import parse as parse_date
            event_time = parse_date(event_time)

        surface = Surface(
            id=generate_id('surface'),
            claim_ids={claim.id},
            entities=set(claim.entity_ids) if claim.entity_ids else set(),
            anchor_entities=anchors,
            sources=sources,
            time_start=event_time,
            time_end=event_time,
            centroid=claim.embedding,
            created_at=datetime.utcnow(),
        )
        surface.compute_support()
        return surface

    async def _update_surface_centroid(
        self,
        surface: Surface,
        new_claim: Claim
    ) -> None:
        """Update surface centroid with new claim."""
        if not new_claim.embedding:
            return

        if surface.centroid:
            # Incremental centroid update
            n = len(surface.claim_ids)
            old_centroid = np.array(surface.centroid)
            new_embedding = np.array(new_claim.embedding)

            # Weighted average (new claim contributes 1/n)
            new_centroid = (old_centroid * (n - 1) + new_embedding) / n
            surface.centroid = new_centroid.tolist()
        else:
            surface.centroid = new_claim.embedding

    async def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        logger.info(f"🛑 Weaver Worker {self.worker_id} stopping...")
        logger.info(
            f"📊 Final stats: {self.claims_processed} claims, "
            f"{self.surfaces_created} new surfaces, "
            f"{self.surfaces_updated} updated"
        )


# =============================================================================
# BOOTSTRAP: Reprocess all claims
# =============================================================================

async def bootstrap_lattice(
    db_pool: asyncpg.Pool,
    neo4j: Neo4jService,
    job_queue: JobQueue,
    batch_size: int = 100
):
    """
    Bootstrap the surface lattice from scratch.

    1. Clear existing surfaces
    2. Queue all claims for reprocessing
    3. Process claims in order

    Args:
        db_pool: PostgreSQL connection pool
        neo4j: Neo4j service
        job_queue: Redis job queue
        batch_size: Claims per batch
    """
    logger.info("🚀 Starting lattice bootstrap...")

    # 1. Clear existing surfaces (keep claims)
    logger.info("🧹 Clearing existing surfaces...")
    await neo4j._execute_write("MATCH (s:Surface) DETACH DELETE s")

    async with db_pool.acquire() as conn:
        await conn.execute("TRUNCATE content.surface_centroids")
        await conn.execute("TRUNCATE content.claim_surfaces")

    # 2. Count claims
    result = await neo4j._execute_read(
        "MATCH (c:Claim) RETURN count(c) as count"
    )
    total_claims = result[0]['count'] if result else 0
    logger.info(f"📚 Found {total_claims} claims to process")

    # 3. Queue claims in batches (ordered by time)
    logger.info("📥 Queueing claims...")
    offset = 0

    while offset < total_claims:
        claims = await neo4j._execute_read("""
            MATCH (c:Claim)
            RETURN c.id as id
            ORDER BY c.created_at ASC
            SKIP $offset
            LIMIT $limit
        """, {'offset': offset, 'limit': batch_size})

        for row in claims:
            await job_queue.enqueue(
                WeaverWorker.QUEUE_NAME,
                json.dumps({"claim_id": row['id']})
            )

        offset += batch_size
        logger.info(f"   Queued {min(offset, total_claims)}/{total_claims}")

    logger.info(f"✅ Queued {total_claims} claims for processing")
    logger.info("🧬 Start weaver worker to process queue")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the weaver worker."""
    import redis.asyncio as redis

    # Database connections
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=2,
        max_size=10
    )

    neo4j = Neo4jService(
        uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'password')
    )
    await neo4j.connect()

    redis_client = redis.from_url(
        os.getenv('REDIS_URL', 'redis://redis:6379')
    )
    job_queue = JobQueue(redis_client)

    # Check for bootstrap flag
    if '--bootstrap' in sys.argv:
        await bootstrap_lattice(db_pool, neo4j, job_queue)
        return

    # Run worker
    worker_id = int(os.getenv('WORKER_ID', 1))
    worker = WeaverWorker(db_pool, neo4j, job_queue, worker_id)

    try:
        await worker.run()
    except KeyboardInterrupt:
        await worker.stop()
    finally:
        await db_pool.close()
        await neo4j.close()
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
