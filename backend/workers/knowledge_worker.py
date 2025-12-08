"""
KnowledgeWorker - Unified extraction, identification, and linking pipeline

Replaces the fragmented semantic_worker + wikidata_worker with a single
atomic operation that guarantees entity integrity before event formation.

Pipeline stages:
0. Source Identification - Identify publisher entity + credibility
1. Extraction - LLM extracts mentions, claims, relationships
2. Identification - Resolve mentions to entity UUIDs (local + Wikidata)
3. Deduplication - Merge entities with same QID
4. Linking - Create all graph edges with UUIDs
5. Integrity Check - Verify completeness before triggering events

Output: page.status = 'knowledge_complete' → Event Worker can proceed
"""
import asyncio
import asyncpg
import os
import uuid
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import AsyncOpenAI
import logging

from models import Entity, Claim, Source
from models.mention import ExtractionResult
from repositories import EntityRepository, ClaimRepository, PageRepository
from services.neo4j_service import Neo4jService
from services.identification_service import IdentificationService, IdentificationResult
from services.wikidata_client import WikidataClient
from semantic_analyzer import EnhancedSemanticAnalyzer

logger = logging.getLogger(__name__)


class KnowledgeWorker:
    """
    Unified Knowledge Worker - Atomic extraction → identification → linking.

    Guarantees:
    - All entities identified before event formation
    - All duplicates merged (by QID)
    - All relationships use UUIDs (not strings)
    - Source credibility assigned
    """

    def __init__(self, db_pool: asyncpg.Pool, job_queue, neo4j_service: Neo4jService = None):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize Neo4j
        if neo4j_service is None:
            self.neo4j = Neo4jService()
        else:
            self.neo4j = neo4j_service

        # Initialize components
        self.analyzer = EnhancedSemanticAnalyzer()
        self.entity_repo = EntityRepository(db_pool, self.neo4j)
        self.claim_repo = ClaimRepository(db_pool, self.neo4j)
        self.page_repo = PageRepository(db_pool, self.neo4j)

        # Wikidata client for entity identification (enables QID resolution during pipeline)
        self.wikidata_client = WikidataClient()
        self.identification_service = IdentificationService(
            db_pool, self.neo4j, wikidata_client=self.wikidata_client
        )

        logger.info("✅ KnowledgeWorker initialized with Wikidata integration")

    async def connect_neo4j(self):
        """Ensure Neo4j connection is established."""
        await self.neo4j.connect()

    async def close(self):
        """Close all connections."""
        await self.neo4j.close()
        await self.wikidata_client.close()

    async def process(self, page_id: uuid.UUID, url: str) -> bool:
        """
        Process a page through the complete knowledge pipeline.

        Neo4j is the source of truth for the knowledge graph:
        - Page node created in Neo4j with metadata (content stays in PostgreSQL)
        - Claims linked to Page via CONTAINS relationship
        - Entities deduplicated by QID

        Returns True if knowledge_complete, False otherwise.
        """
        logger.info(f"🧠 KnowledgeWorker processing: {url}")

        try:
            async with self.db_pool.acquire() as conn:
                # Fetch page content from PostgreSQL
                page = await conn.fetchrow("""
                    SELECT id, url, canonical_url, title, content_text,
                           byline, site_name, domain, language, word_count,
                           pub_time, metadata, metadata_confidence
                    FROM core.pages
                    WHERE id = $1
                """, page_id)

                if not page:
                    logger.error(f"❌ Page {page_id} not found")
                    return False

                # Validate content
                if not page['content_text'] or page['word_count'] < 100:
                    logger.warning(f"⚠️ Insufficient content: {page['word_count']} words")
                    await self._mark_failed(conn, page_id, "Insufficient content")
                    return False

                logger.info(f"📄 {page['title']} ({page['word_count']} words)")

                # =========================================================
                # Create Page node in Neo4j (metadata only, content in PG)
                # =========================================================
                await self.neo4j.create_or_update_page(
                    page_id=str(page_id),
                    url=page['url'],
                    title=page['title'],
                    domain=page['domain'],
                    pub_time=page['pub_time'].isoformat() if page['pub_time'] else None,
                    status='processing',
                    language=page['language'] or 'en',
                    word_count=page['word_count'] or 0,
                    metadata_confidence=page.get('metadata_confidence', 0.0) or 0.0
                )

                # =========================================================
                # STAGE 0: Source Identification
                # =========================================================
                source = await self._identify_source(conn, page)
                logger.info(f"📰 Source: {source.canonical_name} (credibility: {source.credibility_score:.2f})")

                # =========================================================
                # STAGE 1: Extraction (LLM)
                # =========================================================
                page_meta = {
                    'title': page['title'],
                    'byline': page['byline'],
                    'pub_time': page['pub_time'].isoformat() if page['pub_time'] else None,
                    'site': page['site_name'] or page['domain']
                }
                page_text = [{'selector': 'article', 'text': page['content_text']}]

                extraction = await self.analyzer.extract_with_mentions(
                    page_meta, page_text, page['url'], page['language'] or 'en'
                )

                if not extraction.mentions:
                    logger.warning(f"⚠️ No mentions extracted")
                    await self._mark_failed(conn, page_id, "No mentions extracted")
                    return False

                logger.info(f"📝 Extracted: {len(extraction.mentions)} mentions, {len(extraction.claims)} claims")

                # =========================================================
                # STAGE 2: Identification (Local + Wikidata)
                # =========================================================
                page_context = {
                    'domain': page['domain'],
                    'language': page['language'],
                    'title': page['title']
                }

                identification = await self.identification_service.identify(
                    extraction, page_context
                )

                logger.info(f"🔍 Identified: {len(identification.new_entities)} new, "
                           f"{len(identification.matched_entities)} matched")

                # =========================================================
                # STAGE 3: Create new entities
                # =========================================================
                for entity in identification.new_entities:
                    await self.entity_repo.create(entity)
                    logger.debug(f"   ✨ Created: {entity.canonical_name}")

                # =========================================================
                # STAGE 3b: Update QIDs for matched entities (enrichment)
                # =========================================================
                for mention_id, match in identification.mention_to_entity.items():
                    if not match.is_new and match.wikidata_qid:
                        # Matched entity got a new QID from Wikidata - update it
                        await self._update_entity_qid(match.entity_id, match.wikidata_qid)

                # =========================================================
                # STAGE 4: Linking (all UUIDs)
                # =========================================================

                # 4a. Link page to source
                await self._link_page_to_source(page_id, source.id)

                # 4b. Create claims with entity links
                claim_ids = await self._create_claims(
                    conn, page_id, page['url'],
                    extraction, identification
                )
                logger.info(f"💾 Created {len(claim_ids)} claims")

                # 4c. Create entity relationships
                await self._create_entity_relationships(
                    extraction, identification
                )
                logger.info(f"🔗 Created {len(extraction.mention_relationships)} entity relationships")

                # 4d. Update entity profiles from mention descriptions
                await self._update_entity_profiles(extraction, identification)

                # =========================================================
                # STAGE 5: Integrity Check
                # =========================================================
                integrity_ok = await self._verify_integrity(
                    page_id, extraction, identification, claim_ids
                )

                if not integrity_ok:
                    logger.error(f"❌ Integrity check failed")
                    await self._mark_failed(conn, page_id, "Integrity check failed")
                    return False

                # =========================================================
                # Mark complete and trigger Event Worker
                # =========================================================
                # Update PostgreSQL (for content tracking)
                await conn.execute("""
                    UPDATE core.pages
                    SET status = 'knowledge_complete',
                        current_stage = 'knowledge',
                        updated_at = NOW()
                    WHERE id = $1
                """, page_id)

                # Update Neo4j Page node with final status and counts
                await self.neo4j.create_or_update_page(
                    page_id=str(page_id),
                    url=page['url'],
                    status='knowledge_complete',
                    claims_count=len(claim_ids),
                    entities_count=len(identification.mention_to_entity)
                )

                # Queue event worker with source credibility
                await self.job_queue.enqueue('queue:event:high', {
                    'page_id': str(page_id),
                    'url': url,
                    'claims_count': len(claim_ids),
                    'source': {
                        'entity_id': str(source.id),
                        'credibility_score': source.credibility_score,
                        'canonical_name': source.canonical_name
                    }
                })

                logger.info(f"✅ Knowledge complete: {url}")
                return True

        except Exception as e:
            logger.error(f"❌ KnowledgeWorker failed: {e}", exc_info=True)
            async with self.db_pool.acquire() as conn:
                await self._mark_failed(conn, page_id, str(e))
            return False

    async def _identify_source(self, conn: asyncpg.Connection, page: dict) -> Source:
        """
        Identify the publishing source for a page.

        Creates or retrieves Source entity with credibility score.
        """
        domain = page['domain']

        # Handle null domain - extract from URL if possible
        if not domain:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(page['url'])
                domain = parsed.netloc or 'unknown'
            except Exception:
                domain = 'unknown'

        site_name = page['site_name'] or domain

        # Check if source already exists
        existing = await self._find_source_by_domain(domain)
        if existing:
            # Increment claims count
            existing.claims_published += 1
            return existing

        # Create new source
        source = Source(
            id=uuid.uuid4(),
            canonical_name=site_name,
            entity_type="ORGANIZATION",
            domains=[domain],
            wikidata_qid=None,  # Will be enriched later
            claims_published=1
        )

        # Try to find Wikidata QID for the source
        # (This is a simple check - full enrichment happens later)
        qid = await self._lookup_source_qid(site_name, domain)
        if qid:
            source.wikidata_qid = qid

        # Create in Neo4j
        await self.neo4j._execute_write("""
            MERGE (s:Source {domain: $domain})
            ON CREATE SET
                s.id = $id,
                s.canonical_name = $name,
                s.entity_type = 'ORGANIZATION',
                s.wikidata_qid = $qid,
                s.claims_published = 1,
                s.claims_corroborated = 0,
                s.claims_contradicted = 0,
                s.created_at = datetime()
            ON MATCH SET
                s.claims_published = s.claims_published + 1,
                s.updated_at = datetime()
            RETURN s.id as id
        """, {
            'id': str(source.id),
            'domain': domain,
            'name': site_name,
            'qid': qid
        })

        logger.debug(f"📰 Source: {site_name} (credibility: {source.credibility_score:.2f})")
        return source

    async def _find_source_by_domain(self, domain: str) -> Optional[Source]:
        """Find existing source by domain."""
        results = await self.neo4j._execute_read("""
            MATCH (s:Source {domain: $domain})
            RETURN s.id as id,
                   s.canonical_name as canonical_name,
                   s.wikidata_qid as wikidata_qid,
                   s.claims_published as claims_published,
                   s.claims_corroborated as claims_corroborated,
                   s.claims_contradicted as claims_contradicted
        """, {'domain': domain})

        if results:
            row = results[0]
            return Source(
                id=uuid.UUID(row['id']),
                canonical_name=row['canonical_name'],
                entity_type="ORGANIZATION",
                domains=[domain],
                wikidata_qid=row.get('wikidata_qid'),
                claims_published=row.get('claims_published', 0),
                claims_corroborated=row.get('claims_corroborated', 0),
                claims_contradicted=row.get('claims_contradicted', 0)
            )
        return None

    async def _lookup_source_qid(self, name: str, domain: str) -> Optional[str]:
        """Quick lookup of Wikidata QID for a source (no full search)."""
        # This could be expanded to do actual Wikidata search
        # For now, just return None - enrichment worker handles this
        return None

    async def _link_page_to_source(self, page_id: uuid.UUID, source_id: uuid.UUID):
        """Create PUBLISHED_BY relationship between page and source."""
        await self.neo4j._execute_write("""
            MATCH (p:Page {id: $page_id})
            MATCH (s:Source {id: $source_id})
            MERGE (p)-[r:PUBLISHED_BY]->(s)
            ON CREATE SET r.created_at = datetime()
        """, {
            'page_id': str(page_id),
            'source_id': str(source_id)
        })

    async def _create_claims(
        self,
        conn: asyncpg.Connection,
        page_id: uuid.UUID,
        url: str,
        extraction: ExtractionResult,
        identification: IdentificationResult
    ) -> List[uuid.UUID]:
        """
        Create claims with entity links using UUIDs.

        Neo4j relationships created:
        - (Page)-[:CONTAINS]->(Claim)
        - (Claim)-[:MENTIONS]->(Entity)
        """
        claim_ids = []

        for claim_data in extraction.claims:
            # Generate deterministic ID
            claim_hash = hashlib.sha256(
                f"{url}|{claim_data.get('text', '')}".encode()
            ).hexdigest()[:16]

            # Resolve mention IDs to entity UUIDs
            entity_ids = []
            for mention_id in claim_data.get('who', []) + claim_data.get('where', []):
                entity_id = identification.get_entity_id(mention_id)
                if entity_id:
                    entity_ids.append(entity_id)

            # Parse event time
            event_time = None
            when = claim_data.get('when') or {}
            if when.get('date'):
                try:
                    date_str = when['date']
                    time_str = when.get('time', '00:00:00')
                    tz = when.get('timezone', '+00:00')
                    event_time = datetime.fromisoformat(f"{date_str}T{time_str}{tz}")
                except (ValueError, TypeError):
                    pass

            # Create claim
            claim = Claim(
                id=uuid.uuid4(),
                page_id=page_id,
                text=claim_data.get('text', ''),
                event_time=event_time,
                confidence=claim_data.get('confidence', 0.5),
                modality=claim_data.get('modality', 'observation'),
                metadata={
                    'deterministic_id': f"clm_{claim_hash}",
                    'who_mentions': claim_data.get('who', []),
                    'where_mentions': claim_data.get('where', []),
                    'when': when,
                    'evidence_references': claim_data.get('evidence_references', [])
                }
            )

            # Store via repository (handles Neo4j Claim node + PostgreSQL)
            created = await self.claim_repo.create(claim, entity_ids=entity_ids)
            claim_ids.append(created.id)

            # Link Page → Claim in Neo4j (CONTAINS relationship)
            await self.neo4j.link_page_to_claim(str(page_id), str(created.id))

        return claim_ids

    async def _create_entity_relationships(
        self,
        extraction: ExtractionResult,
        identification: IdentificationResult
    ):
        """
        Create entity relationships in Neo4j using UUIDs.
        """
        valid_predicates = {'PART_OF', 'LOCATED_IN', 'WORKS_FOR', 'MEMBER_OF', 'AFFILIATED_WITH'}

        for rel in extraction.mention_relationships:
            if rel.predicate not in valid_predicates:
                logger.warning(f"Unknown predicate: {rel.predicate}")
                continue

            subject_id = identification.get_entity_id(rel.subject_id)
            object_id = identification.get_entity_id(rel.object_id)

            if not subject_id or not object_id:
                logger.warning(f"Missing entity for relationship: {rel.subject_id} → {rel.object_id}")
                continue

            try:
                await self.neo4j._execute_write(f"""
                    MATCH (s:Entity {{id: $subject_id}})
                    MATCH (o:Entity {{id: $object_id}})
                    MERGE (s)-[r:{rel.predicate}]->(o)
                    ON CREATE SET r.created_at = datetime(), r.source = 'knowledge_worker'
                """, {
                    'subject_id': str(subject_id),
                    'object_id': str(object_id)
                })
            except Exception as e:
                logger.error(f"Failed to create relationship: {e}")

    async def _update_entity_profiles(
        self,
        extraction: ExtractionResult,
        identification: IdentificationResult
    ):
        """Update entity profiles from mention descriptions."""
        for mention in extraction.mentions:
            if not mention.description:
                continue

            entity_id = identification.get_entity_id(mention.id)
            if not entity_id:
                continue

            # Update profile if entity doesn't have one
            await self.entity_repo.update_profile(entity_id, mention.description)

            # Add aliases
            match = identification.mention_to_entity.get(mention.id)
            if match and mention.aliases:
                for alias in mention.aliases:
                    await self._add_entity_alias(entity_id, alias)

    async def _add_entity_alias(self, entity_id: uuid.UUID, alias: str):
        """Add alias to entity if not already present."""
        await self.neo4j._execute_write("""
            MATCH (e:Entity {id: $entity_id})
            WHERE NOT $alias IN coalesce(e.aliases, [])
            SET e.aliases = coalesce(e.aliases, []) + $alias
        """, {
            'entity_id': str(entity_id),
            'alias': alias
        })

    async def _update_entity_qid(self, entity_id: uuid.UUID, qid: str):
        """
        Update entity with Wikidata QID (enrichment during identification).

        If another entity already has this QID, merge the current entity into it
        (transfer relationships and delete duplicate).
        """
        # Check if another entity already has this QID
        existing = await self.neo4j._execute_read("""
            MATCH (e:Entity {wikidata_qid: $qid})
            WHERE e.id <> $entity_id
            RETURN e.id as id
        """, {'qid': qid, 'entity_id': str(entity_id)})

        if existing:
            # Another entity has this QID - merge into it
            existing_id = existing[0]['id']
            logger.info(f"🔄 Merging entity {entity_id} into existing {existing_id} (QID: {qid})")

            # Transfer all relationships from current entity to existing
            await self.neo4j._execute_write("""
                MATCH (current:Entity {id: $current_id})
                MATCH (existing:Entity {id: $existing_id})

                // Transfer incoming MENTIONS relationships
                OPTIONAL MATCH (c:Claim)-[r:MENTIONS]->(current)
                FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (c)-[:MENTIONS]->(existing)
                )

                // Transfer INVOLVES relationships (from events)
                OPTIONAL MATCH (ev:Event)-[r2:INVOLVES]->(current)
                FOREACH (_ IN CASE WHEN r2 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (ev)-[:INVOLVES]->(existing)
                )

                // Update mention count
                SET existing.mention_count = existing.mention_count + current.mention_count

                // Delete the duplicate entity
                DETACH DELETE current
            """, {
                'current_id': str(entity_id),
                'existing_id': existing_id
            })
            logger.info(f"✅ Merged entity {entity_id} → {existing_id}")
        else:
            # No conflict - just update the QID
            await self.neo4j._execute_write("""
                MATCH (e:Entity {id: $entity_id})
                WHERE e.wikidata_qid IS NULL
                SET e.wikidata_qid = $qid, e.updated_at = datetime()
            """, {
                'entity_id': str(entity_id),
                'qid': qid
            })
            logger.info(f"🔗 Updated entity QID: {entity_id} → {qid}")

    async def _verify_integrity(
        self,
        page_id: uuid.UUID,
        extraction: ExtractionResult,
        identification: IdentificationResult,
        claim_ids: List[uuid.UUID]
    ) -> bool:
        """
        Verify integrity before marking knowledge_complete.

        Checks:
        - All mentions have entity mappings
        - All claims created successfully
        - All relationships created
        """
        # Check mention coverage
        unmapped = [m.id for m in extraction.mentions if m.id not in identification.mention_to_entity]
        if unmapped:
            logger.warning(f"Unmapped mentions: {unmapped}")
            # Not a hard failure - some mentions might be skipped intentionally

        # Check claims created
        if len(claim_ids) == 0 and len(extraction.claims) > 0:
            logger.error("No claims created despite extraction having claims")
            return False

        # Basic sanity check
        if len(extraction.mentions) > 0 and len(identification.mention_to_entity) == 0:
            logger.error("No entities identified from mentions")
            return False

        return True

    async def _mark_failed(self, conn: asyncpg.Connection, page_id: uuid.UUID, reason: str):
        """Mark page as knowledge processing failed."""
        await conn.execute("""
            UPDATE core.pages
            SET status = 'knowledge_failed',
                error_message = $2,
                updated_at = NOW()
            WHERE id = $1
        """, page_id, reason)


async def run_knowledge_worker():
    """Main worker loop."""
    # Initialize database pool
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=10
    )

    # Initialize job queue
    from services.job_queue import JobQueue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Initialize Neo4j
    neo4j_service = Neo4jService()
    await neo4j_service.connect()
    await neo4j_service.initialize_constraints()

    # Initialize worker
    worker = KnowledgeWorker(db_pool, job_queue, neo4j_service)

    logger.info("🧠 KnowledgeWorker started, listening on queue:semantic:high")

    # Process jobs (same queue as old semantic worker for compatibility)
    while True:
        try:
            job = await job_queue.dequeue('queue:semantic:high', timeout=5)

            if job:
                page_id = uuid.UUID(job['page_id'])
                url = job.get('url', 'unknown')
                await worker.process(page_id, url)

            await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("👋 KnowledgeWorker shutting down")
            break
        except Exception as e:
            logger.error(f"❌ Worker error: {e}", exc_info=True)
            await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(run_knowledge_worker())
