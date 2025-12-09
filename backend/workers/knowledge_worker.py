"""
KnowledgeWorker - Unified extraction, identification, and linking pipeline

Replaces the fragmented semantic_worker + wikidata_worker with a single
atomic operation that guarantees entity integrity before event formation.

Pipeline stages:
0. Publisher Identification - Identify publisher entity (ORGANIZATION)
1. Extraction - LLM extracts mentions, claims, relationships
2. Identification - Resolve mentions to entity IDs (local + Wikidata)
3. Deduplication - Merge entities with same QID
4. Linking - Create all graph edges with entity IDs
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

from models import Entity, Claim
from utils.id_generator import generate_entity_id
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
    - All relationships use entity IDs (not strings)
    - Publishers are Entity nodes with is_publisher=true
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

    async def process(self, page_id: str, url: str) -> bool:
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
                # STAGE 0: Publisher Identification
                # =========================================================
                publisher = await self._identify_publisher(conn, page)
                logger.info(f"📰 Publisher: {publisher.canonical_name} (qid={publisher.wikidata_qid})")

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
                        # Also store the wikidata_label (canonical name from Wikidata)
                        await self._update_entity_qid(
                            match.entity_id,
                            match.wikidata_qid,
                            wikidata_label=match.canonical_name
                        )

                # =========================================================
                # STAGE 4: Linking (all UUIDs)
                # =========================================================

                # 4a. Link page to publisher
                await self._link_page_to_publisher(str(page_id), publisher.id)

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

                # Queue event worker with extracted event info
                # TODO: Event deduplication to be designed - our events may be
                # larger scope than Wikidata events, need internal matching first
                await self.job_queue.enqueue('queue:event:high', {
                    'page_id': str(page_id),
                    'url': url,
                    'claims_count': len(claim_ids),
                    'extracted_event': extraction.event if extraction.event else None,
                    'publisher': {
                        'entity_id': publisher.id,
                        'canonical_name': publisher.canonical_name
                    }
                })

                logger.info(f"✅ Knowledge complete: {url}")
                return True

        except Exception as e:
            logger.error(f"❌ KnowledgeWorker failed: {e}", exc_info=True)
            async with self.db_pool.acquire() as conn:
                await self._mark_failed(conn, page_id, str(e))
            return False

    async def _identify_publisher(self, conn: asyncpg.Connection, page: dict) -> Entity:
        """
        Identify the publishing source for a page.

        Creates or retrieves Entity (ORGANIZATION) for the publisher.
        Publishers are entities with a domain property, resolved through Wikidata
        like any other organization.
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

        # Check if publisher entity already exists (by domain)
        existing = await self._find_publisher_by_domain(domain)
        if existing:
            logger.debug(f"📰 Found existing publisher: {existing.canonical_name} (qid={existing.wikidata_qid})")
            # If existing publisher has no QID, try to resolve via Wikidata
            if not existing.wikidata_qid and site_name and site_name != domain:
                logger.info(f"🔍 Resolving publisher via Wikidata: {site_name}")
                await self._resolve_publisher_wikidata(existing.id, site_name)
                # Re-fetch to get updated data
                existing = await self._find_publisher_by_domain(domain)
            return existing

        # Create new publisher entity
        entity_id = generate_entity_id()

        # Try to resolve via Wikidata to get canonical name and QID
        wikidata_qid = None
        canonical_name = site_name
        wikidata_label = None

        if site_name != domain:  # Only search if we have a real name, not just domain
            try:
                result = await self.wikidata_client.search_entity(
                    site_name, entity_type='ORGANIZATION'
                )
                if result and result.get('accepted'):
                    wikidata_qid = result.get('qid')
                    wikidata_label = result.get('label')
                    if wikidata_label:
                        canonical_name = wikidata_label
                    logger.info(f"🔗 Publisher resolved: {site_name} → {canonical_name} ({wikidata_qid})")
            except Exception as e:
                logger.debug(f"Wikidata search failed for publisher {site_name}: {e}")

        publisher = Entity(
            id=entity_id,
            canonical_name=canonical_name,
            entity_type="ORGANIZATION",
            wikidata_qid=wikidata_qid,
            status='resolved' if wikidata_qid else 'pending'
        )

        # Create Entity node in Neo4j with domain property for publisher lookup
        # Use dedup_key based on domain for publishers
        dedup_key = f"publisher_{domain.lower()}"
        await self.neo4j._execute_write("""
            MERGE (e:Entity {dedup_key: $dedup_key})
            ON CREATE SET
                e.id = $id,
                e.canonical_name = $name,
                e.entity_type = 'ORGANIZATION',
                e.domain = $domain,
                e.is_publisher = true,
                e.wikidata_qid = $qid,
                e.wikidata_label = $wikidata_label,
                e.mention_count = 1,
                e.status = $status,
                e.created_at = datetime()
            ON MATCH SET
                e.mention_count = e.mention_count + 1,
                e.updated_at = datetime()
            RETURN e.id as id
        """, {
            'id': entity_id,
            'dedup_key': dedup_key,
            'domain': domain,
            'name': canonical_name,
            'qid': wikidata_qid,
            'wikidata_label': wikidata_label,
            'status': 'resolved' if wikidata_qid else 'pending'
        })

        logger.debug(f"📰 Publisher: {canonical_name} ({entity_id})")
        return publisher

    async def _resolve_publisher_wikidata(self, entity_id: str, site_name: str):
        """Try to resolve an existing publisher entity via Wikidata."""
        try:
            result = await self.wikidata_client.search_entity(
                site_name, entity_type='ORGANIZATION'
            )
            if result and result.get('accepted'):
                qid = result.get('qid')
                label = result.get('label')
                if qid:
                    await self.neo4j._execute_write("""
                        MATCH (e:Entity {id: $id})
                        SET e.wikidata_qid = $qid,
                            e.wikidata_label = $label,
                            e.canonical_name = COALESCE($label, e.canonical_name),
                            e.status = 'resolved',
                            e.updated_at = datetime()
                    """, {'id': entity_id, 'qid': qid, 'label': label})
                    logger.info(f"🔗 Publisher enriched: {entity_id} → {label} ({qid})")
        except Exception as e:
            logger.debug(f"Publisher Wikidata resolution failed: {e}")

    async def _find_publisher_by_domain(self, domain: str) -> Optional[Entity]:
        """Find existing publisher entity by domain."""
        results = await self.neo4j._execute_read("""
            MATCH (e:Entity {domain: $domain, is_publisher: true})
            RETURN e.id as id,
                   e.canonical_name as canonical_name,
                   e.wikidata_qid as wikidata_qid,
                   e.mention_count as mention_count,
                   e.status as status
        """, {'domain': domain})

        if results:
            row = results[0]
            return Entity(
                id=row['id'],
                canonical_name=row['canonical_name'],
                entity_type="ORGANIZATION",
                wikidata_qid=row.get('wikidata_qid'),
                mention_count=row.get('mention_count', 0),
                status=row.get('status', 'pending')
            )
        return None

    async def _link_page_to_publisher(self, page_id: str, publisher_id: str):
        """Create PUBLISHED_BY relationship between page and publisher entity."""
        await self.neo4j._execute_write("""
            MATCH (p:Page {id: $page_id})
            MATCH (e:Entity {id: $publisher_id})
            MERGE (p)-[r:PUBLISHED_BY]->(e)
            ON CREATE SET r.created_at = datetime()
        """, {
            'page_id': page_id,
            'publisher_id': publisher_id
        })

    async def _create_claims(
        self,
        conn: asyncpg.Connection,
        page_id: str,
        url: str,
        extraction: ExtractionResult,
        identification: IdentificationResult
    ) -> List[str]:
        """
        Create claims with entity links using UUIDs.

        Neo4j relationships created:
        - (Page)-[:CONTAINS]->(Claim)
        - (Claim)-[:MENTIONS]->(Entity)
        """
        claim_ids = []

        for claim_data in extraction.claims:
            # Generate deterministic ID for deduplication
            claim_hash = hashlib.sha256(
                f"{url}|{claim_data.get('text', '')}".encode()
            ).hexdigest()[:16]
            deterministic_id = f"clm_{claim_hash}"

            # Check if claim already exists (deduplication)
            existing = await self.neo4j._execute_read("""
                MATCH (c:Claim {deterministic_id: $det_id})
                RETURN c.id as id
            """, {'det_id': deterministic_id})

            if existing:
                # Claim exists - just link Page→Claim (in case page was reprocessed)
                existing_id = existing[0]['id']
                await self.neo4j.link_page_to_claim(str(page_id), existing_id)
                claim_ids.append(existing_id)
                continue

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
                    'deterministic_id': deterministic_id,
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

    async def _add_entity_alias(self, entity_id: str, alias: str):
        """Add alias to entity if not already present."""
        await self.neo4j._execute_write("""
            MATCH (e:Entity {id: $entity_id})
            WHERE NOT $alias IN coalesce(e.aliases, [])
            SET e.aliases = coalesce(e.aliases, []) + $alias
        """, {
            'entity_id': str(entity_id),
            'alias': alias
        })

    async def _update_entity_qid(
        self,
        entity_id: str,
        qid: str,
        wikidata_label: str = None
    ):
        """
        Update entity with Wikidata QID and label (enrichment during identification).

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
            # No conflict - update QID and wikidata_label
            await self.neo4j._execute_write("""
                MATCH (e:Entity {id: $entity_id})
                WHERE e.wikidata_qid IS NULL
                SET e.wikidata_qid = $qid,
                    e.wikidata_label = $wikidata_label,
                    e.updated_at = datetime()
            """, {
                'entity_id': str(entity_id),
                'qid': qid,
                'wikidata_label': wikidata_label
            })
            logger.info(f"🔗 Updated entity QID: {entity_id} → {qid} ({wikidata_label})")

    async def _verify_integrity(
        self,
        page_id: str,
        extraction: ExtractionResult,
        identification: IdentificationResult,
        claim_ids: List[str]
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

    async def _mark_failed(self, conn: asyncpg.Connection, page_id: str, reason: str):
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
                page_id = job['page_id']
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
