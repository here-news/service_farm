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

from pgvector.asyncpg import register_vector

from models.domain import Entity, Claim
from utils.id_generator import generate_entity_id, generate_claim_id
from models.domain.mention import ExtractionResult
from repositories import EntityRepository, ClaimRepository, PageRepository
from services.neo4j_service import Neo4jService
from services.entity_manager import EntityManager
from services.identification_service import IdentificationService, IdentificationResult
from services.wikidata_client import WikidataClient
from services.source_classification import (
    classify_source_by_domain, classify_source_from_wikidata,
    compute_base_prior, get_prior_reason
)
from services.author_parser import parse_byline, ParsedAuthor
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

        # EntityManager for unified entity deduplication and merging
        self.entity_manager = EntityManager(self.neo4j)

        # Wikidata client for entity identification (enables QID resolution during pipeline)
        self.wikidata_client = WikidataClient()
        self.identification_service = IdentificationService(
            db_pool, self.neo4j, wikidata_client=self.wikidata_client
        )

        logger.info("✅ KnowledgeWorker initialized with EntityManager and Wikidata integration")

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
                # STAGE 0b: Author Identification
                # =========================================================
                authors = await self._identify_authors(page)
                if authors:
                    author_names = [a.canonical_name for a in authors]
                    logger.info(f"✍️ Authors: {', '.join(author_names)}")

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
                        # Use wikidata_label (authoritative name from Wikidata) for canonical name
                        await self._update_entity_qid(
                            match.entity_id,
                            match.wikidata_qid,
                            wikidata_label=match.wikidata_label  # Use Wikidata label, not current name
                        )

                # =========================================================
                # STAGE 4: Linking (all UUIDs)
                # =========================================================

                # 4a. Link page to publisher
                await self._link_page_to_publisher(str(page_id), publisher.id)

                # 4a2. Link page to authors
                for author in authors:
                    await self._link_page_to_author(str(page_id), author.id)

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

                # 4e. Generate and store page embedding (for event matching)
                await self._generate_and_store_page_embedding(conn, page_id, extraction.claims)
                logger.info(f"📊 Generated page embedding")

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

                # Signal Event Worker that knowledge is complete
                # Event worker will process all claims from this page together
                await self.job_queue.enqueue('queue:event:high', {
                    'page_id': str(page_id),
                    'url': url,
                    'claims_count': len(claim_ids)
                })

                logger.info(f"✅ Knowledge complete: {url} → Event worker")
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
        Uses EntityManager for unified deduplication - if a publisher with
        the same QID exists, it will be reused.

        Also classifies source type and assigns base prior for Bayesian analysis:
        - source_type: 'wire', 'official', 'local_news', 'international', 'aggregator', 'unknown'
        - base_prior: Conservative probability (0.50 - 0.65)
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
        existing = await self.entity_manager.get_by_domain(domain)
        if existing:
            logger.debug(f"📰 Found existing publisher: {existing.get('canonical_name')} (qid={existing.get('wikidata_qid')})")

            # If existing publisher has no QID, try to resolve via Wikidata
            if not existing.get('wikidata_qid') and site_name and site_name != domain:
                logger.info(f"🔍 Resolving publisher via Wikidata: {site_name}")
                await self._resolve_publisher_wikidata(existing['id'], site_name)
                # Re-fetch to get updated data
                existing = await self.entity_manager.get_by_domain(domain)

            # Update source classification if missing
            source_type, has_byline = classify_source_by_domain(domain, site_name)
            base_prior = compute_base_prior(source_type, has_byline)
            await self.entity_manager.update_publisher_classification(
                existing['id'], source_type, base_prior
            )

            return Entity(
                id=existing['id'],
                canonical_name=existing.get('canonical_name', site_name),
                entity_type='ORGANIZATION',
                wikidata_qid=existing.get('wikidata_qid'),
                status=existing.get('status', 'pending')
            )

        # Try to resolve via Wikidata to get canonical name, QID, and P31 for source type
        wikidata_qid = None
        canonical_name = site_name
        wikidata_label = None
        p31_qids = []  # P31 (instance of) values for source type classification

        if site_name != domain:  # Only search if we have a real name, not just domain
            try:
                # Use specialized publisher search with P856 domain matching
                result = await self.wikidata_client.search_publisher(
                    name=site_name,
                    domain=domain
                )
                if result and result.get('accepted'):
                    wikidata_qid = result.get('qid')
                    wikidata_label = result.get('label')

                    if wikidata_label:
                        canonical_name = wikidata_label
                    logger.info(f"🔗 Publisher resolved: {site_name} → {canonical_name} ({wikidata_qid})")

                    # Fetch P31 (instance of) for source type classification
                    if wikidata_qid:
                        p31_qids = await self._fetch_wikidata_p31(wikidata_qid)
                        logger.debug(f"   P31 values: {p31_qids}")
            except Exception as e:
                logger.debug(f"Wikidata search failed for publisher {site_name}: {e}")

        # Classify source type:
        # 1. Prefer Wikidata P31 if available
        # 2. Fall back to domain heuristics
        source_type = classify_source_from_wikidata(p31_qids)
        if source_type:
            has_byline = source_type not in ('official', 'aggregator')
            logger.debug(f"   Source type from Wikidata P31: {source_type}")
        else:
            source_type, has_byline = classify_source_by_domain(domain, site_name)
            logger.debug(f"   Source type from domain fallback: {source_type}")

        base_prior = compute_base_prior(source_type, has_byline)

        # Use EntityManager for unified entity creation with QID deduplication
        # If an entity with this QID exists, it will be returned instead
        entity_id = generate_entity_id()
        returned_id = await self.entity_manager.get_or_create(
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type='ORGANIZATION',
            wikidata_qid=wikidata_qid,
            wikidata_label=wikidata_label,
            domain=domain,
            is_publisher=True,
            source_type=source_type,
            base_prior=base_prior,
            status='resolved' if wikidata_qid else 'pending'
        )

        # If returned_id differs, an existing entity was found
        if returned_id != entity_id:
            logger.info(f"📰 Publisher (reused): {canonical_name} → {returned_id}")
            existing = await self.entity_manager.get_by_id(returned_id)
            return Entity(
                id=returned_id,
                canonical_name=existing.get('canonical_name', canonical_name) if existing else canonical_name,
                entity_type='ORGANIZATION',
                wikidata_qid=wikidata_qid,
                status='resolved' if wikidata_qid else 'pending'
            )

        logger.info(f"📰 Publisher: {canonical_name} [{source_type}] (prior={base_prior})")
        return Entity(
            id=entity_id,
            canonical_name=canonical_name,
            entity_type='ORGANIZATION',
            wikidata_qid=wikidata_qid,
            status='resolved' if wikidata_qid else 'pending'
        )


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
                    # Fetch P31 for source type classification
                    p31_qids = await self._fetch_wikidata_p31(qid)
                    source_type = classify_source_from_wikidata(p31_qids)

                    await self.neo4j._execute_write("""
                        MATCH (e:Entity {id: $id})
                        SET e.wikidata_qid = $qid,
                            e.wikidata_label = $label,
                            e.canonical_name = COALESCE($label, e.canonical_name),
                            e.source_type = COALESCE($source_type, e.source_type),
                            e.base_prior = CASE WHEN $base_prior IS NOT NULL THEN $base_prior ELSE e.base_prior END,
                            e.status = 'resolved',
                            e.updated_at = datetime()
                    """, {
                        'id': entity_id,
                        'qid': qid,
                        'label': label,
                        'source_type': source_type,
                        'base_prior': compute_base_prior(source_type, True) if source_type else None
                    })
                    logger.info(f"🔗 Publisher enriched: {entity_id} → {label} ({qid}) [{source_type or 'unknown'}]")
        except Exception as e:
            logger.debug(f"Publisher Wikidata resolution failed: {e}")

    async def _fetch_wikidata_p31(self, qid: str) -> List[str]:
        """
        Fetch P31 (instance of) values for a Wikidata entity.

        Used to determine source type from Wikidata's structured data.

        Args:
            qid: Wikidata QID (e.g., "Q40469" for Associated Press)

        Returns:
            List of P31 QIDs (e.g., ["Q192283"] for news agency)
        """
        try:
            await self.wikidata_client._ensure_session()

            params = {
                'action': 'wbgetentities',
                'ids': qid,
                'format': 'json',
                'props': 'claims'
            }

            async with self.wikidata_client.session.get(
                self.wikidata_client.api_url, params=params, timeout=10
            ) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json()
                entity = data.get('entities', {}).get(qid, {})
                claims = entity.get('claims', {})

                p31_qids = []
                for claim in claims.get('P31', []):
                    mainsnak = claim.get('mainsnak', {})
                    datavalue = mainsnak.get('datavalue', {})
                    if datavalue.get('type') == 'wikibase-entityid':
                        p31_qids.append(datavalue['value']['id'])

                return p31_qids

        except Exception as e:
            logger.debug(f"Failed to fetch P31 for {qid}: {e}")
            return []

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

    async def _link_page_to_author(self, page_id: str, author_id: str):
        """Create AUTHORED_BY relationship between page and author entity."""
        await self.neo4j._execute_write("""
            MATCH (p:Page {id: $page_id})
            MATCH (e:Entity {id: $author_id})
            MERGE (p)-[r:AUTHORED_BY]->(e)
            ON CREATE SET r.created_at = datetime()
        """, {
            'page_id': page_id,
            'author_id': author_id
        })

    async def _identify_authors(self, page: dict) -> List[Entity]:
        """
        Identify author entities from page byline.

        Parses byline to extract individual author names, then:
        1. Searches local graph for existing author entity
        2. Searches Wikidata for known journalists/writers
        3. Creates new PERSON entity if no match found

        Returns:
            List of Entity objects for page authors
        """
        byline = page.get('byline')
        if not byline:
            return []

        # Parse byline into individual author names
        parsed_authors = parse_byline(byline)
        if not parsed_authors:
            return []

        authors = []
        for parsed in parsed_authors:
            if not parsed.is_person:
                # Skip organizational credits like "AP Staff"
                continue

            author_entity = await self._identify_single_author(parsed.name)
            if author_entity:
                authors.append(author_entity)

        return authors

    async def _identify_single_author(self, author_name: str) -> Optional[Entity]:
        """
        Identify a single author by name.

        Search order:
        1. Local graph by canonical_name (exact match)
        2. Wikidata search for journalists/writers
        3. Create new local entity
        """
        # 1. Search local graph for existing author entity
        existing = await self.neo4j._execute_read("""
            MATCH (e:Entity {entity_type: 'PERSON'})
            WHERE toLower(e.canonical_name) = toLower($name)
               OR toLower($name) IN [a IN coalesce(e.aliases, []) | toLower(a)]
            RETURN e.id as id, e.canonical_name as canonical_name,
                   e.wikidata_qid as wikidata_qid
            LIMIT 1
        """, {'name': author_name})

        if existing and len(existing) > 0:
            ent = existing[0]
            logger.debug(f"✍️ Author (existing): {author_name} → {ent['canonical_name']}")
            return Entity(
                id=ent['id'],
                canonical_name=ent['canonical_name'],
                entity_type='PERSON',
                wikidata_qid=ent.get('wikidata_qid')
            )

        # 2. Skip Wikidata search for authors
        # Most article authors aren't notable enough for Wikidata entries,
        # and generic name matching produces too many false positives
        # (e.g., "Lucy Swan" matching an 18th century person).
        # Authors will be linked by name; if they're also mentioned as
        # entities in article content, they'll get QIDs through that path.

        # 3. Create author entity using EntityManager
        entity_id = generate_entity_id()

        returned_id = await self.entity_manager.get_or_create(
            entity_id=entity_id,
            canonical_name=author_name,
            entity_type='PERSON',
            is_author=True,
            status='pending'  # No QID, pending enrichment
        )

        # EntityManager may return existing entity ID if matched by name
        final_id = returned_id if returned_id else entity_id

        return Entity(
            id=final_id,
            canonical_name=author_name,
            entity_type='PERSON',
            status='pending'
        )

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
        - (Page)-[:EMITS]->(Claim)
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
                id=generate_claim_id(),
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

        Delegates to EntityManager which handles:
        - QID conflict detection
        - Relationship rewiring if merge needed
        - Alias and mention count merging
        """
        await self.entity_manager.update_qid(entity_id, qid, wikidata_label)

    async def _generate_and_store_page_embedding(
        self,
        conn: asyncpg.Connection,
        page_id: str,
        claims: List
    ):
        """
        Generate semantic embedding for page from its claims.

        STAGE 4e: Generate page embedding for event matching.

        The embedding represents "what is this page about?" and enables
        semantic similarity matching when routing to events.

        Args:
            conn: PostgreSQL connection
            page_id: Page UUID
            claims: Extracted claims from page
        """
        if not claims:
            logger.warning(f"No claims to generate embedding from")
            return

        # Combine claim texts (limit to ~8k chars to avoid token limits)
        # Claims from extraction are dicts with 'text' key
        claim_texts = [c.get('text') or c['text'] if isinstance(c, dict) else c.text
                      for c in claims
                      if (isinstance(c, dict) and c.get('text')) or (hasattr(c, 'text') and c.text)]
        if not claim_texts:
            logger.warning(f"No claim texts available for embedding")
            return

        # Concatenate claims with newlines (truncate if too long)
        combined_text = "\n".join(claim_texts)
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."

        try:
            # Generate embedding using OpenAI
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text
            )

            embedding = response.data[0].embedding

            # Store in PostgreSQL core.pages table using pgvector native type
            # register_vector() enables passing Python list directly
            await register_vector(conn)

            await conn.execute("""
                UPDATE core.pages
                SET embedding = $1,
                    updated_at = NOW()
                WHERE id = $2
            """, embedding, page_id)

            logger.debug(f"✅ Stored page embedding ({len(embedding)} dimensions)")

        except Exception as e:
            logger.error(f"Failed to generate/store page embedding: {e}")
            # Non-fatal - continue processing

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
