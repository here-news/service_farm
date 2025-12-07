"""
Gen2 Semantic Worker - Extract claims and entities from pages

Architecture: Claims-first approach (Demo's proven method)
1. Extract claims with 6 premise checks
2. Extract entities FROM claims (no orphan entities)
3. Resolve entities (coreference + deduplication)
4. Link entities back to claims
5. Generate embeddings
6. Store in PostgreSQL AND Neo4j (via repositories)
7. Commission event worker

Based on: Demo's semantic_analyzer.py
Adapted for: Gen2 PostgreSQL schema + Neo4j graph
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

# Import the proven semantic analyzer from Demo
from semantic_analyzer import EnhancedSemanticAnalyzer

# Import domain models and repositories
from models import Entity, Claim
from models.relationships import ClaimEntityLink
from repositories import EntityRepository, ClaimRepository, PageRepository
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class SemanticWorker:
    """
    Gen2 Semantic Worker - Processes extracted pages to generate claims and entities

    Workflow:
    1. Fetch extracted page from database
    2. Run Demo's claim extraction (with 6 premise checks)
    3. Extract entities FROM claims
    4. Resolve and deduplicate entities
    5. Store claims, entities, relationships
    6. Generate page embedding
    7. Commission event worker
    """

    def __init__(self, db_pool: asyncpg.Pool, job_queue, neo4j_service: Neo4jService = None):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use Demo's proven semantic analyzer
        self.analyzer = EnhancedSemanticAnalyzer()

        # Initialize Neo4j service if not provided
        if neo4j_service is None:
            self.neo4j = Neo4jService()
        else:
            self.neo4j = neo4j_service

        # Initialize repositories (handle ALL data access)
        self.page_repo = PageRepository(db_pool)
        self.entity_repo = EntityRepository(db_pool, self.neo4j)
        self.claim_repo = ClaimRepository(db_pool, self.neo4j)

        logger.info("✅ SemanticWorker initialized with Demo's semantic analyzer + Neo4j repositories")

    async def connect_neo4j(self):
        """Ensure Neo4j connection is established"""
        await self.neo4j.connect()

    async def close_neo4j(self):
        """Close Neo4j connection"""
        await self.neo4j.close()

    async def process(self, page_id: uuid.UUID, url: str):
        """
        Process a page for semantic extraction

        Args:
            page_id: Page UUID
            url: Page URL (for logging)

        Returns:
            bool: Success status
        """
        logger.info(f"🧠 Starting semantic processing for {url}")

        try:
            async with self.db_pool.acquire() as conn:
                # 1. Fetch page content and metadata
                page = await conn.fetchrow("""
                    SELECT
                        id, url, canonical_url, title, content_text,
                        byline, site_name, domain, language, word_count,
                        pub_time, metadata, created_at
                    FROM core.pages
                    WHERE id = $1
                """, page_id)

                if not page:
                    logger.error(f"❌ Page {page_id} not found")
                    return False

                # Validation: Must have content
                if not page['content_text'] or page['word_count'] < 100:
                    logger.warning(f"⚠️ Page {page_id} has insufficient content ({page['word_count']} words)")
                    await self._mark_semantic_failed(conn, page_id, "Insufficient content")
                    return False

                # Warning: Missing critical metadata (but continue processing)
                if not page['title']:
                    logger.warning(f"⚠️ Page {page_id} is missing title - metadata incomplete but continuing")

                logger.info(f"📄 Processing: {page['title']} ({page['word_count']} words)")

                # 2. Prepare page metadata for analyzer
                page_meta = {
                    'title': page['title'],
                    'byline': page['byline'],
                    'pub_time': page['pub_time'].isoformat() if page['pub_time'] else None,
                    'site': page['site_name'] or page['domain']
                }

                # Format content for analyzer (expects list of text blocks)
                page_text = [{'selector': 'article', 'text': page['content_text']}]

                # 3. Extract claims and entities using Demo's analyzer
                logger.info("🔍 Extracting claims with 6 premise checks...")
                result = await self.analyzer.extract_enhanced_claims(
                    page_meta,
                    page_text,
                    page['url'],
                    datetime.utcnow().isoformat(),
                    page['language'] or 'en',
                    None  # about_text (optional summary)
                )

                claims = result.get('claims', [])
                excluded_claims = result.get('excluded_claims', [])
                entity_descriptions = result.get('entity_descriptions', {})  # {"PERSON:Name": "description", ...}
                gist = result.get('gist', '')

                logger.info(f"✅ Extracted {len(claims)} claims, excluded {len(excluded_claims)} claims")
                logger.info(f"📊 Entity descriptions from gpt-4o: {len(entity_descriptions)}")

                if len(claims) == 0:
                    logger.warning(f"⚠️ No valid claims extracted from {url}")
                    await self._mark_semantic_failed(conn, page_id, "No valid claims")
                    return False

                # 4. Extract entities FROM claims FIRST (WHO/WHERE have correct type prefixes)
                entity_mapping = await self._extract_entities_from_claims(
                    conn, claims, page['language']
                )
                logger.info(f"💾 Extracted {len(entity_mapping)} unique entities from claims")

                # 5. Store claims with entity_ids in metadata
                # NOTE: reported_time comes from page.pub_time via JOIN, not stored in claim
                claim_ids = await self._store_claims(
                    conn, page_id, claims, page['url'],
                    entity_mapping
                )
                logger.info(f"💾 Stored {len(claim_ids)} claims")
                logger.info(f"🔗 Linked entities to claims")

                # 7. Link entities to page (metadata) with centrality scoring
                await self._link_entities_to_page(conn, page_id, entity_mapping, claims)
                logger.info(f"🔗 Linked entities to page")

                # 8. Use entity descriptions from gpt-4o claim extraction (no extra LLM call)
                # entity_descriptions already contains {"PERSON:Name": "description", ...} from gpt-4o
                await self._apply_entity_descriptions(entity_mapping, entity_descriptions)
                logger.info(f"📝 Applied {len(entity_descriptions)} entity descriptions from claim extraction")

                # 9. Queue Wikidata enrichment for all new entities
                await self._queue_wikidata_enrichment(conn, entity_mapping, claims)
                logger.info(f"🔗 Queued Wikidata enrichment for {len(entity_mapping)} entities")

                # 10. Generate and store page embedding
                await self._generate_page_embedding(
                    conn, page_id, page['title'], gist, claims
                )
                logger.info(f"🎯 Generated page embedding")

                # 11. Update page status
                await conn.execute("""
                    UPDATE core.pages
                    SET status = 'semantic_complete',
                        current_stage = 'semantic',
                        updated_at = NOW()
                    WHERE id = $1
                """, page_id)

                # 12. Commission event worker
                await self.job_queue.enqueue('queue:event:high', {
                    'page_id': str(page_id),
                    'url': url,
                    'claims_count': len(claims)
                })
                logger.info(f"📤 Commissioned event worker")

                logger.info(f"✅ Semantic processing complete for {url}")
                return True

        except Exception as e:
            logger.error(f"❌ Semantic processing failed for {url}: {e}", exc_info=True)
            async with self.db_pool.acquire() as conn:
                await self._mark_semantic_failed(conn, page_id, str(e))
            return False


    async def _store_claims(
        self, conn: asyncpg.Connection, page_id: uuid.UUID,
        claims: List[Dict], url: str, entity_mapping: Dict[str, uuid.UUID]
    ) -> Dict[str, uuid.UUID]:
        """
        Store claims using ClaimRepository with entity_ids in metadata

        Args:
            entity_mapping: Dict mapping entity references (e.g., "PERSON:John") to UUIDs

        Returns mapping: {claim_deterministic_id: claim_uuid}

        Note: reported_time is NOT stored - it's queried from page.pub_time via JOIN
        """
        claim_mapping = {}

        for claim in claims:
            # Generate deterministic ID (for idempotency)
            claim_hash = hashlib.sha256(
                f"{url}|{claim['text']}".encode()
            ).hexdigest()[:16]
            claim_det_id = f"clm_{claim_hash}"

            # Parse event time
            event_time = None
            when = claim.get('when', {})

            # Try event_time_iso first (full datetime with timezone)
            if when and when.get('event_time_iso'):
                try:
                    iso_str = when['event_time_iso']
                    # Handle both 'Z' (UTC) and '+HH:MM'/'-HH:MM' timezone formats
                    if iso_str.endswith('Z'):
                        iso_str = iso_str.replace('Z', '+00:00')
                    event_time = datetime.fromisoformat(iso_str)
                except (ValueError, AttributeError):
                    pass

            # Fallback: try parsing just the date part if event_time_iso didn't work
            if event_time is None and when and when.get('date'):
                try:
                    date_str = when.get('date')
                    if date_str:
                        # Parse date and set time to midnight UTC
                        event_time = datetime.fromisoformat(date_str + 'T00:00:00+00:00')
                        logger.debug(f"Parsed date-only timestamp: {date_str} → {event_time}")
                except (ValueError, AttributeError):
                    pass

            # NOTE: Claim embeddings are NOT generated here
            # They will be generated on-demand in event_service when needed
            # This saves API calls and only generates embeddings for claims that need them

            # Extract entity IDs and names from claim's who/where fields
            entity_ids = []
            entity_names = []
            for entity_ref in claim.get('who', []) + claim.get('where', []):
                entity_uuid = entity_mapping.get(entity_ref)
                if entity_uuid:
                    entity_ids.append(entity_uuid)
                    # Extract canonical name from "PREFIX:Name" format
                    if ':' in entity_ref:
                        entity_names.append(entity_ref.split(':', 1)[1])

            # Create Claim domain model (no embedding)
            # NOTE: reported_time will be populated by repository via JOIN with pages
            claim_model = Claim(
                id=uuid.uuid4(),
                page_id=page_id,
                text=claim['text'],
                event_time=event_time,  # When the fact occurred (ground truth)
                reported_time=None,  # Queried from page.pub_time via JOIN
                confidence=claim.get('confidence', 0.5),
                modality=claim.get('modality', 'observation'),
                embedding=None,  # Generated on-demand
                metadata={
                    'who': claim.get('who', []),
                    'where': claim.get('where', []),
                    'when': when,
                    'evidence_references': claim.get('evidence_references', []),
                    'deterministic_id': claim_det_id,
                    'failed_checks': claim.get('failed_checks', []),
                    'verification_needed': claim.get('verification_needed', False)
                }
            )

            # Repository handles storage with entity_ids in metadata
            created_claim = await self.claim_repo.create(claim_model, entity_ids=entity_ids, entity_names=entity_names)
            claim_mapping[claim_det_id] = created_claim.id

        return claim_mapping


    async def _extract_entities_from_claims(
        self, conn: asyncpg.Connection, claims: List[Dict], language: str
    ) -> Dict[str, uuid.UUID]:
        """
        Extract entities from claims' WHO/WHERE fields (which have correct type prefixes)

        This is the proper approach - the WHO/WHERE fields have correct prefixes like:
        - "PERSON:John Doe"
        - "ORG:CNN"
        - "LOCATION:New York"

        Note: LLM may misclassify entities (e.g., "PERSON:Hong Kong"). This should be
        handled by entity resolution/enrichment workers that query Wikidata for correct types.

        Returns mapping: {prefixed_name: entity_uuid}
        """
        entity_mapping = {}
        seen_entities = set()

        # Map Demo's entity type prefixes to PostgreSQL enum values
        type_mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'LOCATION': 'LOCATION',
            'GPE': 'LOCATION'  # Geo-Political Entity (countries, cities, states)
        }

        for claim in claims:
            # Extract from WHO field (persons and organizations)
            for who in claim.get('who', []):
                if ':' not in who:
                    continue
                prefix, canonical_name = who.split(':', 1)
                canonical_name = canonical_name.strip()

                if not canonical_name or len(canonical_name) < 2:
                    continue

                # Map to PostgreSQL enum value
                entity_type = type_mapping.get(prefix)
                if not entity_type:
                    logger.warning(f"Unknown entity type prefix: {prefix}")
                    continue

                # Deduplicate
                key = f"{prefix}:{canonical_name}"
                if key in seen_entities:
                    continue
                seen_entities.add(key)

                entity_id = await self._upsert_entity(
                    conn, canonical_name, entity_type, language
                )
                entity_mapping[key] = entity_id

            # Extract from WHERE field (locations and sometimes organizations)
            for where in claim.get('where', []):
                if ':' not in where:
                    continue
                prefix, canonical_name = where.split(':', 1)
                canonical_name = canonical_name.strip()

                if not canonical_name or len(canonical_name) < 2:
                    continue

                # Map to PostgreSQL enum value
                entity_type = type_mapping.get(prefix)
                if not entity_type:
                    logger.warning(f"Unknown entity type prefix: {prefix}")
                    continue

                # Deduplicate
                key = f"{prefix}:{canonical_name}"
                if key in seen_entities:
                    continue
                seen_entities.add(key)

                entity_id = await self._upsert_entity(
                    conn, canonical_name, entity_type, language
                )
                entity_mapping[key] = entity_id

        return entity_mapping

    async def _resolve_and_store_entities(
        self, conn: asyncpg.Connection, entities_dict: Dict[str, List[str]],
        language: str
    ) -> Dict[str, uuid.UUID]:
        """
        DEPRECATED: Old approach using entities dict (has type errors like CNN as location)

        Resolve entities and store in database with deduplication

        Returns mapping: {canonical_name: entity_uuid}
        """
        entity_mapping = {}

        # Process people
        for person in entities_dict.get('people', []):
            canonical_name = person.strip()
            if not canonical_name or len(canonical_name) < 2:
                continue

            entity_id = await self._upsert_entity(
                conn, canonical_name, 'PERSON', language
            )
            entity_mapping[f"PERSON:{canonical_name}"] = entity_id

        # Process organizations
        for org in entities_dict.get('organizations', []):
            canonical_name = org.strip()
            if not canonical_name or len(canonical_name) < 2:
                continue

            # Validation: reject short fragments like "UN", "AL"
            if len(canonical_name) <= 2 and not self._is_known_acronym(canonical_name):
                logger.debug(f"⚠️ Rejected short fragment: {canonical_name}")
                continue

            entity_id = await self._upsert_entity(
                conn, canonical_name, 'ORGANIZATION', language
            )
            entity_mapping[f"ORG:{canonical_name}"] = entity_id

        # Process locations (with misclassification fixes)
        for location in entities_dict.get('locations', []):
            canonical_name = location.strip()
            if not canonical_name or len(canonical_name) < 2:
                continue

            # Fix common misclassifications: news orgs classified as locations
            known_orgs = {'CNN', 'BBC', 'Fox', 'MSNBC', 'NBC', 'CBS', 'ABC', 'NPR', 'Reuters', 'AP', 'AFP'}
            if canonical_name in known_orgs or canonical_name.endswith(' News') or canonical_name.endswith(' Post'):
                # Reclassify as organization
                entity_id = await self._upsert_entity(
                    conn, canonical_name, 'ORGANIZATION', language
                )
                entity_mapping[f"ORG:{canonical_name}"] = entity_id
                logger.debug(f"🔧 Reclassified {canonical_name} from LOCATION to ORGANIZATION")
            else:
                entity_id = await self._upsert_entity(
                    conn, canonical_name, 'LOCATION', language
                )
                entity_mapping[f"LOCATION:{canonical_name}"] = entity_id

        return entity_mapping


    def _is_known_acronym(self, text: str) -> bool:
        """Check if text is a known valid acronym"""
        known_acronyms = {
            'UN', 'EU', 'US', 'UK', 'FBI', 'CIA', 'WHO', 'NATO', 'NASA',
            'IMF', 'WTO', 'UN', 'UAE', 'UAE', 'BBC', 'CNN', 'AP', 'AFP'
        }
        return text.upper() in known_acronyms


    async def _upsert_entity(
        self, conn: asyncpg.Connection, canonical_name: str,
        entity_type: str, language: str
    ) -> uuid.UUID:
        """
        Upsert entity with local dedup using fuzzy matching.

        Strategy:
        1. Try exact match first (fast path)
        2. If no exact match, search for similar entities using fuzzy matching
        3. Only create new entity if no similar match found

        This prevents duplicates like "Hong Kong Labor Department" vs
        "Hong Kong labor department" from being created.
        """
        # 1. Try exact match first (fast path)
        existing = await self.entity_repo.get_by_canonical_name(canonical_name, entity_type)
        if existing:
            await self.entity_repo.increment_mention_count(existing.id)
            return existing.id

        # 2. Try fuzzy matching to find similar existing entity
        similar = await self.entity_repo.find_similar_entity(
            canonical_name=canonical_name,
            entity_type=entity_type,
            threshold=85.0  # 85% similarity threshold
        )

        if similar:
            similar_entity, similarity_score = similar
            logger.info(f"🔗 Dedup: '{canonical_name}' → '{similar_entity.canonical_name}' ({similarity_score:.0f}%)")
            await self.entity_repo.increment_mention_count(similar_entity.id)
            return similar_entity.id

        # 3. No match found - create new entity
        entity = Entity(
            id=uuid.uuid4(),
            canonical_name=canonical_name,
            entity_type=entity_type,
            mention_count=1,
            aliases=[],
            metadata={
                'language': language,
                'semantic_confidence': 0.7,
                'status': 'stub'
            }
        )

        created_entity = await self.entity_repo.create(entity)
        logger.debug(f"✨ Created new entity: {canonical_name} ({entity_type})")
        return created_entity.id


    async def _link_entities_to_claims(
        self, conn: asyncpg.Connection, claims: List[Dict],
        claim_mapping: Dict[str, uuid.UUID], entity_mapping: Dict[str, uuid.UUID]
    ):
        """Link entities to claims (many-to-many)"""
        total_links = 0
        for claim in claims:
            # Get claim UUID
            claim_hash = hashlib.sha256(
                f"{claim.get('_url', '')}|{claim['text']}".encode()
            ).hexdigest()[:16]
            claim_det_id = f"clm_{claim_hash}"
            claim_uuid = claim_mapping.get(claim_det_id)

            if not claim_uuid:
                logger.warning(f"❌ Claim UUID not found for {claim_det_id}: {claim['text'][:50]}")
                continue

            # Extract entities from claim.who and claim.where
            entities_in_claim = set()

            for who in claim.get('who', []):
                entities_in_claim.add(who)

            for where in claim.get('where', []):
                entities_in_claim.add(where)

            logger.info(f"🔍 Claim '{claim['text'][:60]}...' has {len(entities_in_claim)} entities: {entities_in_claim}")

            # Link each entity to claim
            for entity_ref in entities_in_claim:
                entity_uuid = entity_mapping.get(entity_ref)
                if not entity_uuid:
                    logger.warning(f"❌ Entity UUID not found for {entity_ref}")
                    continue

                # Note: claim-entity links will be created in Neo4j by event worker
                # when it creates Claim nodes in the graph. Entities are already in Neo4j.
                total_links += 1

        logger.info(f"✅ Tracked {total_links} claim-entity links (will be created in Neo4j by event worker)")


    async def _link_entities_to_page(
        self, conn: asyncpg.Connection, page_id: uuid.UUID,
        entity_mapping: Dict[str, uuid.UUID], claims: List[Dict]
    ):
        """Link all entities to page with centrality scoring"""
        # Count how many claims each entity appears in
        entity_claim_counts = {}

        for claim in claims:
            entities_in_claim = set()

            # Extract entities from WHO and WHERE
            for who in claim.get('who', []):
                entities_in_claim.add(who)
            for where in claim.get('where', []):
                entities_in_claim.add(where)

            # Increment count for each unique entity in this claim
            for entity_ref in entities_in_claim:
                entity_claim_counts[entity_ref] = entity_claim_counts.get(entity_ref, 0) + 1

        # Note: page-entity links will be created in Neo4j by event worker
        # when it builds the event graph. Entities are already in Neo4j.
        # Mention counts are tracked in Neo4j Entity nodes via MERGE ON MATCH.
        pass


    async def _apply_entity_descriptions(
        self, entity_mapping: Dict[str, uuid.UUID],
        entity_descriptions: Dict[str, str]
    ):
        """
        Apply entity descriptions extracted by gpt-4o during claim extraction.
        No additional LLM calls needed - descriptions come from the same gpt-4o call.
        """
        if not entity_descriptions:
            logger.info("No entity descriptions provided from claim extraction")
            return

        logger.info(f"Entity descriptions from gpt-4o: {list(entity_descriptions.keys())}")
        logger.info(f"Entities to update: {list(entity_mapping.keys())}")

        for entity_ref, entity_uuid in entity_mapping.items():
            # entity_ref is like "PERSON:Samuel Chu"
            description = entity_descriptions.get(entity_ref)

            if description:
                # Store in Neo4j
                await self.entity_repo.update_profile(entity_uuid, description)
                logger.info(f"📝 Applied: {entity_ref} -> {description[:50]}...")
            else:
                logger.info(f"⚠️ No description for {entity_ref}")

    async def _generate_entity_descriptions(
        self, conn: asyncpg.Connection, entity_mapping: Dict[str, uuid.UUID],
        claims: List[Dict], page_context: Dict[str, str] = None
    ):
        """DEPRECATED: Previously used gpt-4o-mini. Now using gpt-4o descriptions from claim extraction."""
        entity_contexts = {}

        # Collect all claims mentioning each entity
        for claim in claims:
            entities_in_claim = set()

            for who in claim.get('who', []):
                entities_in_claim.add(who)
            for where in claim.get('where', []):
                entities_in_claim.add(where)

            for entity_ref in entities_in_claim:
                if entity_ref not in entity_contexts:
                    entity_contexts[entity_ref] = []
                entity_contexts[entity_ref].append(claim['text'])

        # Extract location hint from page context
        page_location_hint = ""
        if page_context:
            domain = page_context.get('domain', '')
            site_name = page_context.get('site_name', '')
            title = page_context.get('title', '')

            # Infer location from domain/site
            if 'hk' in domain or 'hongkong' in domain.lower() or 'scmp' in domain:
                page_location_hint = "Hong Kong"
            elif 'china' in domain.lower() or '.cn' in domain:
                page_location_hint = "China"
            elif '.uk' in domain or 'bbc' in domain:
                page_location_hint = "UK"
            elif '.jp' in domain:
                page_location_hint = "Japan"

            # Also check site_name and title
            for loc in ['Hong Kong', 'China', 'Taiwan', 'Singapore', 'UK', 'US', 'Japan']:
                if loc.lower() in (site_name or '').lower() or loc.lower() in (title or '').lower():
                    page_location_hint = loc
                    break

        # Generate LLM-based descriptions (batch to save API calls)
        for entity_ref, entity_uuid in entity_mapping.items():
            if entity_ref not in entity_contexts:
                continue

            contexts = entity_contexts[entity_ref]
            entity_type, entity_name = entity_ref.split(':', 1) if ':' in entity_ref else ('UNKNOWN', entity_ref)

            # Use LLM to generate concise description
            context_text = '\n'.join(contexts[:3])  # Use up to 3 claims for context

            # Add page context hint
            article_context = ""
            if page_context and page_context.get('title'):
                article_context = f"\nArticle: {page_context['title']}"
            if page_location_hint:
                article_context += f"\nArticle location context: {page_location_hint}"

            try:
                # Build prompt based on entity type
                if entity_type == 'PERSON':
                    prompt = f"""Based on these mentions, describe WHO "{entity_name}" is.
IMPORTANT: Include their LOCATION (country/city) and ROLE/OCCUPATION. If location is not explicit in mentions but article is from a specific region, assume that location.

Mentions:
{context_text}{article_context}

Format: "[Location] [role/occupation] who [key action/characteristic]"
Examples:
- "Hong Kong activist who campaigns for democracy"
- "American politician serving as senator"
- "British journalist covering financial news"

Only return the description (max 20 words), nothing else."""
                elif entity_type in ('ORGANIZATION', 'LOCATION'):
                    prompt = f"""Based on these mentions, describe WHAT "{entity_name}" is.
IMPORTANT: Include LOCATION (country/city) and TYPE/FUNCTION. If location is not explicit but article is from a specific region, assume that location.

Mentions:
{context_text}{article_context}

Format: "[Location] [type] that [primary function]"
Examples:
- "Hong Kong government department overseeing building safety"
- "American technology company developing electric vehicles"
- "District in northern Hong Kong"

Only return the description (max 20 words), nothing else."""
                else:
                    prompt = f"""Based on these mentions of "{entity_name}", write a single concise sentence describing what/who this entity is (max 20 words):

{context_text}{article_context}

Format: "A/An [type] that/who [brief description]"
Only return the description, nothing else."""

                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=60
                )

                profile_summary = response.choices[0].message.content.strip()

                # Remove quotes if present
                profile_summary = profile_summary.strip('"\'')

                # Update profile in Neo4j via repository
                await self.entity_repo.update_profile(entity_uuid, profile_summary)

                logger.debug(f"📝 Generated description for {entity_name}: {profile_summary}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to generate description for {entity_name}: {e}")
                # Fallback: use first claim context (truncated)
                profile_summary = contexts[0][:200] if contexts else None
                if profile_summary:
                    # Update profile in Neo4j via repository
                    await self.entity_repo.update_profile(entity_uuid, profile_summary)

    async def _queue_wikidata_enrichment(
        self, conn: asyncpg.Connection, entity_mapping: Dict[str, uuid.UUID],
        claims: List[Dict]
    ):
        """
        Queue Wikidata enrichment jobs for entities that need linking

        Only queues entities that:
        - Don't already have a wikidata_qid
        - Are linkable types (PERSON, ORGANIZATION, LOCATION, GPE)
        """
        # Get entity details from Neo4j via repository
        entity_ids = list(entity_mapping.values())
        all_entities = await self.entity_repo.get_by_ids(entity_ids)

        # Filter entities that need enrichment
        entities_to_enrich = [
            entity for entity in all_entities
            if entity.wikidata_qid is None
            and entity.entity_type in ('PERSON', 'ORGANIZATION', 'LOCATION')
        ]

        # Collect claim contexts for each entity
        entity_contexts = {}
        for claim in claims:
            entities_in_claim = set()

            for who in claim.get('who', []):
                entities_in_claim.add(who)
            for where in claim.get('where', []):
                entities_in_claim.add(where)

            for entity_ref in entities_in_claim:
                if entity_ref not in entity_contexts:
                    entity_contexts[entity_ref] = []
                entity_contexts[entity_ref].append(claim['text'])

        # Queue enrichment jobs
        for entity in entities_to_enrich:
            entity_ref = f"{entity.entity_type}:{entity.canonical_name}"
            mentions = entity_contexts.get(entity_ref, [])

            await self.job_queue.enqueue('wikidata_enrichment', {
                'entity_id': str(entity.id),
                'canonical_name': entity.canonical_name,
                'entity_type': entity.entity_type,
                'context': {
                    'description': entity.profile_summary or '',
                    'mentions': mentions[:5]  # Include up to 5 claim contexts
                }
            })

        logger.debug(f"🔗 Queued {len(entities_to_enrich)} entities for Wikidata enrichment")

    async def _generate_page_embedding(
        self, conn: asyncpg.Connection, page_id: uuid.UUID,
        title: str, gist: str, claims: List[Dict]
    ):
        """Generate embedding from title + gist + top 5 claims"""
        # Combine text: title + gist + top 5 claim texts
        embedding_text = f"{title or ''}\n\n{gist or ''}\n\n"

        # Add top 5 claims (sorted by confidence)
        sorted_claims = sorted(
            claims, key=lambda c: c.get('confidence', 0), reverse=True
        )[:5]

        for claim in sorted_claims:
            embedding_text += f"- {claim['text']}\n"

        # Generate embedding
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=embedding_text[:8000]  # Token limit
            )
            embedding = response.data[0].embedding

            # Store in database (pgvector format)
            # Convert list to pgvector string format: '[0.1,0.2,0.3]'
            embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
            await conn.execute("""
                UPDATE core.pages
                SET embedding = $2::vector
                WHERE id = $1
            """, page_id, embedding_str)

        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")


    async def _generate_claim_embedding(
        self, claim_text: str
    ) -> Optional[List[float]]:
        """Generate embedding for a single claim text"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=claim_text[:8000]  # Token limit
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Failed to generate claim embedding: {e}")
            return None


    async def _mark_semantic_failed(
        self, conn: asyncpg.Connection, page_id: uuid.UUID, reason: str
    ):
        """
        Mark page as semantic processing failed

        Decision tree for failures:
        - word_count < 100 → Rogue task (likely paywall/anti-scraping)
        - word_count >= 100 → Playwright task (JS-heavy content, partial extraction)
        """
        # Get current page state
        page = await conn.fetchrow("""
            SELECT word_count, url FROM core.pages WHERE id = $1
        """, page_id)

        if not page:
            logger.error(f"Page {page_id} not found in _mark_semantic_failed")
            return

        word_count = page['word_count'] or 0
        url = page['url']

        # Insufficient content with some text → likely JS-heavy, try Playwright
        if "Insufficient content" in reason and word_count >= 100:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'playwright_pending',
                    error_message = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, page_id, f"Semantic failed: {reason} (queuing Playwright extraction)")

            # Queue for Playwright worker
            await self.job_queue.enqueue('playwright_extraction', {
                'page_id': str(page_id),
                'url': url,
                'reason': 'semantic_insufficient_content'
            })

            logger.info(
                f"🎭 Queued Playwright extraction for {url} (word_count={word_count}, reason: {reason})"
            )

        # Insufficient content with little/no text → likely paywall, try rogue
        elif "Insufficient content" in reason and word_count < 100:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'semantic_failed',
                    error_message = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, page_id, reason)

            # Create rogue task (browser extension handles auth/paywalls)
            await conn.execute("""
                INSERT INTO core.rogue_extraction_tasks (page_id, url, status, created_at)
                VALUES ($1, $2, 'pending', NOW())
                ON CONFLICT DO NOTHING
            """, page_id, url)

            logger.info(
                f"🔴 Created rogue task for {url} (word_count={word_count}, likely paywall)"
            )

        # Other semantic failures (no valid claims, etc.) with decent content → try Playwright
        elif word_count >= 100:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'playwright_pending',
                    error_message = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, page_id, f"Semantic failed: {reason} (queuing Playwright extraction)")

            # Queue for Playwright worker
            await self.job_queue.enqueue('playwright_extraction', {
                'page_id': str(page_id),
                'url': url,
                'reason': reason
            })

            logger.info(
                f"🎭 Queued Playwright extraction for {url} (word_count={word_count}, reason: {reason})"
            )

        # Other failures with low content → dead end
        else:
            await conn.execute("""
                UPDATE core.pages
                SET status = 'semantic_failed',
                    error_message = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, page_id, reason)

            logger.warning(
                f"❌ Semantic failed for {url} (word_count={word_count}, reason: {reason}) - no fallback"
            )


async def run_semantic_worker():
    """Main worker loop"""
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

    # Initialize Neo4j service
    neo4j_service = Neo4jService()
    await neo4j_service.connect()
    await neo4j_service.initialize_constraints()

    # Initialize worker
    worker = SemanticWorker(db_pool, job_queue, neo4j_service)

    logger.info("🧠 SemanticWorker started with Neo4j, listening on queue:semantic:high")

    # Process jobs
    while True:
        try:
            job = await job_queue.dequeue('queue:semantic:high', timeout=5)

            if job:
                page_id = uuid.UUID(job['page_id'])
                url = job.get('url', 'unknown')

                await worker.process(page_id, url)

            await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("👋 SemanticWorker shutting down")
            break
        except Exception as e:
            logger.error(f"❌ Worker error: {e}", exc_info=True)
            await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(run_semantic_worker())
