"""
Gen2 Semantic Worker - Extract claims and entities from pages

Architecture: Claims-first approach (Demo's proven method)
1. Extract claims with 6 premise checks
2. Extract entities FROM claims (no orphan entities)
3. Resolve entities (coreference + deduplication)
4. Link entities back to claims
5. Generate embeddings
6. Store in PostgreSQL
7. Commission event worker

Based on: Demo's semantic_analyzer.py
Adapted for: Gen2 PostgreSQL schema
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

    def __init__(self, db_pool: asyncpg.Pool, job_queue):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use Demo's proven semantic analyzer
        self.analyzer = EnhancedSemanticAnalyzer()

        logger.info("✅ SemanticWorker initialized with Demo's semantic analyzer")


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
                entities_dict = result.get('entities', {})
                gist = result.get('gist', '')

                logger.info(f"✅ Extracted {len(claims)} claims, excluded {len(excluded_claims)} claims")
                logger.info(f"📊 Entities: {len(entities_dict.get('people', []))} people, "
                           f"{len(entities_dict.get('organizations', []))} orgs, "
                           f"{len(entities_dict.get('locations', []))} locations")

                if len(claims) == 0:
                    logger.warning(f"⚠️ No valid claims extracted from {url}")
                    await self._mark_semantic_failed(conn, page_id, "No valid claims")
                    return False

                # 4. Store claims in database
                claim_ids = await self._store_claims(conn, page_id, claims, page['url'])
                logger.info(f"💾 Stored {len(claim_ids)} claims")

                # 5. Extract entities FROM claims (WHO/WHERE have correct type prefixes)
                entity_mapping = await self._extract_entities_from_claims(
                    conn, claims, page['language']
                )
                logger.info(f"💾 Extracted {len(entity_mapping)} unique entities from claims")

                # 6. Link entities to claims
                # Add _url to claims for deterministic ID matching
                for claim in claims:
                    claim['_url'] = page['url']
                await self._link_entities_to_claims(conn, claims, claim_ids, entity_mapping)
                logger.info(f"🔗 Linked entities to claims")

                # 7. Link entities to page (metadata) with centrality scoring
                await self._link_entities_to_page(conn, page_id, entity_mapping, claims)
                logger.info(f"🔗 Linked entities to page")

                # 8. Generate basic entity descriptions from context
                await self._generate_entity_descriptions(conn, entity_mapping, claims)
                logger.info(f"📝 Generated entity descriptions")

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
        claims: List[Dict], url: str
    ) -> Dict[str, uuid.UUID]:
        """
        Store claims in database

        Returns mapping: {claim_deterministic_id: claim_uuid}
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
            if when and when.get('event_time_iso'):
                try:
                    event_time = datetime.fromisoformat(
                        when['event_time_iso'].replace('Z', '+00:00')
                    )
                except (ValueError, AttributeError):
                    pass

            # Generate claim embedding
            claim_embedding = await self._generate_claim_embedding(claim['text'])
            claim_embedding_str = None
            if claim_embedding:
                claim_embedding_str = '[' + ','.join(str(x) for x in claim_embedding) + ']'

            # Store claim
            claim_uuid = await conn.fetchval("""
                INSERT INTO core.claims (
                    page_id, text, event_time, confidence, modality, metadata, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                RETURNING id
            """,
                page_id,
                claim['text'],
                event_time,
                claim.get('confidence', 0.5),
                claim.get('modality', 'observation'),
                json.dumps({
                    'who': claim.get('who', []),
                    'where': claim.get('where', []),
                    'when': when,
                    'evidence_references': claim.get('evidence_references', []),
                    'deterministic_id': claim_det_id,
                    'failed_checks': claim.get('failed_checks', []),
                    'verification_needed': claim.get('verification_needed', False)
                }),
                claim_embedding_str
            )

            claim_mapping[claim_det_id] = claim_uuid

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
        Upsert entity with deduplication (Demo's approach)

        Uses UNIQUE constraint on (canonical_name, entity_type)
        """
        # Try to find existing entity
        existing = await conn.fetchval("""
            SELECT id FROM core.entities
            WHERE canonical_name = $1 AND entity_type = $2
        """, canonical_name, entity_type)

        if existing:
            # Update mention count and last_seen
            await conn.execute("""
                UPDATE core.entities
                SET mention_count = mention_count + 1,
                    last_seen = NOW(),
                    updated_at = NOW()
                WHERE id = $1
            """, existing)
            return existing

        # Create new entity
        entity_id = await conn.fetchval("""
            INSERT INTO core.entities (
                canonical_name, entity_type, semantic_confidence,
                mention_count, first_seen, last_seen, status
            )
            VALUES ($1, $2, $3, 1, NOW(), NOW(), 'stub')
            RETURNING id
        """, canonical_name, entity_type, 0.7)  # Default confidence

        return entity_id


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

                # Upsert link
                await conn.execute("""
                    INSERT INTO core.claim_entities (claim_id, entity_id, mention_context)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (claim_id, entity_id) DO NOTHING
                """, claim_uuid, entity_uuid, claim['text'][:500])
                total_links += 1

        logger.info(f"✅ Created {total_links} claim-entity links")


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

        # Store with actual mention counts
        for entity_ref, entity_uuid in entity_mapping.items():
            mention_count = entity_claim_counts.get(entity_ref, 1)

            await conn.execute("""
                INSERT INTO core.page_entities (page_id, entity_id, mention_count)
                VALUES ($1, $2, $3)
                ON CONFLICT (page_id, entity_id)
                DO UPDATE SET mention_count = $3
            """, page_id, entity_uuid, mention_count)


    async def _generate_entity_descriptions(
        self, conn: asyncpg.Connection, entity_mapping: Dict[str, uuid.UUID],
        claims: List[Dict]
    ):
        """Generate entity descriptions using LLM based on claim contexts"""
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

        # Generate LLM-based descriptions (batch to save API calls)
        for entity_ref, entity_uuid in entity_mapping.items():
            if entity_ref not in entity_contexts:
                continue

            contexts = entity_contexts[entity_ref]
            entity_type, entity_name = entity_ref.split(':', 1) if ':' in entity_ref else ('UNKNOWN', entity_ref)

            # Use LLM to generate concise description
            context_text = '\n'.join(contexts[:3])  # Use up to 3 claims for context

            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Based on these mentions of "{entity_name}", write a single concise sentence describing what/who this entity is (max 20 words):

{context_text}

Format: "A/An [type] that/who [brief description]"
Example: "An American political journalism organization"
Only return the description, nothing else."""
                    }],
                    temperature=0.3,
                    max_tokens=60
                )

                profile_summary = response.choices[0].message.content.strip()

                # Remove quotes if present
                profile_summary = profile_summary.strip('"\'')

                await conn.execute("""
                    UPDATE core.entities
                    SET profile_summary = $2
                    WHERE id = $1
                """, entity_uuid, profile_summary)

                logger.debug(f"📝 Generated description for {entity_name}: {profile_summary}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to generate description for {entity_name}: {e}")
                # Fallback: use first claim context (truncated)
                profile_summary = contexts[0][:200] if contexts else None
                if profile_summary:
                    await conn.execute("""
                        UPDATE core.entities
                        SET profile_summary = $2
                        WHERE id = $1
                    """, entity_uuid, profile_summary)

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
        # Get entity details for queuing
        entity_ids = list(entity_mapping.values())

        entities_to_enrich = await conn.fetch("""
            SELECT id, canonical_name, entity_type, profile_summary, wikidata_qid
            FROM core.entities
            WHERE id = ANY($1::uuid[])
            AND wikidata_qid IS NULL
            AND entity_type IN ('PERSON', 'ORGANIZATION', 'LOCATION')
        """, entity_ids)

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
            entity_ref = f"{entity['entity_type']}:{entity['canonical_name']}"
            mentions = entity_contexts.get(entity_ref, [])

            await self.job_queue.enqueue('wikidata_enrichment', {
                'entity_id': str(entity['id']),
                'canonical_name': entity['canonical_name'],
                'entity_type': entity['entity_type'],
                'context': {
                    'description': entity['profile_summary'],
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

    # Initialize worker
    worker = SemanticWorker(db_pool, job_queue)

    logger.info("🧠 SemanticWorker started, listening on queue:semantic:high")

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
