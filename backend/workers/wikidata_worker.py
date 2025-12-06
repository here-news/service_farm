"""
Wikidata Enrichment Worker - High Accuracy Entity Linking

Links entities to Wikidata QIDs using robust disambiguation.

Design Philosophy (based on Gen1):
- ACCURACY OVER SPEED: Wrong QIDs poison event merging
- Multi-stage filtering: label match → type match → structural scoring
- Conservative thresholds: Only link when confident
- Rich logging: Track disambiguation decisions

Architecture:
1. Triggered after semantic worker creates entities
2. Search Wikidata API for candidate matches
3. Filter by entity type (P31 instance-of)
4. Score using structural signals (notability, richness, relevance)
5. Only update entity if confidence > threshold

This enables:
- Cross-article entity resolution (same QID = same entity)
- Authoritative identifiers for event merging
- Rich entity metadata (aliases, coords, images)
"""
import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import aiohttp
import asyncpg
from rapidfuzz import fuzz

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import JobQueue
from services.neo4j_service import Neo4jService
from repositories.entity_repository import EntityRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Wikidata Primary Entity Types (P31 - instance of)
# These represent top-level, primary entities (not derivatives or meta-entities)
PRIMARY_INSTANCE_QIDS = {
    'Q5',        # human
    'Q6256',     # country
    'Q515',      # city
    'Q532',      # village
    'Q486972',   # human settlement
    'Q3918',     # university
    'Q4830453',  # business
    'Q783794',   # company
    'Q891723',   # public company
    'Q43229',    # organization
    'Q7210356',  # political organization
    'Q48204',    # nonprofit organization
    'Q2085381',  # publisher
    'Q11032',    # newspaper
    'Q47459',    # armed forces
    'Q17149090', # military organization
    'Q41710',    # ethnic group
    'Q82794',    # geographic region
    'Q35657',    # U.S. state
    'Q7275',     # state (country subdivision)
    'Q24698',    # province
    'Q23442',    # island
    'Q8502',     # mountain
    'Q4022',     # river
    'Q23397',    # lake
    'Q41176',    # building
}

# Derivative/Meta Entity Types (to penalize)
DERIVATIVE_INSTANCE_QIDS = {
    'Q215380',   # music band
    'Q4438121',  # sports organization
    'Q15944511', # sports season
    'Q13393265', # sports team season
    'Q27020041', # season of television series
    'Q21191270', # television series episode
    'Q13442814', # scientific article
    'Q191067',   # article (generic)
    'Q4167836',  # Wikimedia category
    'Q13406463', # Wikimedia list article
}

# Entity Type to Wikidata Classes (using P279 subclass-of hierarchy)
# Hybrid approach:
# - Top-level root classes (Q43229 for org) enable P279 traversal for ANY subtype
# - Common specific types added for faster direct matching (no API calls needed)
ENTITY_TYPE_FILTERS = {
    'PERSON': {
        'Q5',  # human
    },
    'ORGANIZATION': {
        'Q43229',    # organization (root - covers all via P279 traversal)
        # Common specific org types (for fast matching without hierarchy lookup)
        'Q783794',   # company
        'Q4830453',  # business
        'Q3918',     # university
        'Q11032',    # newspaper
        'Q7210356',  # political organization
        'Q47459',    # armed forces
        'Q17149090', # military organization
        'Q891723',   # public company
        'Q2085381',  # publisher
        'Q48204',    # nonprofit organization
    },
    'GPE': {  # Geo-political entities
        'Q56061',    # administrative territorial entity (root)
        'Q486972',   # human settlement (root)
        # Common specific types
        'Q6256',     # country
        'Q515',      # city
        'Q532',      # village
        'Q35657',    # U.S. state
        'Q7275',     # state (subdivision)
        'Q24698',    # province
    },
    'LOCATION': {  # Physical/geographic locations
        'Q2221906',  # geographic location (root)
        'Q618123',   # geographical feature (root)
        # Common specific types
        'Q23442',    # island
        'Q8502',     # mountain
        'Q4022',     # river
        'Q23397',    # lake
        'Q82794',    # geographic region
        'Q41176',    # building
    }
}

# Maximum depth for P279 (subclass of) traversal to prevent infinite loops
MAX_SUBCLASS_DEPTH = 5

# Minimum confidence threshold for linking
MIN_CONFIDENCE_THRESHOLD = 0.60  # Moderate threshold for context-aware matching (lowered from 0.65)
HIGH_AMBIGUITY_THRESHOLD = 0.5   # Flag for review if ambiguity > 0.5


class WikidataWorker:
    """
    Worker that enriches entities with Wikidata QIDs and merges duplicates

    Combines two responsibilities:
    1. Entity enrichment - Links entities to Wikidata QIDs with context-aware disambiguation
    2. Entity merging - Consolidates duplicate entities sharing same QID

    This unified approach ensures:
    - Merge runs after enrichment batches (efficient)
    - No separate worker to manage
    - Natural flow: enrich → merge
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    MERGE_BATCH_SIZE = 10  # Run merge after every N enrichments

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, neo4j_service: Neo4jService = None, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.session: Optional[aiohttp.ClientSession] = None
        # Cache for P279 (subclass of) chains to avoid repeated API calls
        self.subclass_cache: Dict[str, set] = {}

        # Track enrichments to trigger merge
        self.enrichments_since_merge = 0

        # Initialize Neo4j and repository
        self.neo4j = neo4j_service or Neo4jService()
        self.entity_repo = EntityRepository(db_pool, self.neo4j)

    async def start(self):
        """Start worker loop with enrichment and periodic merging"""
        logger.info(f"🔗 wikidata-worker-{self.worker_id} started (enrichment + merge mode)")

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'HereNews/2.0 (Entity enrichment; https://herenews.com)'
            }
        )

        retry_count = 0
        max_retries = 5
        base_backoff = 2  # seconds
        last_merge_time = asyncio.get_event_loop().time()
        merge_interval = 300  # Run merge every 5 minutes if idle

        try:
            while True:
                try:
                    job = await self.job_queue.dequeue('wikidata_enrichment', timeout=5)

                    if job:
                        await self.process_job(job)
                        # Reset retry counter on successful processing
                        retry_count = 0

                        # Trigger merge after batch of enrichments
                        self.enrichments_since_merge += 1
                        if self.enrichments_since_merge >= self.MERGE_BATCH_SIZE:
                            logger.info(f"🔄 Triggering merge after {self.enrichments_since_merge} enrichments")
                            await self.run_merge_pass()
                            self.enrichments_since_merge = 0
                            last_merge_time = asyncio.get_event_loop().time()
                    else:
                        # No job - check if we should run periodic merge
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_merge_time >= merge_interval:
                            logger.info(f"⏰ Running periodic merge check...")
                            await self.run_merge_pass()
                            last_merge_time = current_time

                except (asyncpg.PostgresConnectionError,
                        asyncpg.InterfaceError,
                        ConnectionRefusedError,
                        ConnectionResetError) as e:
                    # DB connection errors - use exponential backoff
                    retry_count += 1
                    if retry_count <= max_retries:
                        backoff = base_backoff * (2 ** (retry_count - 1))
                        logger.warning(
                            f"⚠️  DB connection error (retry {retry_count}/{max_retries}): {e}"
                        )
                        logger.info(f"🔄 Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(
                            f"❌ Max retries ({max_retries}) exceeded for DB connection, resetting counter"
                        )
                        retry_count = 0
                        await asyncio.sleep(base_backoff)

                except Exception as e:
                    # Other errors - log and continue with short delay
                    logger.error(f"❌ Job processing error: {e}", exc_info=True)
                    retry_count = 0
                    await asyncio.sleep(5)

        finally:
            if self.session:
                await self.session.close()

    async def process_job(self, job: Dict):
        """
        Process a Wikidata enrichment job with robust disambiguation

        Job format: {
            'entity_id': 'uuid',
            'canonical_name': 'Israel',
            'entity_type': 'LOCATION',
            'context': {
                'description': 'LLM-generated description',
                'mentions': ['claim1', 'claim2']
            }
        }
        """
        entity_id = uuid.UUID(job['entity_id'])
        canonical_name = job['canonical_name']
        entity_type = job['entity_type']
        context = job.get('context', {})

        logger.info(f"🔍 Processing: {canonical_name} ({entity_type})")

        try:
            # Check if already enriched using repository
            entity = await self.entity_repo.get_by_id(entity_id)

            if not entity:
                logger.warning(f"⚠️  Entity {entity_id} not found in Neo4j")
                return

            if entity.wikidata_qid and entity.wikidata_image:
                logger.info(f"⏭️  Already enriched: {canonical_name} -> {entity.wikidata_qid}")
                return

            # If entity has QID but missing image, just fetch image
            if entity.wikidata_qid and not entity.wikidata_image:
                logger.info(f"🖼️  Fetching image for existing QID: {entity.wikidata_qid}")
                thumbnail_url = await self._get_wikidata_image(entity.wikidata_qid)

                if thumbnail_url:
                    # Update only the image
                    metadata = entity.metadata or {}
                    metadata['thumbnail_url'] = thumbnail_url

                    await self.entity_repo.enrich(
                        entity_id=entity_id,
                        wikidata_qid=entity.wikidata_qid,
                        wikidata_label=entity.wikidata_label or canonical_name,
                        wikidata_description=entity.wikidata_description or '',
                        confidence=entity.confidence,
                        aliases=entity.aliases,
                        metadata=metadata
                    )
                    logger.info(f"✅ Added image to {canonical_name} ({entity.wikidata_qid})")
                else:
                    logger.info(f"ℹ️  No image found for {entity.wikidata_qid}")

                return

            # Stage 1: Search Wikidata for candidates (full-text search with location)
            # If context mentions a location, use full-text search to find entities by description
            location = self._extract_location_from_context(context) if context else None

            if location:
                enriched_query = f"{canonical_name} {location}"
                logger.info(f"🔍 Full-text search with location: '{enriched_query}'")
                # Use full-text search which searches descriptions, not just labels
                candidates = await self._search_wikidata(enriched_query, use_fulltext=True)

                # Fallback: If full-text search fails, try entity search (label-based)
                if not candidates:
                    logger.info(f"   No results from full-text, trying label search")
                    candidates = await self._search_wikidata(canonical_name, use_fulltext=False)
            else:
                # No location context - use standard entity search (label-based)
                candidates = await self._search_wikidata(canonical_name, use_fulltext=False)

            if not candidates:
                logger.warning(f"⚠️  No Wikidata candidates for '{canonical_name}'")
                await self.entity_repo.mark_checked(entity_id)
                return

            logger.info(f"📋 Found {len(candidates)} candidates")

            # Stage 2: Filter by label similarity (with alias lookup for PERSON)
            filtered_candidates = await self._filter_by_label_match(
                candidates, canonical_name, entity_type
            )

            if not filtered_candidates:
                logger.warning(f"⚠️  No label matches for '{canonical_name}'")
                await self.entity_repo.mark_checked(entity_id)
                return

            logger.info(f"✓ {len(filtered_candidates)} passed label matching")

            # Stage 3: Fetch full entity data and filter by type
            typed_candidates = []
            for candidate in filtered_candidates:
                entity_data = await self._fetch_wikidata_entity(candidate['qid'])

                # Type filtering (CRITICAL for accuracy)
                # Uses P279 (subclass of) hierarchy traversal for comprehensive matching
                if entity_type in ENTITY_TYPE_FILTERS:
                    expected_types = ENTITY_TYPE_FILTERS[entity_type]
                    instance_of_qids = entity_data.get('instance_of_qids', [])

                    # Check type match using hierarchy traversal
                    has_matching_type = await self._check_type_match(
                        instance_of_qids, expected_types
                    )

                    if not has_matching_type:
                        instance_labels = entity_data.get('instance_of_labels', [])
                        instance_str = ', '.join(instance_labels) if instance_labels else 'unknown'
                        qid_str = ', '.join(instance_of_qids[:3])
                        logger.debug(
                            f"  ⏭️  Skipping {candidate['qid']} - wrong type "
                            f"(expected {entity_type}, got {instance_str} [{qid_str}])"
                        )
                        continue

                typed_candidates.append({
                    **candidate,
                    'entity_data': entity_data
                })

            if not typed_candidates:
                logger.warning(
                    f"⚠️  No candidates matched expected type ({entity_type})"
                )
                await self.entity_repo.mark_checked(entity_id)
                return

            logger.info(f"✓ {len(typed_candidates)} passed type filtering")

            # Stage 4: Score using structural signals + context
            scored_candidates = []
            for idx, candidate in enumerate(typed_candidates, start=1):
                structural_score = await self._score_entity_structural(
                    entity_data=candidate['entity_data'],
                    search_rank=idx,
                    total_candidates=len(typed_candidates)
                )

                # Add context-aware scoring if available
                context_score = 0.0
                if context:
                    context_score = await self._score_entity_context(
                        entity_data=candidate['entity_data'],
                        context=context,
                        canonical_name=canonical_name
                    )

                # Hybrid scoring: structural first, then context confirmation
                # Stage 1: Pure structural (Wikidata signals) - cheap, reliable
                # Stage 2: Context boost (semantic) - expensive, only for disambiguation
                total_score = structural_score  # Base score from Wikidata

                scored_candidates.append({
                    **candidate,
                    'structural_score': structural_score,
                    'context_score': context_score,
                    'total_score': total_score,
                    'search_rank': idx
                })

            # Stage 5: Bayesian inference (structural prior + context likelihood)

            # Normalize structural scores to get priors P(candidate)
            # Use softmax to convert scores to probability distribution
            import math

            max_struct = max(c['structural_score'] for c in scored_candidates)
            exp_scores = [math.exp(c['structural_score'] - max_struct) for c in scored_candidates]
            sum_exp = sum(exp_scores)

            for i, cand in enumerate(scored_candidates):
                cand['prior'] = exp_scores[i] / sum_exp  # P(candidate) from structural

            if context:
                # Bayesian update: P(candidate|context) ∝ P(context|candidate) × P(candidate)
                # P(context|candidate) = context_score normalized as likelihood

                logger.info(f"📊 Bayesian inference with {len(scored_candidates)} candidates:")

                # Normalize context scores to likelihoods P(context|candidate)
                # Use softmax to handle negative context scores
                max_ctx = max(c['context_score'] for c in scored_candidates)
                exp_ctx = [math.exp(c['context_score'] - max_ctx) for c in scored_candidates]
                sum_exp_ctx = sum(exp_ctx)

                # Calculate posterior: P(candidate|context) ∝ P(context|candidate) × P(candidate)
                posteriors = []
                for i, cand in enumerate(scored_candidates):
                    likelihood = exp_ctx[i] / sum_exp_ctx  # P(context|candidate)
                    posterior = likelihood * cand['prior']  # Unnormalized posterior
                    posteriors.append(posterior)

                # Normalize posteriors to sum to 1
                sum_posterior = sum(posteriors)
                for i, cand in enumerate(scored_candidates):
                    cand['posterior'] = posteriors[i] / sum_posterior  # P(candidate|context)
                    cand['total_score'] = cand['posterior']  # Use posterior as total score

                    logger.info(
                        f"   {cand['qid']}: prior={cand['prior']:.3f}, "
                        f"ctx_score={cand['context_score']:.1f}, "
                        f"posterior={cand['posterior']:.3f}"
                    )

                # Sort by posterior probability
                scored_candidates.sort(key=lambda x: x['posterior'], reverse=True)
                best_candidate = scored_candidates[0]

                # Decision: Accept if posterior > threshold (e.g., 0.7)
                # This implements "given context, we should be able to solve it"
                POSTERIOR_THRESHOLD = 0.70

                if best_candidate['posterior'] >= POSTERIOR_THRESHOLD:
                    confidence = best_candidate['posterior']  # Use posterior as confidence
                    logger.info(f"   ✅ Bayesian decision: P={confidence:.3f} >= {POSTERIOR_THRESHOLD}")
                else:
                    confidence = 0.0  # Not confident enough even with context
                    logger.info(
                        f"   ❌ Bayesian rejection: P={best_candidate['posterior']:.3f} < {POSTERIOR_THRESHOLD}"
                    )
                    if len(scored_candidates) > 1:
                        logger.info(
                            f"      Ambiguous: top 2 posteriors: "
                            f"{best_candidate['posterior']:.3f} vs {scored_candidates[1]['posterior']:.3f}"
                        )

            else:
                # No context - use structural prior only
                logger.info(f"⚠️  No context available, using structural prior only")

                # Sort by prior (structural)
                scored_candidates.sort(key=lambda x: x['prior'], reverse=True)
                best_candidate = scored_candidates[0]

                for cand in scored_candidates:
                    cand['total_score'] = cand['prior']

                # Accept if prior is strong enough (e.g., > 0.6)
                PRIOR_THRESHOLD = 0.60
                if best_candidate['prior'] >= PRIOR_THRESHOLD:
                    confidence = best_candidate['prior']
                    logger.info(f"   ✅ Structural prior: P={confidence:.3f} >= {PRIOR_THRESHOLD}")
                else:
                    confidence = 0.0
                    logger.info(f"   ❌ Structural prior too weak: P={best_candidate['prior']:.3f} < {PRIOR_THRESHOLD}")

            # Log final ranking
            logger.info(f"🏆 Final ranking:")
            for i, cand in enumerate(scored_candidates[:3], 1):
                prob_str = f"P={cand.get('posterior', cand.get('prior', 0.0)):.3f}"
                logger.info(
                    f"   {i}. {cand['qid']} - {cand['description'][:60]} ({prob_str})"
                )

            # Stage 6: Confidence threshold check (Bayesian decision already made)
            if confidence == 0.0:
                # Bayesian inference rejected this match
                ed = best_candidate['entity_data']
                logger.warning(
                    f"⚠️  Rejected '{canonical_name}': {best_candidate['qid']} "
                    f"(posterior={best_candidate.get('posterior', 0.0):.3f} too low)"
                )
                await self.entity_repo.mark_checked(entity_id)
                return

            # If we got here, Bayesian inference accepted the match (confidence > 0)
            # confidence is already set to posterior probability

            if confidence < 0.85:  # Log cases with moderate confidence
                logger.info(
                    f"📌 Moderate confidence: P={confidence:.3f} "
                    f"(accepted but consider review)"
                )

            # Calculate ambiguity from posterior distribution
            # Low ambiguity = winner has high posterior and runner-up is far behind
            # High ambiguity = multiple candidates have similar posteriors
            if len(scored_candidates) > 1:
                top_posterior = scored_candidates[0].get('posterior', scored_candidates[0].get('prior', 0.0))
                second_posterior = scored_candidates[1].get('posterior', scored_candidates[1].get('prior', 0.0))
                # Ambiguity = how close is the runner-up? (0.0 = clear winner, 1.0 = tie)
                ambiguity_score = second_posterior / top_posterior if top_posterior > 0 else 0.0
            else:
                ambiguity_score = 0.0  # Single candidate = unambiguous

            # Stage 7: Fetch additional properties (image, coords)
            entity_data = best_candidate['entity_data']
            thumbnail_url = await self._get_wikidata_image(best_candidate['qid'])

            # Fetch aliases for cross-lingual matching
            aliases_dict = await self._fetch_wikidata_aliases(best_candidate['qid'])
            # Flatten aliases dict to simple list for Neo4j (Neo4j doesn't support nested maps)
            aliases = []
            for lang_aliases in aliases_dict.values():
                aliases.extend(lang_aliases)

            # Stage 8: Update entity using repository
            metadata = {
                'thumbnail_url': thumbnail_url,
                'sitelinks_count': entity_data.get('sitelinks_count', 0),
                'claims_count': entity_data.get('claims_count', 0),
                'latitude': entity_data.get('latitude'),
                'longitude': entity_data.get('longitude'),
                'enriched_at': datetime.utcnow().isoformat(),
                'ambiguity_score': round(ambiguity_score, 3)
            }

            await self.entity_repo.enrich(
                entity_id=entity_id,
                wikidata_qid=best_candidate['qid'],
                wikidata_label=best_candidate['label'],
                wikidata_description=best_candidate['description'],
                confidence=confidence,
                aliases=aliases,
                metadata=metadata
            )

            logger.info(
                f"✅ Enriched {canonical_name} -> {best_candidate['qid']} "
                f"(confidence={confidence:.2f}, ambiguity={ambiguity_score:.2f})"
            )

        except Exception as e:
            logger.error(f"❌ Failed to enrich {canonical_name}: {e}", exc_info=True)

    def _extract_location_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Extract geographic location from entity description

        Examples:
        - "Hong Kong political figure" → "Hong Kong"
        - "American mathematician" → "American"
        - "governmental agency" → None

        Returns single location term or None
        """
        if not context:
            return None

        description = context.get('description', '')
        if not description:
            return None

        # Common geographic indicators (places, countries, regions)
        # These are likely to be in Wikidata labels/aliases
        locations = [
            'Hong Kong', 'China', 'United States', 'American', 'British', 'Japanese',
            'Korean', 'German', 'French', 'Canadian', 'Australian', 'Indian',
            'European', 'Asian', 'African', 'London', 'Tokyo', 'Singapore',
            'Beijing', 'Shanghai', 'Taiwan', 'Macau'
        ]

        desc_lower = description.lower()
        for loc in locations:
            if loc.lower() in desc_lower:
                return loc

        return None

    async def _search_wikidata(self, name: str, language: str = 'en', use_fulltext: bool = False) -> List[Dict]:
        """
        Search Wikidata for entity candidates

        Args:
            name: Search query
            language: Language code
            use_fulltext: If True, use full-text search (searches descriptions, not just labels)

        Returns list of candidates with QID, label, description
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            if use_fulltext:
                # Full-text search - searches entity descriptions and content
                # Better for context-enhanced queries like "John Lee Hong Kong"
                params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': name,
                    'format': 'json',
                    'srlimit': 10
                }

                async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Wikidata full-text search error: {resp.status}")
                        return []

                    data = await resp.json()
                    results = data.get('query', {}).get('search', [])

                    # Extract QIDs and fetch entity details
                    qids = [r['title'] for r in results if r['title'].startswith('Q')]

                    if not qids:
                        return []

                    # Batch fetch entity details
                    candidates = []
                    for qid in qids[:10]:
                        entity_params = {
                            'action': 'wbgetentities',
                            'ids': qid,
                            'format': 'json',
                            'languages': language,
                            'props': 'labels|descriptions'
                        }

                        async with self.session.get(self.WIKIDATA_API, params=entity_params) as entity_resp:
                            if entity_resp.status == 200:
                                entity_data = await entity_resp.json()
                                if 'entities' in entity_data and qid in entity_data['entities']:
                                    entity = entity_data['entities'][qid]
                                    label = entity.get('labels', {}).get(language, {}).get('value', '')
                                    description = entity.get('descriptions', {}).get(language, {}).get('value', '')

                                    # Quality check: skip meta entities
                                    generic_terms = [
                                        'wikimedia', 'disambiguation', 'category',
                                        'template', 'list of', 'wikipedia', 'scientific article'
                                    ]
                                    if description and not any(term in description.lower() for term in generic_terms):
                                        candidates.append({
                                            'qid': qid,
                                            'label': label or qid,
                                            'description': description
                                        })

                    return candidates

            else:
                # Entity search - searches only labels/aliases
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'language': language,
                    'search': name,
                    'limit': 10,
                    'type': 'item'
                }

                async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Wikidata API error: {resp.status}")
                        return []

                    data = await resp.json()
                    results = data.get('search', [])

                    candidates = []
                    for result in results:
                        # Quality check: skip meta entities
                        description = result.get('description', '')
                        generic_terms = [
                            'wikimedia', 'disambiguation', 'category',
                            'template', 'list of', 'wikipedia'
                        ]
                        if any(term in description.lower() for term in generic_terms):
                            continue

                        candidates.append({
                            'qid': result['id'],
                            'label': result.get('label', ''),
                            'description': description
                        })

                    return candidates

        except Exception as e:
            logger.error(f"Wikidata search error: {e}")
            return []

    async def _filter_by_label_match(
        self, candidates: List[Dict], canonical_name: str, entity_type: str = None
    ) -> List[Dict]:
        """
        Filter candidates by label similarity

        For PERSON entities, also checks Wikidata aliases if label doesn't match
        (fixes issue where "John Lee" wouldn't match "John Lee Ka-chiu")
        """
        filtered = []
        canonical_lower = canonical_name.lower().strip()

        for candidate in candidates:
            label = candidate['label'].lower().strip()
            matched = False

            # Exact match
            if label == canonical_lower:
                filtered.append(candidate)
                continue

            # Substring match
            if canonical_lower in label or label in canonical_lower:
                filtered.append(candidate)
                continue

            # Fuzzy match (85% threshold)
            if fuzz.ratio(canonical_lower, label) >= 85:
                filtered.append(candidate)
                continue

            # For PERSON entities: Check aliases if label didn't match
            # This handles cases like "John Lee" vs "John Lee Ka-chiu"
            if entity_type == 'PERSON':
                qid = candidate['qid']
                aliases_dict = await self._fetch_wikidata_aliases(qid, languages=['en', 'zh'])

                if aliases_dict:
                    for lang, aliases in aliases_dict.items():
                        for alias in aliases:
                            alias_lower = alias.lower().strip()
                            if (alias_lower == canonical_lower or
                                canonical_lower in alias_lower or
                                alias_lower in canonical_lower):
                                logger.debug(f"      ✅ Alias match: '{alias}' ({lang}) for {qid}")
                                filtered.append(candidate)
                                matched = True
                                break
                        if matched:
                            break

        return filtered

    async def _fetch_wikidata_entity(self, qid: str) -> Dict[str, Any]:
        """
        Fetch full Wikidata entity data for disambiguation

        Returns dict with instance_of, sitelinks, claims_count, coords, etc.
        """
        if not self.session:
            return {}

        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'languages': 'en',
            'props': 'claims|labels|descriptions|sitelinks'
        }

        try:
            async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if 'entities' not in data or qid not in data['entities']:
                    return {}

                entity = data['entities'][qid]
                claims = entity.get('claims', {})

                # P31 = instance of (CRITICAL for type filtering)
                instance_of_qids = []
                instance_of_labels = []
                if 'P31' in claims:
                    for claim in claims['P31']:
                        if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                            instance_qid = claim['mainsnak']['datavalue']['value']['id']
                            instance_of_qids.append(instance_qid)
                            # Optionally fetch labels (adds latency, could cache)
                            label = await self._fetch_wikidata_label(instance_qid)
                            if label:
                                instance_of_labels.append(label)

                # Sitelinks count (notability signal)
                sitelinks_count = len(entity.get('sitelinks', {}))

                # Claims count (richness signal)
                claims_count = len(claims)

                # P625 = coordinate location
                latitude = None
                longitude = None
                if 'P625' in claims:
                    for claim in claims['P625']:
                        if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                            coords = claim['mainsnak']['datavalue']['value']
                            latitude = coords.get('latitude')
                            longitude = coords.get('longitude')
                            break

                return {
                    'qid': qid,
                    'description': entity.get('descriptions', {}).get('en', {}).get('value', ''),
                    'instance_of_qids': instance_of_qids,
                    'instance_of_labels': instance_of_labels,
                    'sitelinks_count': sitelinks_count,
                    'claims_count': claims_count,
                    'latitude': latitude,
                    'longitude': longitude
                }

        except Exception as e:
            logger.error(f"Failed to fetch entity {qid}: {e}")
            return {}

    async def _fetch_wikidata_label(self, qid: str) -> Optional[str]:
        """Fetch English label for a Wikidata QID"""
        if not self.session:
            return None

        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'languages': 'en',
            'props': 'labels'
        }

        try:
            async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                if 'entities' in data and qid in data['entities']:
                    return data['entities'][qid].get('labels', {}).get('en', {}).get('value')

        except Exception:
            pass

        return None

    async def _fetch_wikidata_aliases(
        self, qid: str, languages: List[str] = ['en', 'zh', 'es', 'ar', 'he']
    ) -> Dict[str, List[str]]:
        """
        Fetch aliases for entity in multiple languages

        Returns: {'en': ['alias1', 'alias2'], 'zh': ['别名1'], ...}
        """
        if not self.session:
            return {}

        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'languages': '|'.join(languages),
            'props': 'labels|aliases'
        }

        try:
            async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()
                if 'entities' not in data or qid not in data['entities']:
                    return {}

                entity = data['entities'][qid]
                result = {}

                # Collect labels
                for lang in languages:
                    labels = []

                    # Main label
                    if lang in entity.get('labels', {}):
                        labels.append(entity['labels'][lang]['value'])

                    # Aliases
                    if lang in entity.get('aliases', {}):
                        for alias in entity['aliases'][lang]:
                            labels.append(alias['value'])

                    if labels:
                        result[lang] = labels

                return result

        except Exception as e:
            logger.error(f"Failed to fetch aliases for {qid}: {e}")
            return {}

    async def _get_wikidata_image(self, qid: str) -> Optional[str]:
        """
        Fetch thumbnail image URL from Wikidata entity

        Returns Commons thumbnail URL (400px width)
        """
        if not self.session:
            return None

        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'props': 'claims'
        }

        try:
            async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                if 'entities' not in data or qid not in data['entities']:
                    return None

                claims = data['entities'][qid].get('claims', {})

                # P18 = image
                if 'P18' in claims and len(claims['P18']) > 0:
                    claim = claims['P18'][0]
                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                        filename = claim['mainsnak']['datavalue']['value']
                        # Convert to Commons thumbnail URL
                        filename_clean = filename.replace(' ', '_')
                        return f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename_clean}?width=400"

        except Exception:
            pass

        return None

    async def _score_entity_context(
        self,
        entity_data: Dict[str, Any],
        context: Dict[str, Any],
        canonical_name: str
    ) -> float:
        """
        Score Wikidata entity using context from claims

        Uses entity description to compare against Wikidata description.

        Returns:
            float: Context score (0-10, higher = better match)
        """
        score = 0.0

        entity_description = context.get('description', '')
        wikidata_desc = entity_data.get('description', '').lower()

        # Simplified: Just check if key terms from entity description appear in Wikidata description
        if entity_description and wikidata_desc:
            from rapidfuzz import fuzz

            # Token set ratio handles word order and subset matches well
            similarity = fuzz.token_set_ratio(
                entity_description.lower(),
                wikidata_desc
            ) / 100.0

            score = similarity * 10.0

            logger.debug(
                f"      Context: similarity={similarity:.2f} -> score={score:.1f}/10.0"
            )

        return score

    async def _score_entity_structural(
        self,
        entity_data: Dict[str, Any],
        search_rank: int,
        total_candidates: int
    ) -> float:
        """
        Score Wikidata entity using structural signals (Gen1 logic)

        Signals:
        1. Search rank (Wikidata's relevance ordering)
        2. Sitelinks count (notability - # of Wikipedia language versions)
        3. Claims count (data richness - # of statements)
        4. Instance type (primary vs derivative entities)

        Returns:
            float: Structural score (0-55+, higher = better match)
        """
        score = 0.0

        # Signal 1: Search Rank (Wikidata's relevance)
        rank_score = ((total_candidates - search_rank + 1) / total_candidates) * 10.0
        score += rank_score

        # Signal 2: Sitelinks Count (Notability)
        sitelinks_count = entity_data.get('sitelinks_count', 0)
        sitelinks_score = min(sitelinks_count / 50.0, 1.0) * 15.0
        score += sitelinks_score

        # Signal 3: Claims Count (Data Richness)
        claims_count = entity_data.get('claims_count', 0)
        claims_score = min(claims_count / 100.0, 1.0) * 10.0
        score += claims_score

        # Signal 4: Instance Type (Primary vs Derivative)
        # Uses P279 hierarchy traversal to check if entity belongs to primary/derivative classes
        instance_of_qids = entity_data.get('instance_of_qids', [])

        is_primary = False
        is_derivative = False

        # Check each instance_of QID via hierarchy
        for qid in instance_of_qids:
            # Get full subclass hierarchy for this QID
            hierarchy = await self._fetch_subclass_hierarchy(qid)

            # Check if hierarchy contains primary types
            if hierarchy & PRIMARY_INSTANCE_QIDS:
                is_primary = True
                break

            # Check if hierarchy contains derivative types
            if hierarchy & DERIVATIVE_INSTANCE_QIDS:
                is_derivative = True

        if is_primary:
            score += 20.0  # Strong boost for primary entities
        elif is_derivative:
            score -= 15.0  # Penalty for derivative entities

        logger.debug(
            f"      Structural score: {score:.1f} "
            f"(rank={rank_score:.1f}, sitelinks={sitelinks_score:.1f}, "
            f"claims={claims_score:.1f}, primary={is_primary})"
        )

        return score

    async def _fetch_subclass_hierarchy(
        self, qid: str, depth: int = 0, visited: Optional[set] = None
    ) -> set:
        """
        Recursively fetch P279 (subclass of) hierarchy for a QID

        Returns set of all ancestor class QIDs (includes the QID itself)
        Uses caching and depth limits to prevent infinite loops

        Example: Q47459 (armed forces) -> {Q47459, Q17149090, Q43229, ...}
        """
        if visited is None:
            visited = set()

        # Check cache first
        if qid in self.subclass_cache:
            return self.subclass_cache[qid]

        # Prevent infinite loops
        if depth >= MAX_SUBCLASS_DEPTH or qid in visited:
            return {qid}

        visited.add(qid)
        ancestors = {qid}  # Include self

        if not self.session:
            return ancestors

        # Fetch P279 (subclass of) claims
        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'props': 'claims'
        }

        try:
            async with self.session.get(self.WIKIDATA_API, params=params) as resp:
                if resp.status != 200:
                    return ancestors

                data = await resp.json()

                if 'entities' not in data or qid not in data['entities']:
                    return ancestors

                claims = data['entities'][qid].get('claims', {})

                # P279 = subclass of
                if 'P279' in claims:
                    for claim in claims['P279']:
                        if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                            parent_qid = claim['mainsnak']['datavalue']['value']['id']
                            # Recursively fetch parent hierarchy
                            parent_ancestors = await self._fetch_subclass_hierarchy(
                                parent_qid, depth + 1, visited
                            )
                            ancestors.update(parent_ancestors)

        except Exception as e:
            logger.debug(f"Error fetching P279 for {qid}: {e}")

        # Cache the result
        self.subclass_cache[qid] = ancestors
        return ancestors

    async def _check_type_match(
        self, instance_of_qids: List[str], expected_types: set
    ) -> bool:
        """
        Check if any instance_of QIDs match expected types via P279 hierarchy

        Uses hybrid approach:
        1. Fast path: Direct QID match (no API calls)
        2. Slow path: P279 hierarchy traversal (for uncommon subtypes)

        Args:
            instance_of_qids: List of P31 (instance of) QIDs from entity
            expected_types: Set of expected root type QIDs

        Returns:
            True if any instance_of type (or its ancestors) match expected types
        """
        # Fast path: Direct match (most common cases)
        if any(qid in expected_types for qid in instance_of_qids):
            logger.debug(f"  ✓ Direct type match: {set(instance_of_qids) & expected_types}")
            return True

        # Slow path: Check P279 (subclass of) hierarchy
        for qid in instance_of_qids:
            # Fetch full subclass hierarchy for this QID
            hierarchy = await self._fetch_subclass_hierarchy(qid)

            # Check if any ancestor matches expected types
            if hierarchy & expected_types:  # Set intersection
                logger.debug(f"  ✓ Hierarchy type match: {qid} → {hierarchy & expected_types}")
                return True

        return False

    async def run_merge_pass(self):
        """
        Find and merge all duplicate entities using Neo4j

        Consolidates entities that share the same Wikidata QID.
        This handles variants like:
        - "Police" and "Hong Kong Police" (both → Q25859)
        - "Fire Department" and "Fire Services Department" (both → Q1595073)
        """
        try:
            # Find all QIDs with multiple entities from Neo4j
            duplicates = await self.neo4j.find_duplicate_entities()

            if not duplicates:
                logger.info("✓ No duplicate entities found")
                return

            logger.info(f"🔍 Found {len(duplicates)} QIDs with duplicate entities")

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

                # Perform merge in Neo4j
                total_mentions = sum(mention_counts)
                await self.neo4j.merge_entities(
                    canonical_id=canonical_id,
                    duplicate_ids=duplicate_ids,
                    total_mentions=total_mentions
                )

                logger.info(f"✅ Merged → {canonical_name} (total: {total_mentions} mentions)")
                merged_count += len(duplicate_ids)

            logger.info(f"✓ Merge pass complete: {merged_count} entities merged")

        except Exception as e:
            logger.error(f"❌ Merge pass failed: {e}", exc_info=True)

    def _pick_canonical(self, names: List[str], mention_counts: List[int]) -> int:
        """
        Pick the best canonical entity from duplicates

        Priority:
        1. Most mentions (entity used most frequently)
        2. Longest name (usually more specific)
        3. First alphabetically

        Args:
            names: List of entity names
            mention_counts: List of mention counts for each entity

        Returns:
            Index of the canonical entity
        """
        # Find max mentions
        max_mentions = max(mention_counts)

        # Filter to entities with max mentions
        candidates = [
            (i, name, count)
            for i, (name, count) in enumerate(zip(names, mention_counts))
            if count == max_mentions
        ]

        # If tie, pick longest name (more specific)
        if len(candidates) > 1:
            candidates.sort(key=lambda x: (len(x[1]), x[1]), reverse=True)

        return candidates[0][0]


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

    # Initialize Neo4j
    neo4j_service = Neo4jService()
    await neo4j_service.connect()

    worker = WikidataWorker(db_pool, job_queue, neo4j_service=neo4j_service, worker_id=worker_id)
    logger.info(f"🔗 Starting Wikidata enrichment worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()
        await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(main())
