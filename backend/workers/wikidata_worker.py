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
MIN_CONFIDENCE_THRESHOLD = 0.65  # Conservative: only link when reasonably confident
HIGH_AMBIGUITY_THRESHOLD = 0.5   # Flag for review if ambiguity > 0.5


class WikidataWorker:
    """
    Worker that enriches entities with Wikidata QIDs using robust disambiguation
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    def __init__(self, db_pool: asyncpg.Pool, job_queue: JobQueue, worker_id: int = 1):
        self.db_pool = db_pool
        self.job_queue = job_queue
        self.worker_id = worker_id
        self.session: Optional[aiohttp.ClientSession] = None
        # Cache for P279 (subclass of) chains to avoid repeated API calls
        self.subclass_cache: Dict[str, set] = {}

    async def start(self):
        """Start worker loop"""
        logger.info(f"🔗 wikidata-worker-{self.worker_id} started (high-accuracy mode)")

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'HereNews/2.0 (Entity enrichment; https://herenews.com)'
            }
        )

        try:
            while True:
                try:
                    job = await self.job_queue.dequeue('wikidata_enrichment', timeout=5)

                    if job:
                        await self.process_job(job)

                except Exception as e:
                    logger.error(f"❌ Job processing error: {e}", exc_info=True)
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
            async with self.db_pool.acquire() as conn:
                # Check if already enriched
                existing = await conn.fetchval(
                    "SELECT wikidata_qid FROM core.entities WHERE id = $1",
                    entity_id
                )

                if existing:
                    logger.info(f"⏭️  Already enriched: {canonical_name} -> {existing}")
                    return

                # Stage 1: Search Wikidata for candidates
                candidates = await self._search_wikidata(canonical_name)

                if not candidates:
                    logger.warning(f"⚠️  No Wikidata candidates for '{canonical_name}'")
                    await self._mark_entity_checked(conn, entity_id)
                    return

                logger.info(f"📋 Found {len(candidates)} candidates")

                # Stage 2: Filter by label similarity
                filtered_candidates = self._filter_by_label_match(
                    candidates, canonical_name
                )

                if not filtered_candidates:
                    logger.warning(f"⚠️  No label matches for '{canonical_name}'")
                    await self._mark_entity_checked(conn, entity_id)
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
                    await self._mark_entity_checked(conn, entity_id)
                    return

                logger.info(f"✓ {len(typed_candidates)} passed type filtering")

                # Stage 4: Score using structural signals
                scored_candidates = []
                for idx, candidate in enumerate(typed_candidates, start=1):
                    structural_score = await self._score_entity_structural(
                        entity_data=candidate['entity_data'],
                        search_rank=idx,
                        total_candidates=len(typed_candidates)
                    )

                    scored_candidates.append({
                        **candidate,
                        'structural_score': structural_score,
                        'search_rank': idx
                    })

                # Sort by score (highest first)
                scored_candidates.sort(key=lambda x: x['structural_score'], reverse=True)

                # Stage 5: Calculate confidence and ambiguity
                best_candidate = scored_candidates[0]
                ambiguity_score = 0.0

                if len(scored_candidates) > 1:
                    second_best = scored_candidates[1]
                    score_diff = best_candidate['structural_score'] - second_best['structural_score']
                    # Ambiguity: 0.0 = clear winner (diff > 10), 1.0 = tie (diff < 2)
                    ambiguity_score = max(0.0, min(1.0, (10.0 - score_diff) / 10.0))

                # Confidence = structural score normalized + ambiguity penalty
                # Max structural score ≈ 55 (rank 10 + sitelinks 15 + claims 10 + primary 20)
                confidence = min(1.0, best_candidate['structural_score'] / 55.0)
                confidence *= (1.0 - ambiguity_score * 0.3)  # Ambiguity penalty

                # Log ranking for transparency (BEFORE confidence check for debugging)
                logger.info(f"🏆 Candidate ranking:")
                for i, cand in enumerate(scored_candidates[:3], 1):
                    logger.info(
                        f"   {i}. {cand['qid']} - {cand['description'][:60]} "
                        f"(score: {cand['structural_score']:.1f})"
                    )

                # Stage 6: Confidence threshold check
                if confidence < MIN_CONFIDENCE_THRESHOLD:
                    # Show scoring breakdown for low confidence cases
                    ed = best_candidate['entity_data']
                    instance_qids = ed.get('instance_of_qids', [])
                    logger.warning(
                        f"⚠️  Low confidence for '{canonical_name}': "
                        f"{best_candidate['qid']} (confidence={confidence:.2f}, "
                        f"threshold={MIN_CONFIDENCE_THRESHOLD})"
                    )
                    logger.info(
                        f"   Structural breakdown: sitelinks={ed.get('sitelinks_count', 0)}, "
                        f"claims={ed.get('claims_count', 0)}, "
                        f"P31_instance_of={instance_qids}, "
                        f"total_score={best_candidate['structural_score']:.1f}/55.0"
                    )
                    await self._mark_entity_checked(conn, entity_id)
                    return

                if ambiguity_score > HIGH_AMBIGUITY_THRESHOLD:
                    logger.warning(
                        f"📌 High ambiguity: {ambiguity_score:.2f} "
                        f"(score diff: {score_diff:.1f})"
                    )

                # Stage 7: Fetch additional properties (image, coords)
                entity_data = best_candidate['entity_data']
                thumbnail_url = await self._get_wikidata_image(best_candidate['qid'])

                # Fetch aliases for cross-lingual matching
                aliases = await self._fetch_wikidata_aliases(best_candidate['qid'])

                # Stage 8: Update entity with enriched data
                wikidata_properties = {
                    'label': best_candidate['label'],
                    'description': best_candidate['description'],
                    'aliases': aliases,
                    'thumbnail_url': thumbnail_url,
                    'sitelinks_count': entity_data.get('sitelinks_count', 0),
                    'claims_count': entity_data.get('claims_count', 0),
                    'latitude': entity_data.get('latitude'),
                    'longitude': entity_data.get('longitude'),
                    'enriched_at': datetime.utcnow().isoformat(),
                    'confidence': round(confidence, 3),
                    'ambiguity_score': round(ambiguity_score, 3)
                }

                await conn.execute("""
                    UPDATE core.entities
                    SET wikidata_qid = $2,
                        wikidata_properties = $3,
                        names_by_language = COALESCE(names_by_language, '{}'::jsonb) || $4,
                        semantic_confidence = GREATEST(semantic_confidence, $5),
                        status = 'enriched',
                        updated_at = NOW()
                    WHERE id = $1
                """,
                    entity_id,
                    best_candidate['qid'],
                    json.dumps(wikidata_properties),
                    json.dumps(aliases),
                    confidence
                )

                logger.info(
                    f"✅ Enriched {canonical_name} -> {best_candidate['qid']} "
                    f"(confidence={confidence:.2f}, ambiguity={ambiguity_score:.2f})"
                )

        except Exception as e:
            logger.error(f"❌ Failed to enrich {canonical_name}: {e}", exc_info=True)

    async def _search_wikidata(self, name: str, language: str = 'en') -> List[Dict]:
        """
        Search Wikidata for entity candidates

        Returns list of candidates with QID, label, description
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': language,
            'search': name,
            'limit': 10,
            'type': 'item'
        }

        try:
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

    def _filter_by_label_match(
        self, candidates: List[Dict], canonical_name: str
    ) -> List[Dict]:
        """
        Filter candidates by label similarity

        Uses fuzzy matching to handle minor variations
        """
        filtered = []
        canonical_lower = canonical_name.lower().strip()

        for candidate in candidates:
            label = candidate['label'].lower().strip()

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

    async def _mark_entity_checked(self, conn: asyncpg.Connection, entity_id: uuid.UUID):
        """Mark entity as checked even if no Wikidata match found"""
        await conn.execute("""
            UPDATE core.entities
            SET status = 'checked',
                updated_at = NOW()
            WHERE id = $1
        """, entity_id)

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

    worker = WikidataWorker(db_pool, job_queue, worker_id=worker_id)
    logger.info(f"🔗 Starting Wikidata enrichment worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        await db_pool.close()
        await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
