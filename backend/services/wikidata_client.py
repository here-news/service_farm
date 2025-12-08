"""
WikidataClient - Wikidata search with Bayesian disambiguation.

Extracted from wikidata_worker for use during entity identification.
This enables QID resolution DURING the knowledge pipeline (not async after).

Uses:
1. Structural scoring (sitelinks, statements count, search rank)
2. Embedding-based context scoring (OpenAI embeddings)
3. Bayesian inference to combine structural prior + context likelihood

Usage:
    client = WikidataClient()
    result = await client.search_entity(
        name="Samuel Chu",
        entity_type="PERSON",
        context="Hong Kong activist, Campaign for Hong Kong",
        aliases=[]
    )
    # Returns: {'qid': 'Q...', 'label': 'Samuel Chu', 'confidence': 0.85}
"""
import aiohttp
import asyncio
import logging
import math
import os
import re
from typing import Optional, Dict, List, Any
from rapidfuzz import fuzz
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Entity Type to Wikidata root classes (P31 instance-of → P279 subclass-of hierarchy)
# These are checked via hierarchy traversal for rigorous type matching
ENTITY_TYPE_QIDS = {
    'PERSON': {
        'Q5',  # human
    },
    'ORGANIZATION': {
        'Q43229',    # organization (root)
        'Q783794',   # company
        'Q4830453',  # business
        'Q3918',     # university
        'Q11032',    # newspaper
        'Q7210356',  # political organization
        'Q47459',    # armed forces
        'Q48204',    # nonprofit organization
    },
    'LOCATION': {
        'Q56061',    # administrative territorial entity
        'Q486972',   # human settlement
        'Q515',      # city
        'Q3957',     # town
        'Q532',      # village
        'Q82794',    # geographic region
        'Q41176',    # building
        'Q35127',    # geographic location
        'Q27096213', # geographic entity
    },
}

# Max depth for P279 subclass traversal (prevent infinite loops)
MAX_SUBCLASS_DEPTH = 5

# Generic names to skip (won't have useful Wikidata matches)
GENERIC_PATTERNS = [
    r'^block\s*\d+$',
    r'^floor\s*\d+$',
    r'^building\s*\d*$',
    r'^the\s+(government|police|authorities|officials)$',
    r'^(police|authorities|officials|residents|victims)$',
]


class WikidataClient:
    """
    Wikidata search client with Bayesian disambiguation.

    Uses structural scoring + embedding-based context matching
    to accurately disambiguate entities like "Samuel Chu".
    """

    # Bayesian thresholds (probability-based)
    POSTERIOR_THRESHOLD = 0.70  # With context - higher bar (absolute)
    PRIOR_THRESHOLD = 0.60      # Without context - structural only
    TEMPERATURE = 5.0           # Softmax temperature - higher = flatter priors

    # Relative threshold: accept if best is significantly better than 2nd best
    # even when absolute threshold isn't met
    RELATIVE_THRESHOLD = 1.3    # best/second ratio required
    MIN_POSTERIOR = 0.35        # minimum absolute to consider relative

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_url = "https://www.wikidata.org/w/api.php"
        self.headers = {
            'User-Agent': 'HereNews/1.0 (https://here.news; contact@here.news) aiohttp/3.9'
        }
        # OpenAI client for embeddings
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_cache: Dict[str, List[float]] = {}
        # P279 subclass hierarchy cache
        self.subclass_cache: Dict[str, set] = {}

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def search_entity(
        self,
        name: str,
        entity_type: str,
        context: str = "",
        aliases: List[str] = None
    ) -> Optional[Dict]:
        """
        Search Wikidata for an entity match.

        Args:
            name: Entity name (surface form)
            entity_type: PERSON, ORGANIZATION, or LOCATION
            context: Surrounding text for disambiguation
            aliases: Alternative names

        Returns:
            Dict with {qid, label, description, confidence} or None
        """
        await self._ensure_session()

        # Skip generic entities
        if self._is_generic(name, entity_type):
            logger.debug(f"⏭️  Skipping generic: {name}")
            return None

        try:
            # Search Wikidata WITH context for better disambiguation
            # e.g., "John Lee Hong Kong" finds the right John Lee first
            search_query = f"{name} {context}".strip() if context else name
            candidates = await self._search_wikidata(search_query)

            if not candidates:
                # Fallback: try name only
                candidates = await self._search_wikidata(name)

            if not candidates:
                # Try with aliases
                if aliases:
                    for alias in aliases:
                        candidates = await self._search_wikidata(alias)
                        if candidates:
                            break

            if not candidates:
                logger.debug(f"No Wikidata candidates for: {name}")
                return None

            # Filter by P31→P279 type hierarchy
            filtered = await self._filter_by_type(candidates, entity_type)

            if not filtered:
                logger.debug(f"No type-matched candidates for: {name} ({entity_type})")
                return None

            # Score candidates with Bayesian inference
            best = await self._score_and_select(filtered, name, context, entity_type)

            if best and best.get('accepted', False):
                logger.info(f"🔗 Wikidata match: {name} → {best['qid']} ({best['label']}) [P={best['confidence']:.2f}]")
                return best

            best_conf = best['confidence'] if best else 0
            logger.debug(f"No confident match for {name} (best P={best_conf:.2f})")
            return None

        except Exception as e:
            logger.warning(f"Wikidata search failed for {name}: {e}")
            return None

    async def _search_wikidata(self, name: str) -> List[Dict]:
        """
        Search Wikidata using CirrusSearch (full-text search).

        This is the same search the Wikidata web interface uses.
        More powerful than wbsearchentities - finds partial matches.
        """
        # Stage 1: CirrusSearch to find candidate QIDs
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': name,
            'srnamespace': 0,
            'format': 'json',
            'srlimit': 15
        }

        try:
            async with self.session.get(self.api_url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json()
                results = data.get('query', {}).get('search', [])

                if not results:
                    return []

                # Extract QIDs
                qids = [r['title'] for r in results if r['title'].startswith('Q')]
                if not qids:
                    return []

                # Stage 2: Fetch labels and descriptions for QIDs
                return await self._fetch_entity_labels(qids)

        except asyncio.TimeoutError:
            logger.warning(f"Wikidata search timeout for: {name}")
            return []
        except Exception as e:
            logger.warning(f"Wikidata API error: {e}")
            return []

    async def _fetch_entity_labels(self, qids: List[str]) -> List[Dict]:
        """Fetch labels and descriptions for a list of QIDs."""
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(qids[:15]),
            'format': 'json',
            'props': 'labels|descriptions|aliases',
            'languages': 'en'
        }

        try:
            async with self.session.get(self.api_url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json()
                entities = data.get('entities', {})

                candidates = []
                for qid in qids:
                    if qid not in entities or 'missing' in entities[qid]:
                        continue

                    entity = entities[qid]
                    label = entity.get('labels', {}).get('en', {}).get('value', '')
                    description = entity.get('descriptions', {}).get('en', {}).get('value', '')
                    aliases = [a['value'] for a in entity.get('aliases', {}).get('en', [])]

                    candidates.append({
                        'qid': qid,
                        'label': label,
                        'description': description,
                        'aliases': aliases
                    })

                return candidates

        except Exception as e:
            logger.warning(f"Failed to fetch entity labels: {e}")
            return []

    async def _fetch_entity_data(self, qids: List[str]) -> Dict[str, Dict]:
        """
        Fetch full entity data from Wikidata (sitelinks, claims, P31).

        This is critical for proper structural scoring - notable entities like
        "Hong Kong" have 200+ sitelinks while obscure entities have few.
        """
        if not qids:
            return {}

        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(qids[:10]),  # Max 10 at a time
            'format': 'json',
            'props': 'sitelinks|claims'
        }

        try:
            async with self.session.get(self.api_url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()
                entities = data.get('entities', {})

                result = {}
                for qid, entity in entities.items():
                    if 'missing' in entity:
                        continue

                    # Count sitelinks (Wikipedia language versions)
                    sitelinks = entity.get('sitelinks', {})
                    sitelinks_count = len(sitelinks)

                    # Count claims (statements)
                    claims = entity.get('claims', {})
                    claims_count = sum(len(v) for v in claims.values())

                    # Get P31 (instance of) values
                    instance_of_qids = []
                    for claim in claims.get('P31', []):
                        mainsnak = claim.get('mainsnak', {})
                        datavalue = mainsnak.get('datavalue', {})
                        if datavalue.get('type') == 'wikibase-entityid':
                            instance_of_qids.append(datavalue['value']['id'])

                    result[qid] = {
                        'sitelinks_count': sitelinks_count,
                        'claims_count': claims_count,
                        'instance_of_qids': instance_of_qids
                    }

                return result

        except Exception as e:
            logger.warning(f"Failed to fetch entity data: {e}")
            return {}

    def _is_generic(self, name: str, entity_type: str) -> bool:
        """Check if name is too generic for Wikidata search."""
        name_lower = name.lower().strip()

        for pattern in GENERIC_PATTERNS:
            if re.match(pattern, name_lower, re.IGNORECASE):
                return True

        # Very short names are often generic (but allow 3-letter orgs like CNN, BBC)
        if len(name) < 3:
            return True
        if len(name) == 3 and entity_type not in ('PERSON', 'ORGANIZATION'):
            return True

        return False

    async def _filter_by_type(self, candidates: List[Dict], entity_type: str) -> List[Dict]:
        """
        Filter candidates by entity type using P31→P279 hierarchy.

        Uses Wikidata's structured type system for rigorous filtering.
        """
        expected_types = ENTITY_TYPE_QIDS.get(entity_type)
        if not expected_types:
            return candidates

        # Fetch entity data (includes P31 instance_of)
        qids = [c['qid'] for c in candidates]
        entity_data = await self._fetch_entity_data(qids)

        filtered = []
        candidates_with_p31 = 0

        for cand in candidates:
            qid = cand['qid']
            ed = entity_data.get(qid, {})
            instance_of_qids = ed.get('instance_of_qids', [])

            if instance_of_qids:
                candidates_with_p31 += 1

            # Check if candidate's type matches expected via hierarchy
            if await self._check_type_match(instance_of_qids, expected_types):
                cand['_entity_data'] = ed  # Cache for later use
                filtered.append(cand)
            else:
                logger.debug(f"  ⏭️  Type mismatch: {qid} ({cand['label']}) - P31={instance_of_qids}")

        if filtered:
            return filtered

        # Only fallback if most candidates had missing P31 data
        # If candidates had P31 but wrong type, reject all (don't match a plant to a person!)
        if candidates_with_p31 < len(candidates) / 2:
            logger.debug(f"  ⚠️  No type matches but P31 data missing, falling back to top 3")
            return candidates[:3]

        # Most candidates had P31 but none matched - no fallback
        logger.debug(f"  ❌ No type matches (all candidates have wrong P31 types)")
        return []

    async def _check_type_match(self, instance_of_qids: List[str], expected_types: set) -> bool:
        """
        Check if any P31 QIDs match expected types via P279 hierarchy.

        Fast path: direct QID match (no API calls)
        Slow path: P279 subclass traversal
        """
        if not instance_of_qids:
            return False

        # Fast path: direct match
        if any(qid in expected_types for qid in instance_of_qids):
            return True

        # Slow path: check P279 hierarchy
        for qid in instance_of_qids:
            hierarchy = await self._fetch_subclass_hierarchy(qid)
            if hierarchy & expected_types:
                return True

        return False

    async def _fetch_subclass_hierarchy(self, qid: str, depth: int = 0, visited: set = None) -> set:
        """
        Recursively fetch P279 (subclass of) hierarchy for a QID.

        Returns set of all ancestor class QIDs (includes the QID itself).
        """
        if visited is None:
            visited = set()

        # Check cache
        if qid in self.subclass_cache:
            return self.subclass_cache[qid]

        # Prevent infinite loops
        if depth >= MAX_SUBCLASS_DEPTH or qid in visited:
            return {qid}

        visited.add(qid)
        ancestors = {qid}

        try:
            params = {
                'action': 'wbgetentities',
                'ids': qid,
                'format': 'json',
                'props': 'claims'
            }

            async with self.session.get(self.api_url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return ancestors

                data = await resp.json()
                entity = data.get('entities', {}).get(qid, {})
                claims = entity.get('claims', {})

                # P279 = subclass of
                for claim in claims.get('P279', []):
                    mainsnak = claim.get('mainsnak', {})
                    datavalue = mainsnak.get('datavalue', {})
                    if datavalue.get('type') == 'wikibase-entityid':
                        parent_qid = datavalue['value']['id']
                        parent_ancestors = await self._fetch_subclass_hierarchy(
                            parent_qid, depth + 1, visited
                        )
                        ancestors.update(parent_ancestors)

        except Exception as e:
            logger.debug(f"Error fetching P279 for {qid}: {e}")

        # Cache result
        self.subclass_cache[qid] = ancestors
        return ancestors

    async def _score_and_select(
        self,
        candidates: List[Dict],
        name: str,
        context: str,
        entity_type: str
    ) -> Optional[Dict]:
        """
        Score candidates using Bayesian inference.

        All scores are probability-based (0.0-1.0):
        1. Structural score → softmax → prior P(candidate)
        2. Context similarity → likelihood P(context|candidate)
        3. Bayesian update: posterior P(candidate|context) ∝ likelihood × prior

        Returns best candidate with confidence (posterior probability).
        """
        if not candidates:
            return None

        # Fetch full entity data for structural scoring
        qids = [c['qid'] for c in candidates]
        entity_data = await self._fetch_entity_data(qids)

        name_lower = name.lower()
        scored = []
        total_candidates = len(candidates)

        # === Stage 1: Compute raw structural scores ===
        # Focus on NAME MATCH + SEARCH RANK, NOT global popularity (sitelinks)
        # Sitelinks create false confidence for famous but irrelevant entities
        for idx, cand in enumerate(candidates):
            qid = cand['qid']
            ed = entity_data.get(qid, {})

            # Label match (0-20) - exact match is critical
            label_ratio = fuzz.ratio(name_lower, cand['label'].lower()) / 100
            if cand['label'].lower() == name_lower:
                label_score = 20.0
            else:
                label_score = label_ratio * 15.0

            # Search rank (0-10) - Wikidata's relevance ordering
            rank_score = ((total_candidates - idx) / total_candidates) * 10.0

            # NO sitelinks - global popularity ≠ relevance to THIS context
            # NO claims_count - data richness ≠ relevance

            structural_score = label_score + rank_score

            scored.append({
                **cand,
                'structural_score': structural_score,
                'sitelinks_count': ed.get('sitelinks_count', 0),
                'claims_count': ed.get('claims_count', 0),
            })

        # === Stage 2: Convert structural scores to priors via softmax ===
        # Higher temperature → flatter distribution → context has more influence
        max_struct = max(c['structural_score'] for c in scored)
        exp_scores = [
            math.exp((c['structural_score'] - max_struct) / self.TEMPERATURE)
            for c in scored
        ]
        sum_exp = sum(exp_scores)

        for i, cand in enumerate(scored):
            cand['prior'] = exp_scores[i] / sum_exp  # P(candidate) from structural

        # === Stage 3: Compute context likelihood (embedding similarity) ===
        # Prepend entity type to query - lets embeddings capture type compatibility
        has_context = bool(context and context.strip())
        typed_context = f"{entity_type}: {context}" if has_context else ""

        for cand in scored:
            if has_context and cand['description']:
                # Embedding similarity (0-1) as likelihood P(context|candidate)
                similarity = await self._embedding_similarity(typed_context, cand['description'])
                cand['likelihood'] = similarity
            else:
                cand['likelihood'] = 0.5  # Neutral if no context

        # === Stage 4: Bayesian update ===
        if has_context:
            # posterior ∝ likelihood × prior
            posteriors = []
            for cand in scored:
                posterior = cand['likelihood'] * cand['prior']
                posteriors.append(posterior)

            # Normalize posteriors to sum to 1
            sum_posterior = sum(posteriors)
            if sum_posterior > 0:
                for i, cand in enumerate(scored):
                    cand['posterior'] = posteriors[i] / sum_posterior
            else:
                # All posteriors are 0, fall back to priors
                for cand in scored:
                    cand['posterior'] = cand['prior']

            # Sort by posterior
            scored.sort(key=lambda x: x['posterior'], reverse=True)
            threshold = self.POSTERIOR_THRESHOLD
        else:
            # No context - use structural prior only
            for cand in scored:
                cand['posterior'] = cand['prior']
            scored.sort(key=lambda x: x['prior'], reverse=True)
            threshold = self.PRIOR_THRESHOLD

        best = scored[0]
        best['confidence'] = best['posterior']

        # === Acceptance logic ===
        # 1. Absolute threshold: accept if posterior >= threshold
        # 2. Relative threshold: accept if best >> second best (even if below absolute)
        if best['posterior'] >= threshold:
            best['accepted'] = True
            accept_reason = "absolute"
        elif len(scored) > 1 and best['posterior'] >= self.MIN_POSTERIOR:
            # Check relative threshold: best must be significantly better than second
            second = scored[1]['posterior']
            ratio = best['posterior'] / second if second > 0 else float('inf')
            if ratio >= self.RELATIVE_THRESHOLD:
                best['accepted'] = True
                accept_reason = f"relative ({ratio:.2f}x)"
            else:
                best['accepted'] = False
                accept_reason = None
        else:
            best['accepted'] = False
            accept_reason = None

        # === Logging ===
        logger.info(f"📊 Bayesian disambiguation for '{name}' ({len(scored)} candidates):")
        for i, cand in enumerate(scored[:3]):
            marker = "✓" if i == 0 and best['accepted'] else " "
            logger.info(
                f"  {marker} {cand['qid']}: prior={cand['prior']:.3f}, "
                f"lik={cand['likelihood']:.3f}, post={cand['posterior']:.3f} "
                f"({cand['description'][:50]})"
            )

        if best['accepted']:
            logger.info(f"  → Accepted ({accept_reason}): P={best['posterior']:.3f}")
        else:
            logger.info(f"  → Rejected: P={best['posterior']:.3f} < {threshold}")
            if len(scored) > 1:
                second_post = scored[1]['posterior']
                ratio = best['posterior'] / second_post if second_post > 0 else 0
                logger.info(
                    f"     Ambiguous: {best['posterior']:.3f} vs {second_post:.3f} (ratio={ratio:.2f})"
                )

        return best

    async def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        try:
            emb1 = await self._get_embedding(text1)
            emb2 = await self._get_embedding(text2)

            if not emb1 or not emb2:
                return 0.5

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))

            if norm1 == 0 or norm2 == 0:
                return 0.5

            similarity = dot_product / (norm1 * norm2)
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2

        except Exception as e:
            logger.debug(f"Embedding similarity failed: {e}")
            return 0.5

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, with caching."""
        # Truncate long text
        text = text[:500]

        # Check cache
        cache_key = text[:100]
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding

            # Cache it
            self.embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.debug(f"Embedding API failed: {e}")
            return None
