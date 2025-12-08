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

# Entity Type to Wikidata type keywords for filtering
ENTITY_TYPE_KEYWORDS = {
    'PERSON': ['human', 'person', 'politician', 'actor', 'singer', 'athlete', 'activist',
               'journalist', 'writer', 'lawyer', 'businessman', 'researcher', 'professor',
               'executive', 'director', 'chief', 'minister', 'secretary', 'official',
               'photographer', 'artist', 'author', 'born'],
    'ORGANIZATION': ['organization', 'company', 'institution', 'agency', 'party', 'church',
                     'university', 'newspaper', 'network', 'association', 'foundation',
                     'department', 'ministry', 'council', 'commission', 'media', 'press'],
    'LOCATION': ['city', 'district', 'country', 'building', 'estate', 'court', 'place',
                 'region', 'settlement', 'territory', 'province', 'area', 'housing',
                 'neighborhood', 'town', 'village', 'municipality']
}

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

    MIN_CONFIDENCE = 0.65  # Minimum confidence to accept a match

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_url = "https://www.wikidata.org/w/api.php"
        self.headers = {
            'User-Agent': 'HereNews/1.0 (https://here.news; contact@here.news) aiohttp/3.9'
        }
        # OpenAI client for embeddings
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_cache: Dict[str, List[float]] = {}

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
            # Search Wikidata
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

            # Filter by type hint from description
            filtered = self._filter_by_type(candidates, entity_type)

            if not filtered:
                logger.debug(f"No type-matched candidates for: {name} ({entity_type})")
                return None

            # Score candidates with Bayesian inference
            best = await self._score_and_select(filtered, name, context, entity_type)

            if best and best['confidence'] >= self.MIN_CONFIDENCE:
                logger.info(f"🔗 Wikidata match: {name} → {best['qid']} ({best['label']}) [conf={best['confidence']:.2f}]")
                return best

            best_conf = best['confidence'] if best else 0
            logger.debug(f"No confident match for {name} (best conf: {best_conf:.2f})")
            return None

        except Exception as e:
            logger.warning(f"Wikidata search failed for {name}: {e}")
            return None

    async def _search_wikidata(self, name: str) -> List[Dict]:
        """Search Wikidata API for candidates."""
        params = {
            'action': 'wbsearchentities',
            'search': name,
            'language': 'en',
            'format': 'json',
            'limit': 10,
            'type': 'item'
        }

        try:
            async with self.session.get(self.api_url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json()

                candidates = []
                for item in data.get('search', []):
                    candidates.append({
                        'qid': item['id'],
                        'label': item.get('label', ''),
                        'description': item.get('description', ''),
                        'aliases': item.get('aliases', [])
                    })

                return candidates

        except asyncio.TimeoutError:
            logger.warning(f"Wikidata search timeout for: {name}")
            return []
        except Exception as e:
            logger.warning(f"Wikidata API error: {e}")
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

    def _filter_by_type(self, candidates: List[Dict], entity_type: str) -> List[Dict]:
        """Filter candidates by entity type using description keywords."""
        keywords = ENTITY_TYPE_KEYWORDS.get(entity_type, [])
        if not keywords:
            return candidates

        filtered = []
        for cand in candidates:
            desc_lower = cand['description'].lower()

            # Check if description contains type keywords
            for keyword in keywords:
                if keyword in desc_lower:
                    filtered.append(cand)
                    break
            else:
                # For locations, also accept if description mentions "Hong Kong" (local relevance)
                if entity_type == 'LOCATION' and 'hong kong' in desc_lower:
                    filtered.append(cand)

        return filtered if filtered else candidates[:3]  # Fall back to top 3 if no type match

    async def _score_and_select(
        self,
        candidates: List[Dict],
        name: str,
        context: str,
        entity_type: str
    ) -> Optional[Dict]:
        """
        Score candidates using structural signals + context.

        Structural scoring (from original wikidata_worker.py):
        1. Label match (exact match is strongest signal)
        2. Search rank (Wikidata's relevance ordering)
        3. Sitelinks count (notability - # of Wikipedia versions)
        4. Claims count (data richness - # of statements)

        Context scoring:
        5. Embedding similarity between context and description

        Returns best candidate with confidence score.
        """
        if not candidates:
            return None

        # Fetch full entity data for structural scoring
        qids = [c['qid'] for c in candidates]
        entity_data = await self._fetch_entity_data(qids)

        name_lower = name.lower()
        scored = []
        total_candidates = len(candidates)

        for idx, cand in enumerate(candidates):
            qid = cand['qid']
            ed = entity_data.get(qid, {})

            # === Structural Score (like original wikidata_worker) ===
            # Max structural score ~55 points

            # 1. Label match (0-20 points) - exact match is critical
            label_score = fuzz.ratio(name_lower, cand['label'].lower()) / 100
            if cand['label'].lower() == name_lower:
                label_match_score = 20.0  # Exact match bonus
            else:
                label_match_score = label_score * 15.0

            # 2. Search rank (0-10 points)
            rank_score = ((total_candidates - idx) / total_candidates) * 10.0

            # 3. Sitelinks count (0-15 points) - notability signal
            sitelinks_count = ed.get('sitelinks_count', 0)
            sitelinks_score = min(sitelinks_count / 50.0, 1.0) * 15.0

            # 4. Claims count (0-10 points) - data richness
            claims_count = ed.get('claims_count', 0)
            claims_score = min(claims_count / 100.0, 1.0) * 10.0

            structural_score = label_match_score + rank_score + sitelinks_score + claims_score

            # === Context Score (0-30 points) ===
            context_score = 15.0  # Neutral default (half of max)
            if context and cand['description']:
                # Use embeddings for semantic similarity (0.0-1.0 -> 0-30 points)
                embedding_sim = await self._embedding_similarity(context, cand['description'])
                context_score = embedding_sim * 30.0

            # Combined: structural (70%) + context (30%)
            total_score = structural_score * 0.7 + context_score * 0.3
            # Normalize to 0-1 range (max possible ~55*0.7 + 30*0.3 = 47.5)
            normalized_score = total_score / 47.5

            scored.append({
                **cand,
                'structural_score': structural_score,
                'context_score': context_score,
                'total_score': total_score,
                'sitelinks_count': sitelinks_count,
                'claims_count': claims_count,
                'idx': idx
            })

        # === Confidence Calculation (matching original wikidata_worker.py) ===
        # Sort by total_score (structural 70% + context 30%)
        scored.sort(key=lambda x: x['total_score'], reverse=True)

        best = scored[0]

        # Calculate ambiguity score (how close is second-best?)
        ambiguity_score = 0.0
        if len(scored) > 1:
            second_best = scored[1]
            if best['total_score'] > 0:
                # Higher ambiguity when scores are close
                ambiguity_score = 1.0 - (best['total_score'] - second_best['total_score']) / best['total_score']
                ambiguity_score = max(0.0, min(1.0, ambiguity_score))

        # Confidence: normalized score with ambiguity penalty
        # Max possible ~55*0.7 + 30*0.3 = 47.5
        raw_confidence = min(1.0, best['total_score'] / 47.5)
        best['confidence'] = raw_confidence * (1.0 - ambiguity_score * 0.3)

        # Log disambiguation details for debugging
        if len(scored) > 1 and context:
            logger.debug(
                f"Disambiguation for '{name}': "
                f"best={best['label']} (conf={best['confidence']:.2f}, total={best['total_score']:.1f}, "
                f"struct={best['structural_score']:.1f}, ctx={best['context_score']:.1f}, "
                f"sitelinks={best['sitelinks_count']}), "
                f"runner-up={scored[1]['label']} (total={scored[1]['total_score']:.1f})"
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
