"""
WikidataClient - Lightweight Wikidata search for entity identification.

Extracted from wikidata_worker for use during entity identification.
This enables QID resolution DURING the knowledge pipeline (not async after).

Usage:
    client = WikidataClient()
    result = await client.search_entity(
        name="Wang Fuk Court",
        entity_type="LOCATION",
        context="fire at Wang Fuk Court in Tai Po district",
        aliases=[]
    )
    # Returns: {'qid': 'Q20983881', 'label': 'Wang Fuk Court', 'confidence': 0.9}
"""
import aiohttp
import asyncio
import logging
import re
from typing import Optional, Dict, List, Any
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Entity Type to Wikidata type keywords for filtering
ENTITY_TYPE_KEYWORDS = {
    'PERSON': ['human', 'person', 'politician', 'actor', 'singer', 'athlete'],
    'ORGANIZATION': ['organization', 'company', 'institution', 'agency', 'party', 'church', 'university'],
    'LOCATION': ['city', 'district', 'country', 'building', 'estate', 'court', 'place', 'region', 'settlement']
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
    Lightweight Wikidata search client for entity identification.

    Uses Wikidata API's wbsearchentities for fast label-based search.
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_url = "https://www.wikidata.org/w/api.php"
        # User-Agent required by Wikidata API policy
        self.headers = {
            'User-Agent': 'HereNews/1.0 (https://here.news; contact@here.news) aiohttp/3.9'
        }

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

            # Score and select best match
            best = self._select_best(filtered, name, context)

            if best and best['confidence'] >= 0.6:
                logger.info(f"🔗 Wikidata match: {name} → {best['qid']} ({best['label']})")
                return best

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

    def _select_best(self, candidates: List[Dict], name: str, context: str) -> Optional[Dict]:
        """Select best candidate using label similarity and context."""
        if not candidates:
            return None

        name_lower = name.lower()

        scored = []
        for cand in candidates:
            # Label similarity score
            label_score = fuzz.ratio(name_lower, cand['label'].lower()) / 100

            # Exact match bonus
            if cand['label'].lower() == name_lower:
                label_score = 1.0

            # Context match bonus (if description mentions context words)
            context_score = 0.0
            if context:
                context_words = set(context.lower().split())
                desc_words = set(cand['description'].lower().split())
                overlap = len(context_words & desc_words)
                if overlap > 0:
                    context_score = min(0.2, overlap * 0.05)

            # Position penalty (lower ranked = lower confidence)
            position_penalty = candidates.index(cand) * 0.05

            confidence = label_score + context_score - position_penalty
            confidence = max(0.0, min(1.0, confidence))

            scored.append({
                **cand,
                'confidence': round(confidence, 2)
            })

        # Sort by confidence
        scored.sort(key=lambda x: x['confidence'], reverse=True)

        return scored[0] if scored else None
