"""
ClaimTopologyService - Bayesian claim plausibility analysis (Jaynes-Informed)

Implements probability as degree of plausibility, not frequency counting.
Key principles from E.T. Jaynes' Probability Theory:
1. Source reliability priors (stored on publisher at extraction time)
2. Temporal update model (later updates in disasters are expected)
3. Bayesian propagation over claim network
4. Date consensus (dates are factual claims too)
5. Maximum entropy when evidence is ambiguous

Source priors are assigned during knowledge extraction (knowledge_worker)
and stored on Publisher entities. This service reads stored priors rather
than re-computing them from URLs.
"""
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from openai import AsyncOpenAI

from models.domain.claim import Claim
from services.source_classification import (
    classify_source_by_domain, compute_base_prior, SOURCE_PRIORS
)

logger = logging.getLogger(__name__)


@dataclass
class PlausibilityResult:
    """Result of plausibility analysis for a single claim"""
    claim_id: str
    prior: float
    posterior: float
    evidence_for: List[str]
    evidence_against: List[str]
    confidence: float


@dataclass
class TopologyResult:
    """Full topology analysis result"""
    claim_plausibilities: Dict[str, PlausibilityResult]
    consensus_date: Optional[str]
    contradictions: List[Dict]
    pattern: str  # 'consensus', 'progressive', 'contradictory', 'mixed'
    # Update chains: maps superseded_claim_id -> superseding_claim_id
    # Claims in this dict as keys are outdated and should be treated as "earlier reports"
    superseded_by: Dict[str, str] = field(default_factory=dict)


class ClaimTopologyService:
    """
    Bayesian claim topology analysis for LiveEvent.

    Analyzes claim network to compute plausibility scores using:
    - Source priors
    - Corroboration/contradiction detection
    - Temporal progression modeling
    - Date consensus
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client
        self.llm_calls = 0

    def classify_source(self, url: str) -> Tuple[str, bool]:
        """
        Classify source type from URL (fallback only).

        Prefer using stored publisher priors from get_publisher_priors_for_claims().
        This is only used as fallback when publisher data is unavailable.
        """
        if not url:
            return ('unknown', False)

        # Extract domain from URL for classification
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc
        except (ValueError, AttributeError):
            domain = url

        return classify_source_by_domain(domain)

    def compute_source_prior(self, source_type: str, has_byline: bool) -> float:
        """
        Compute prior probability (fallback only).

        Prefer using stored base_prior from publisher entity.
        """
        return compute_base_prior(source_type, has_byline)

    def extract_numbers(self, text: str) -> Dict[str, int]:
        """Extract quantitative claims (deaths, injuries, etc.)"""
        numbers = {}
        text_lower = text.lower()

        # Death patterns
        death_patterns = [
            r'killed\s+(?:at\s+least\s+)?(\d+)',
            r'(\d+)\s+(?:people\s+)?(?:were\s+)?killed',
            r'(\d+)\s+(?:people\s+)?(?:dead|died|deaths?|fatalities)',
            r'death\s+toll\s+(?:rose\s+to\s+|reached\s+|of\s+)?(\d+)',
            r'(\d+)\s+(?:people\s+)?(?:lost\s+their\s+lives|perished)',
            r'at\s+least\s+(\d+)\s+(?:people\s+)?(?:dead|killed|died)',
        ]
        for pattern in death_patterns:
            match = re.search(pattern, text_lower)
            if match:
                numbers['deaths'] = int(match.group(1))
                break

        # Injury patterns
        injury_patterns = [
            r'(\d+)\s+(?:people\s+)?(?:were\s+)?injured',
            r'(\d+)\s+(?:people\s+)?(?:wounded|hurt)',
        ]
        for pattern in injury_patterns:
            match = re.search(pattern, text_lower)
            if match:
                numbers['injured'] = int(match.group(1))
                break

        # Missing patterns
        missing_patterns = [
            r'(\d+)\s+(?:people\s+)?(?:missing|unaccounted)',
        ]
        for pattern in missing_patterns:
            match = re.search(pattern, text_lower)
            if match:
                numbers['missing'] = int(match.group(1))
                break

        return numbers

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not texts:
            return []

        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [d.embedding for d in response.data]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def classify_relations(
        self,
        claim_pairs: List[Tuple[Claim, Claim, float]],  # (claim1, claim2, similarity)
        claim_numbers: Dict[str, Dict[str, int]],  # claim_id -> extracted numbers
        claim_times: Dict[str, Optional[datetime]]  # claim_id -> event_time
    ) -> Dict[Tuple[str, str], str]:
        """
        Classify relations between claim pairs.

        Uses domain knowledge for numeric claims, LLM for semantic.
        Returns: {(id1, id2): relation_type}
        """
        relations = {}
        pairs_for_llm = []

        for c1, c2, similarity in claim_pairs:
            key = tuple(sorted([c1.id, c2.id]))
            nums1 = claim_numbers.get(c1.id, {})
            nums2 = claim_numbers.get(c2.id, {})

            # Pre-classify numeric pairs using domain knowledge
            if 'deaths' in nums1 and 'deaths' in nums2:
                d1, d2 = nums1['deaths'], nums2['deaths']
                t1, t2 = claim_times.get(c1.id), claim_times.get(c2.id)

                if d1 == d2:
                    relations[key] = 'corroborates'
                elif t1 and t2:
                    if t2 > t1 and d2 > d1:
                        relations[key] = 'updates'
                    elif t1 > t2 and d1 > d2:
                        relations[key] = 'updates'
                    else:
                        relations[key] = 'contradicts'
                else:
                    if max(d1, d2) / max(min(d1, d2), 1) > 3:
                        relations[key] = 'updates'
                    else:
                        relations[key] = 'contradicts'
            elif similarity > 0.85:
                relations[key] = 'corroborates'
            elif similarity > 0.6:
                pairs_for_llm.append((c1, c2, similarity))
            else:
                relations[key] = 'complements'

        # Batch classify remaining pairs with LLM
        if pairs_for_llm:
            llm_relations = await self._batch_classify_with_llm(pairs_for_llm, claim_times)
            relations.update(llm_relations)

        return relations

    async def _batch_classify_with_llm(
        self,
        pairs: List[Tuple[Claim, Claim, float]],
        claim_times: Dict[str, Optional[datetime]]
    ) -> Dict[Tuple[str, str], str]:
        """Use LLM to classify semantic relations."""
        if not pairs:
            return {}

        self.llm_calls += 1

        pairs_text = []
        for i, (c1, c2, _) in enumerate(pairs[:20]):
            t1 = claim_times.get(c1.id)
            t2 = claim_times.get(c2.id)
            t1_str = t1.strftime('%Y-%m-%d %H:%M') if t1 else 'unknown'
            t2_str = t2.strftime('%Y-%m-%d %H:%M') if t2 else 'unknown'
            pairs_text.append(f"""
Pair {i+1}:
  A [{t1_str}]: {c1.text[:150]}
  B [{t2_str}]: {c2.text[:150]}
""")

        prompt = f"""Classify the relationship between each claim pair.

RELATIONSHIP TYPES:
- corroborates: Same fact, confirms each other
- contradicts: Conflicting facts about THE SAME subject that cannot both be true
- updates: Later claim updates/revises earlier (common for casualty counts in disasters)
- complements: Related but different aspects, OR claims about DIFFERENT subjects/events

IMPORTANT: Claims about DIFFERENT subjects are NOT contradictions. For example:
- "Mission X launched in August" and "Mission Y launches in December" → complements (different missions)
- "5 people died" and "3 people died" at the same time → contradicts (same event, conflicting numbers)

For "updates": Also specify which claim is newer (A or B).
- If numbers INCREASE over time (e.g., "5 dead" → "36 dead"), this is an UPDATE where the higher number is newer.
- In disasters, casualty counts typically increase as more victims are found.

For "contradicts": Include a brief "note" (max 100 chars) explaining the specific conflict.
- BAD note: "A says X, B says Y" (too vague)
- GOOD note: "Death toll differs: 5 vs 3 reported at same time"

{chr(10).join(pairs_text)}

Return JSON: {{"results": [{{"pair": 1, "relation": "updates", "newer": "B"}}]}}
For contradicts: {{"pair": 2, "relation": "contradicts", "note": "Death toll differs: 5 vs 3 reported simultaneously"}}
For non-update/non-contradicts relations, omit "newer" and "note"."""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            import json
            data = json.loads(response.choices[0].message.content)
            items = data.get('results', data.get('pairs', []))

            results = {}
            for item in items:
                idx = item.get('pair', 0) - 1
                if 0 <= idx < len(pairs):
                    c1, c2, _ = pairs[idx]
                    relation = item.get('relation', 'complements')
                    newer = item.get('newer')  # 'A' or 'B' for updates
                    note = item.get('note')  # Explanation for contradicts

                    # For updates, store as tuple: (relation, newer_claim_id, older_claim_id)
                    if relation == 'updates' and newer:
                        if newer == 'A':
                            results[(c1.id, c2.id)] = ('updates', c1.id, c2.id)  # c1 is newer
                        else:
                            results[(c1.id, c2.id)] = ('updates', c2.id, c1.id)  # c2 is newer
                    # For contradicts, store as tuple with note: (relation, note)
                    elif relation == 'contradicts':
                        key = tuple(sorted([c1.id, c2.id]))
                        results[key] = ('contradicts', note) if note else 'contradicts'
                    else:
                        key = tuple(sorted([c1.id, c2.id]))
                        results[key] = relation

            return results

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return {}

    def compute_posterior(
        self,
        claim: Claim,
        prior: float,
        neighbors: List[Tuple[str, float, str]],  # [(neighbor_id, similarity, relation)]
        neighbor_priors: Dict[str, float]
    ) -> PlausibilityResult:
        """
        Compute posterior probability using Bayes' theorem.

        P(claim accurate | evidence) ∝ P(evidence | claim accurate) × P(claim accurate)
        """
        evidence_for = []
        evidence_against = []
        log_likelihood = 0.0

        for neighbor_id, similarity, relation in neighbors:
            neighbor_prior = neighbor_priors.get(neighbor_id, 0.5)

            if relation == 'corroborates':
                factor = 1.2 + 0.3 * similarity * neighbor_prior
                log_likelihood += math.log(factor)
                evidence_for.append(neighbor_id)

            elif relation == 'contradicts':
                relative_strength = neighbor_prior / max(prior, 0.1)
                factor = 0.7 - 0.2 * similarity * min(relative_strength, 1.5)
                factor = max(0.3, factor)
                log_likelihood += math.log(factor)
                evidence_against.append(neighbor_id)

            elif relation == 'updates':
                # Updates: handled specially - newer claims boost, older penalized
                # This branch is for claims that have an update relation but direction unknown
                factor = 1.0  # Neutral - direction handled in analyze()
                log_likelihood += math.log(factor)

            elif relation == 'complements':
                factor = 1.0 + 0.1 * similarity
                log_likelihood += math.log(factor)

        likelihood = math.exp(log_likelihood) if log_likelihood != 0 else 1.0
        posterior = prior * likelihood

        n_evidence = len(evidence_for) + len(evidence_against)
        confidence = min(0.95, 0.3 + 0.1 * n_evidence)

        return PlausibilityResult(
            claim_id=claim.id,
            prior=prior,
            posterior=posterior,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            confidence=confidence
        )

    async def analyze(
        self,
        claims: List[Claim],
        publisher_priors: Dict[str, dict] = None,  # claim_id -> {'base_prior': float, 'source_type': str}
        page_urls: Dict[str, str] = None  # claim_id -> source URL (fallback)
    ) -> TopologyResult:
        """
        Full Bayesian analysis of claim network.

        Args:
            claims: Claims to analyze
            publisher_priors: Stored publisher priors from ClaimRepository.get_publisher_priors_for_claims()
                             Each entry: {'base_prior': float, 'source_type': str, 'publisher_name': str}
            page_urls: Fallback - source URLs for claims without stored priors

        Returns:
            TopologyResult with plausibilities, consensus date, contradictions
        """
        if not claims:
            return TopologyResult(
                claim_plausibilities={},
                consensus_date=None,
                contradictions=[],
                pattern='empty'
            )

        publisher_priors = publisher_priors or {}
        page_urls = page_urls or {}

        logger.info(f"🧮 Bayesian topology analysis of {len(claims)} claims")

        # Get source priors - prefer stored publisher priors, fallback to URL classification
        claim_priors = {}
        for claim in claims:
            if claim.id in publisher_priors and publisher_priors[claim.id].get('base_prior'):
                # Use stored prior from publisher entity
                claim_priors[claim.id] = publisher_priors[claim.id]['base_prior']
                logger.debug(f"  Using stored prior for {claim.id}: {claim_priors[claim.id]}")
            elif claim.id in page_urls:
                # Fallback: classify from URL
                url = page_urls[claim.id]
                source_type, has_byline = self.classify_source(url)
                claim_priors[claim.id] = self.compute_source_prior(source_type, has_byline)
                logger.debug(f"  Fallback URL prior for {claim.id}: {claim_priors[claim.id]}")
            else:
                # Maximum entropy default
                claim_priors[claim.id] = 0.50

        # Extract numbers
        claim_numbers = {c.id: self.extract_numbers(c.text) for c in claims}

        # Get event times
        claim_times = {}
        for claim in claims:
            if claim.event_time:
                if isinstance(claim.event_time, str):
                    try:
                        claim_times[claim.id] = datetime.fromisoformat(
                            claim.event_time.replace('Z', '+00:00')
                        )
                    except (ValueError, TypeError, AttributeError):
                        pass
                else:
                    claim_times[claim.id] = claim.event_time

        # Generate embeddings for claims without them
        claims_needing_embeddings = [c for c in claims if not c.embedding]
        if claims_needing_embeddings:
            logger.info(f"  📊 Generating {len(claims_needing_embeddings)} embeddings...")
            texts = [c.text for c in claims_needing_embeddings]
            embeddings = await self.generate_embeddings(texts)
            for claim, emb in zip(claims_needing_embeddings, embeddings):
                claim.embedding = emb

        # Build similarity network
        logger.info(f"  🕸️ Building similarity network...")
        network: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        high_sim_pairs = []

        for i, c1 in enumerate(claims):
            for c2 in claims[i+1:]:
                if c1.embedding and c2.embedding:
                    sim = self.cosine_similarity(c1.embedding, c2.embedding)
                    if sim > 0.4:
                        network[c1.id].append((c2.id, sim))
                        network[c2.id].append((c1.id, sim))
                        if sim > 0.6:
                            high_sim_pairs.append((c1, c2, sim))

        # Classify relations
        logger.info(f"  🔗 Classifying {len(high_sim_pairs)} high-similarity pairs...")
        relations = await self.classify_relations(high_sim_pairs, claim_numbers, claim_times)

        # Extract update chains (superseded_id -> superseding_id)
        superseded_by: Dict[str, str] = {}
        for key, val in relations.items():
            if isinstance(val, tuple) and val[0] == 'updates':
                _, newer_id, older_id = val
                superseded_by[older_id] = newer_id
                logger.debug(f"  📝 Update chain: {older_id[:12]} superseded by {newer_id[:12]}")

        # Compute posteriors
        logger.info(f"  📈 Computing posteriors...")
        results = {}
        for claim in claims:
            neighbors = []
            for neighbor_id, similarity in network.get(claim.id, []):
                # Handle both string relations and tuple (for updates)
                key = tuple(sorted([claim.id, neighbor_id]))
                rel = relations.get(key)
                if rel is None:
                    # Try ordered key for updates
                    rel = relations.get((claim.id, neighbor_id)) or relations.get((neighbor_id, claim.id))

                if isinstance(rel, tuple):
                    relation = rel[0]  # 'updates'
                else:
                    relation = rel or 'complements'

                neighbors.append((neighbor_id, similarity, relation))

            result = self.compute_posterior(
                claim,
                claim_priors[claim.id],
                neighbors,
                claim_priors
            )
            results[claim.id] = result

        # Compute date consensus
        date_counts = defaultdict(int)
        for claim in claims:
            if claim.id in claim_times:
                date_key = claim_times[claim.id].strftime('%Y-%m-%d')
                date_counts[date_key] += 1

        consensus_date = None
        if date_counts:
            consensus_date = max(date_counts.keys(), key=lambda d: date_counts[d])
            consensus_count = date_counts[consensus_date]
            logger.info(f"  📅 Date consensus: {consensus_date} ({consensus_count}/{sum(date_counts.values())} claims)")

            # Penalize date outliers
            for claim in claims:
                if claim.id in claim_times:
                    claim_date = claim_times[claim.id].strftime('%Y-%m-%d')
                    if claim_date != consensus_date:
                        claim_year = claim_times[claim.id].year
                        consensus_year = int(consensus_date[:4])
                        if claim_year != consensus_year:
                            results[claim.id].posterior *= 0.3
                            logger.debug(f"  ⚠️ Date outlier penalty: {claim.text[:40]}...")

        # Normalize posteriors
        posteriors = [r.posterior for r in results.values()]
        if posteriors:
            log_posts = [math.log(p + 1e-10) for p in posteriors]
            mean_log = np.mean(log_posts)
            std_log = np.std(log_posts) if len(log_posts) > 1 else 1.0

            for result in results.values():
                z = (math.log(result.posterior + 1e-10) - mean_log) / max(std_log, 0.1)
                result.posterior = 1 / (1 + math.exp(-z * 0.8))
                result.posterior = min(0.95, max(0.10, result.posterior))

        # Apply superseded penalty: claims that have been updated get lower plausibility
        # This ensures newer figures are preferred over older ones
        for old_id, new_id in superseded_by.items():
            if old_id in results and new_id in results:
                old_result = results[old_id]
                new_result = results[new_id]
                # Only penalize if newer claim has decent plausibility
                if new_result.posterior > 0.5:
                    # Reduce old claim's plausibility significantly
                    old_result.posterior = min(old_result.posterior, 0.45)
                    # Boost newer claim slightly
                    new_result.posterior = min(0.95, new_result.posterior * 1.1)
                    logger.debug(f"  ⬇️ Superseded penalty: {old_id[:12]} -> {old_result.posterior:.2f}")

        # Find contradictions
        contradictions = []
        seen = set()
        for cid, result in results.items():
            for contra_id in result.evidence_against:
                pair = tuple(sorted([cid, contra_id]))
                if pair not in seen:
                    seen.add(pair)
                    key = tuple(sorted([cid, contra_id]))
                    rel = relations.get(key)
                    # Check for contradicts (either string or tuple with note)
                    is_contradiction = rel == 'contradicts' or (isinstance(rel, tuple) and rel[0] == 'contradicts')
                    if is_contradiction:
                        c1 = next(c for c in claims if c.id == cid)
                        c2 = next(c for c in claims if c.id == contra_id)
                        # Extract note if present
                        note = rel[1] if isinstance(rel, tuple) and len(rel) > 1 else None
                        contradictions.append({
                            'claim1_id': cid,
                            'claim2_id': contra_id,
                            'text1': c1.text[:100],
                            'text2': c2.text[:100],
                            'posterior1': results[cid].posterior,
                            'posterior2': results[contra_id].posterior,
                            'note': note,
                        })

        # Determine pattern
        # Count updates (now stored as tuples)
        n_updates = sum(1 for rel in relations.values()
                       if (isinstance(rel, tuple) and rel[0] == 'updates') or rel == 'updates')
        n_contradicts = len(contradictions)
        n_corroborates = sum(1 for rel in relations.values() if rel == 'corroborates')

        if n_updates > n_contradicts and n_updates > 3:
            pattern = 'progressive'
        elif n_contradicts > n_corroborates * 0.3:
            pattern = 'contradictory'
        elif n_corroborates > len(claims) * 0.3:
            pattern = 'consensus'
        else:
            pattern = 'mixed'

        logger.info(f"  📊 Pattern: {pattern}, updates: {n_updates}, superseded: {len(superseded_by)}, LLM calls: {self.llm_calls}")

        return TopologyResult(
            claim_plausibilities=results,
            consensus_date=consensus_date,
            contradictions=contradictions,
            pattern=pattern,
            superseded_by=superseded_by
        )

    def enrich_claims_with_plausibility(
        self,
        claims: List[Claim],
        topology: TopologyResult
    ) -> List[dict]:
        """
        Enrich claims with plausibility data for narrative generation.

        This prepares claim data to be passed to EventService._generate_event_narrative
        so we don't duplicate narrative generation logic.

        Returns list of dicts compatible with EventService's enriched claims format,
        with plausibility data added.
        """
        contested_ids = set()
        for contra in topology.contradictions:
            diff = abs(contra['posterior1'] - contra['posterior2'])
            if diff < 0.15:
                contested_ids.add(contra['claim1_id'])
                contested_ids.add(contra['claim2_id'])

        enriched = []
        for claim in claims:
            result = topology.claim_plausibilities.get(claim.id)
            plausibility = result.posterior if result else 0.5

            # Build entity info
            entities = []
            for i, eid in enumerate(claim.entity_ids):
                name = claim.entity_names[i] if i < len(claim.entity_names) else None
                entities.append({'id': eid, 'name': name})

            enriched.append({
                'id': claim.id,
                'text': claim.text,
                'confidence': claim.confidence,
                'plausibility': plausibility,
                'is_contested': claim.id in contested_ids,
                'corroboration_count': len(result.evidence_for) if result else 0,
                'has_time': claim.event_time is not None,
                'event_time': claim.event_time,
                'entities': entities,
                'claim': claim
            })

        # Sort by plausibility (highest first), then by time
        enriched.sort(key=lambda c: (-c['plausibility'], c['event_time'] or ''))

        return enriched

    def get_topology_context(self, topology: TopologyResult) -> dict:
        """
        Extract topology context for narrative generation.

        Returns context dict that can be used to augment narrative prompts.
        """
        return {
            'consensus_date': topology.consensus_date,
            'pattern': topology.pattern,
            'contradictions': [
                {
                    'claim1_id': c.get('claim1_id') or c.get('source_id', ''),
                    'claim2_id': c.get('claim2_id') or c.get('target_id', ''),
                    'text1': c.get('text1', ''),  # May be empty if loaded from storage
                    'text2': c.get('text2', '')   # May be empty if loaded from storage
                }
                for c in topology.contradictions
            ],
            'superseded_by': topology.superseded_by  # older_id -> newer_id
        }

    async def incremental_update(
        self,
        new_claims: List[Claim],
        existing_topology: TopologyResult,
        existing_claims: List[Claim],
        publisher_priors: Dict[str, dict] = None,
        page_urls: Dict[str, str] = None
    ) -> Tuple[TopologyResult, List[dict]]:
        """
        Incrementally update topology with new claims.

        Instead of recomputing everything:
        1. Generate embeddings only for new claims
        2. Compare new claims against existing claims
        3. Classify relationships only for new pairs
        4. Update posteriors locally
        5. Return updated topology + list of new relationships to persist

        Args:
            new_claims: New claims to add
            existing_topology: Current topology result from hydration
            existing_claims: Existing claims (already have embeddings)
            publisher_priors: Priors for new claims
            page_urls: Fallback URLs for new claims

        Returns:
            (updated_topology, new_relationships_to_persist)
        """
        if not new_claims:
            return existing_topology, []

        logger.info(f"🔄 Incremental topology update: {len(new_claims)} new claims vs {len(existing_claims)} existing")

        publisher_priors = publisher_priors or {}
        page_urls = page_urls or {}

        # Get priors for new claims
        new_priors = {}
        for claim in new_claims:
            if claim.id in publisher_priors and publisher_priors[claim.id].get('base_prior'):
                new_priors[claim.id] = publisher_priors[claim.id]['base_prior']
            elif claim.id in page_urls:
                url = page_urls[claim.id]
                source_type, has_byline = self.classify_source(url)
                new_priors[claim.id] = self.compute_source_prior(source_type, has_byline)
            else:
                new_priors[claim.id] = 0.50

        # Generate embeddings for new claims only
        new_embeddings = await self.generate_embeddings([c.text for c in new_claims])

        # Get embeddings for existing claims (they should already have them)
        existing_embeddings = {}
        for claim in existing_claims:
            if hasattr(claim, 'embedding') and claim.embedding:
                existing_embeddings[claim.id] = claim.embedding

        # If existing claims don't have embeddings cached, we need them
        if not existing_embeddings and existing_claims:
            logger.info(f"  📊 Generating {len(existing_claims)} embeddings for existing claims...")
            existing_emb_list = await self.generate_embeddings([c.text for c in existing_claims])
            for i, claim in enumerate(existing_claims):
                existing_embeddings[claim.id] = existing_emb_list[i]

        # Find similar pairs: new claims vs existing claims
        similar_pairs = []
        new_emb_map = {new_claims[i].id: new_embeddings[i] for i in range(len(new_claims))}

        for new_claim in new_claims:
            new_emb = new_emb_map[new_claim.id]

            # Compare against existing claims
            for existing_claim in existing_claims:
                if existing_claim.id not in existing_embeddings:
                    continue
                existing_emb = existing_embeddings[existing_claim.id]

                similarity = np.dot(new_emb, existing_emb) / (
                    np.linalg.norm(new_emb) * np.linalg.norm(existing_emb) + 1e-10
                )

                if similarity > 0.4:  # Same threshold as full analysis
                    similar_pairs.append((new_claim, existing_claim, similarity))

            # Also compare new claims against each other
            for other_new in new_claims:
                if other_new.id >= new_claim.id:  # Avoid duplicates
                    continue
                other_emb = new_emb_map[other_new.id]
                similarity = np.dot(new_emb, other_emb) / (
                    np.linalg.norm(new_emb) * np.linalg.norm(other_emb) + 1e-10
                )
                if similarity > 0.4:
                    similar_pairs.append((new_claim, other_new, similarity))

        logger.info(f"  🔗 Found {len(similar_pairs)} similar pairs to classify")

        # Classify relationships for similar pairs
        new_relationships = []

        if similar_pairs:
            # Use domain knowledge for numeric updates
            new_numbers = {c.id: self.extract_numbers(c.text) for c in new_claims}
            new_times = {}
            for c in new_claims:
                if c.event_time:
                    new_times[c.id] = c.event_time

            existing_numbers = {c.id: self.extract_numbers(c.text) for c in existing_claims}
            existing_times = {}
            for c in existing_claims:
                if c.event_time:
                    existing_times[c.id] = c.event_time

            all_numbers = {**existing_numbers, **new_numbers}
            all_times = {**existing_times, **new_times}

            # Try domain classification first
            pairs_needing_llm = []
            for c1, c2, sim in similar_pairs:
                relation = self.classify_relation_domain(
                    c1.id, c2.id, all_numbers, all_times
                )
                if relation:
                    new_relationships.append({
                        'source': c1.id,
                        'target': c2.id,
                        'type': relation if isinstance(relation, str) else relation[0],
                        'similarity': sim
                    })
                else:
                    pairs_needing_llm.append((c1, c2, sim))

            # LLM classification for remaining pairs
            if pairs_needing_llm:
                logger.info(f"  🤖 LLM classifying {len(pairs_needing_llm)} pairs...")
                llm_relations = await self._batch_classify_relations(pairs_needing_llm)
                for (c1, c2, sim), rel_type in zip(pairs_needing_llm, llm_relations):
                    if rel_type and rel_type != 'complements':
                        new_relationships.append({
                            'source': c1.id,
                            'target': c2.id,
                            'type': rel_type,
                            'similarity': sim
                        })

        # Update the topology result with new claims
        updated_plausibilities = dict(existing_topology.claim_plausibilities)
        updated_contradictions = list(existing_topology.contradictions)
        updated_superseded = dict(existing_topology.superseded_by)

        # Compute posteriors for new claims
        for new_claim in new_claims:
            # Find evidence from relationships
            evidence_for = []
            evidence_against = []

            for rel in new_relationships:
                if rel['source'] == new_claim.id:
                    other_id = rel['target']
                elif rel['target'] == new_claim.id:
                    other_id = rel['source']
                else:
                    continue

                if rel['type'] == 'corroborates':
                    evidence_for.append(other_id)
                elif rel['type'] == 'contradicts':
                    evidence_against.append(other_id)
                    # Add to contradictions list
                    updated_contradictions.append({
                        'claim1_id': new_claim.id,
                        'claim2_id': other_id,
                        'text1': new_claim.text[:100],
                        'text2': next((c.text[:100] for c in existing_claims if c.id == other_id), ''),
                    })
                elif rel['type'] == 'updates':
                    # New claim updates existing - mark existing as superseded
                    updated_superseded[other_id] = new_claim.id

            # Compute posterior for new claim
            prior = new_priors.get(new_claim.id, 0.5)
            posterior = prior

            # Boost from corroborations
            for corr_id in evidence_for:
                if corr_id in updated_plausibilities:
                    corr_prior = updated_plausibilities[corr_id].prior
                    posterior = posterior * (1.2 + 0.3 * corr_prior)

            # Penalty from contradictions
            for contra_id in evidence_against:
                if contra_id in updated_plausibilities:
                    contra_post = updated_plausibilities[contra_id].posterior
                    posterior = posterior * (0.7 - 0.2 * contra_post)

            posterior = min(0.95, max(0.10, posterior))

            updated_plausibilities[new_claim.id] = PlausibilityResult(
                claim_id=new_claim.id,
                prior=prior,
                posterior=posterior,
                evidence_for=evidence_for,
                evidence_against=evidence_against,
                confidence=posterior
            )

        # Also update existing claims affected by new evidence
        for rel in new_relationships:
            existing_id = None
            if rel['source'] in updated_plausibilities and rel['target'] not in [c.id for c in new_claims]:
                existing_id = rel['target']
            elif rel['target'] in updated_plausibilities and rel['source'] not in [c.id for c in new_claims]:
                existing_id = rel['source']

            if existing_id and existing_id in updated_plausibilities:
                existing_result = updated_plausibilities[existing_id]

                if rel['type'] == 'corroborates':
                    # Boost existing claim
                    existing_result.evidence_for.append(
                        rel['source'] if rel['target'] == existing_id else rel['target']
                    )
                    existing_result.posterior = min(0.95, existing_result.posterior * 1.1)

                elif rel['type'] == 'contradicts':
                    # Penalize existing claim slightly
                    existing_result.evidence_against.append(
                        rel['source'] if rel['target'] == existing_id else rel['target']
                    )
                    existing_result.posterior = max(0.10, existing_result.posterior * 0.95)

        # Recompute pattern
        n_corroborates = sum(1 for r in new_relationships if r['type'] == 'corroborates')
        n_contradicts = len([c for c in updated_contradictions])
        n_updates = len(updated_superseded)

        if n_updates > n_contradicts and n_updates > 3:
            pattern = 'progressive'
        elif n_contradicts > n_corroborates * 0.3:
            pattern = 'contradictory'
        elif n_corroborates > len(new_claims) * 0.3:
            pattern = 'consensus'
        else:
            pattern = existing_topology.pattern  # Keep existing pattern

        updated_topology = TopologyResult(
            claim_plausibilities=updated_plausibilities,
            consensus_date=existing_topology.consensus_date,
            contradictions=updated_contradictions,
            pattern=pattern,
            superseded_by=updated_superseded
        )

        logger.info(f"  ✅ Incremental update complete: {len(new_relationships)} new relationships, "
                   f"{len(new_claims)} claims added")

        return updated_topology, new_relationships
