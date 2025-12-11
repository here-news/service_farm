"""
Bayesian Claim Topology Test (Jaynes-Informed)

Implements probability as degree of plausibility, not frequency counting.
Key principles:
1. Source reliability priors (not trust scores - our state of knowledge)
2. Temporal update model (later updates in disasters are expected)
3. Bayesian propagation over claim network
4. Maximum entropy when evidence is ambiguous
5. Explicit uncertainty quantification

Priors (conservative, scrutiny-aware):
- Wire service (AP, Reuters, AFP): 0.60 (+ 0.05 if byline)
- Official release: 0.55 (needs scrutiny - may be preliminary/political)
- Local news with byline: 0.60
- Local news without byline: 0.55
- Aggregator/unknown: 0.50 (maximum entropy - no information)
"""
import asyncio
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import json
from datetime import datetime
import math
import re

sys.path.insert(0, '/app')
from services.neo4j_service import Neo4jService
from openai import AsyncOpenAI
import os


@dataclass
class SourcePrior:
    """Source reliability prior - our state of knowledge about source accuracy"""
    base: float
    byline_bonus: float = 0.05
    reason: str = ""


# Conservative priors - these represent our UNCERTAINTY, not trust
SOURCE_PRIORS = {
    'wire': SourcePrior(0.60, 0.05, "Wire services have editorial standards but still make errors"),
    'official': SourcePrior(0.55, 0.0, "Official sources may be preliminary, political, or face-saving"),
    'local_news': SourcePrior(0.55, 0.05, "Local news varies; byline adds accountability"),
    'international': SourcePrior(0.58, 0.05, "International outlets have distance but resources"),
    'aggregator': SourcePrior(0.50, 0.0, "Maximum entropy - unknown reliability"),
    'unknown': SourcePrior(0.50, 0.0, "Maximum entropy - no information to update from"),
}

# Wire services we recognize
WIRE_SERVICES = {'ap', 'reuters', 'afp', 'associated press', 'xinhua', 'efe', 'ansa', 'dpa'}

# Official source indicators
OFFICIAL_INDICATORS = {'government', 'ministry', 'department', 'official', 'police', 'fire service',
                       'authority', 'bureau', 'commission'}


@dataclass
class ClaimWithMetadata:
    """Claim with all metadata needed for Bayesian analysis"""
    id: str
    text: str
    page_id: str
    page_url: str
    event_time: Optional[datetime]
    source_type: str
    has_byline: bool
    prior: float
    extracted_numbers: Dict[str, int] = field(default_factory=dict)


@dataclass
class BayesianResult:
    """Result of Bayesian analysis"""
    claim_id: str
    prior: float
    likelihood: float
    posterior: float
    evidence_for: List[str]  # claim_ids that support
    evidence_against: List[str]  # claim_ids that contradict
    temporal_factor: float  # adjustment for temporal ordering
    confidence: float  # how confident are we in this posterior


@dataclass
class TopologyAnalysis:
    """Full topology analysis result"""
    results: Dict[str, BayesianResult]
    contradictions: List[Dict]  # explicit contradiction groups
    consensus_claims: List[str]  # high-agreement claim ids
    entropy: float  # overall uncertainty in the system
    pattern: str  # 'consensus', 'progressive', 'contradictory', 'mixed'
    consensus_date: Optional[str] = None  # most agreed-upon date


class BayesianTopologyAnalyzer:
    """
    Jaynes-informed Bayesian claim analysis.

    Key insight: We're computing P(claim is accurate | all evidence, background knowledge)
    NOT counting votes.
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai = openai_client
        self.embeddings: Dict[str, List[float]] = {}
        self.claims: Dict[str, ClaimWithMetadata] = {}
        self.network: Dict[str, List[Tuple[str, float]]] = {}  # claim_id -> [(neighbor_id, similarity)]
        self.relation_cache: Dict[Tuple[str, str], str] = {}  # (id1, id2) -> relation type
        self.llm_calls = 0

    def classify_source(self, url: str) -> Tuple[str, bool]:
        """
        Classify source type and detect byline presence.
        Returns (source_type, has_byline_indicator)
        """
        url_lower = url.lower()

        # Wire services
        for wire in WIRE_SERVICES:
            if wire in url_lower:
                return ('wire', True)  # Wire services always have editorial oversight

        # Check for official sources
        for indicator in OFFICIAL_INDICATORS:
            if indicator in url_lower:
                return ('official', False)

        # Known international outlets
        international = ['bbc', 'cnn', 'guardian', 'nytimes', 'washingtonpost',
                        'aljazeera', 'dw.com', 'france24']
        for outlet in international:
            if outlet in url_lower:
                return ('international', True)

        # Local news (assume byline if .com/.org news site)
        if any(x in url_lower for x in ['news', 'times', 'post', 'herald', 'tribune']):
            return ('local_news', True)

        # Aggregators
        if any(x in url_lower for x in ['msn.com', 'yahoo.com', 'google.com/amp']):
            return ('aggregator', False)

        return ('unknown', False)

    def compute_source_prior(self, source_type: str, has_byline: bool) -> float:
        """
        Compute prior probability based on source type.
        This is P(source reports accurately | background knowledge)
        """
        prior_info = SOURCE_PRIORS.get(source_type, SOURCE_PRIORS['unknown'])
        prior = prior_info.base
        if has_byline:
            prior += prior_info.byline_bonus
        return prior

    def extract_numbers(self, text: str) -> Dict[str, int]:
        """Extract quantitative claims (deaths, injuries, etc.)"""
        numbers = {}
        text_lower = text.lower()

        # Death patterns - comprehensive
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
            r'injured\s+(\d+)',
        ]
        for pattern in injury_patterns:
            match = re.search(pattern, text_lower)
            if match:
                numbers['injured'] = int(match.group(1))
                break

        # Missing patterns
        missing_patterns = [
            r'(\d+)\s+(?:people\s+)?(?:missing|unaccounted)',
            r'(\d+)\s+(?:still\s+)?missing',
        ]
        for pattern in missing_patterns:
            match = re.search(pattern, text_lower)
            if match:
                numbers['missing'] = int(match.group(1))
                break

        # Evacuated/displaced
        evac_patterns = [
            r'(\d+)\s+(?:people\s+)?(?:evacuated|displaced|homeless)',
            r'(\d+)\s+(?:residents?\s+)?(?:sought\s+refuge|fled)',
        ]
        for pattern in evac_patterns:
            match = re.search(pattern, text_lower)
            if match:
                numbers['evacuated'] = int(match.group(1))
                break

        return numbers

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for semantic similarity"""
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity"""
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    async def classify_relation(self, claim1: ClaimWithMetadata, claim2: ClaimWithMetadata) -> str:
        """
        Use LLM to classify relationship between claims.
        Returns: 'corroborates', 'contradicts', 'updates', 'unrelated', 'complements'
        """
        cache_key = tuple(sorted([claim1.id, claim2.id]))
        if cache_key in self.relation_cache:
            return self.relation_cache[cache_key]

        # Check if we have numeric values that might indicate update
        nums1 = claim1.extracted_numbers
        nums2 = claim2.extracted_numbers

        # Quick heuristic for numeric updates
        for key in set(nums1.keys()) & set(nums2.keys()):
            if nums1[key] != nums2[key]:
                # Different numbers for same metric - likely update or contradiction
                # If temporal ordering suggests update, mark as such
                if claim1.event_time and claim2.event_time:
                    if claim2.event_time > claim1.event_time and nums2[key] > nums1[key]:
                        self.relation_cache[cache_key] = 'updates'
                        return 'updates'

        # For non-numeric or ambiguous cases, we'll batch these for LLM
        # For now, use similarity as proxy
        similarity = self.cosine_similarity(
            self.embeddings[claim1.id],
            self.embeddings[claim2.id]
        )

        if similarity > 0.85:
            relation = 'corroborates'
        elif similarity > 0.6:
            relation = 'complements'
        else:
            relation = 'unrelated'

        self.relation_cache[cache_key] = relation
        return relation

    async def batch_classify_relations(self, claim_pairs: List[Tuple[ClaimWithMetadata, ClaimWithMetadata]]) -> Dict[Tuple[str, str], str]:
        """
        Batch classify relations using LLM for efficiency.
        Only for high-similarity pairs that need semantic judgment.
        """
        # First pass: pre-classify pairs with numeric values using domain knowledge
        pairs_for_llm = []
        pre_classified = {}

        for c1, c2 in claim_pairs:
            key = tuple(sorted([c1.id, c2.id]))
            nums1, nums2 = c1.extracted_numbers, c2.extracted_numbers

            # Check for death toll comparisons
            if 'deaths' in nums1 and 'deaths' in nums2:
                d1, d2 = nums1['deaths'], nums2['deaths']
                if d1 == d2:
                    pre_classified[key] = 'corroborates'
                elif c1.event_time and c2.event_time:
                    # Temporal ordering
                    if c2.event_time > c1.event_time and d2 > d1:
                        pre_classified[key] = 'updates'  # Later, higher = update
                    elif c1.event_time > c2.event_time and d1 > d2:
                        pre_classified[key] = 'updates'
                    else:
                        # Numbers differ at same time or decrease - genuine conflict
                        pre_classified[key] = 'contradicts'
                else:
                    # No temporal info - treat larger jump as update, similar as contradiction
                    if max(d1, d2) / min(d1, d2) > 3:
                        pre_classified[key] = 'updates'  # Big jump likely update
                    else:
                        pre_classified[key] = 'contradicts'

                self.relation_cache[key] = pre_classified[key]
            else:
                pairs_for_llm.append((c1, c2))

        if pre_classified:
            print(f"      Pre-classified {len(pre_classified)} numeric pairs")

        if not pairs_for_llm:
            return pre_classified

        self.llm_calls += 1

        # Format remaining pairs for LLM
        pairs_text = []
        for i, (c1, c2) in enumerate(pairs_for_llm[:20]):  # Limit batch size
            t1 = c1.event_time.strftime('%Y-%m-%d %H:%M') if c1.event_time else 'unknown'
            t2 = c2.event_time.strftime('%Y-%m-%d %H:%M') if c2.event_time else 'unknown'
            pairs_text.append(f"""
Pair {i+1}:
  A [{t1}]: {c1.text}
  B [{t2}]: {c2.text}
""")

        prompt = f"""Classify the relationship between each claim pair.

RELATIONSHIP TYPES:
- corroborates: Same fact, confirms each other
- contradicts: Conflicting facts that cannot both be true
- updates: Later claim updates/revises earlier (common for casualty counts)
- complements: Related but different aspects of same event
- unrelated: No meaningful relationship

IMPORTANT for casualty numbers:
- If numbers INCREASE over time (e.g., "5 dead" → "36 dead"), this is typically an UPDATE, not contradiction
- Disaster death tolls almost always increase as bodies are found
- Only mark as "contradicts" if numbers conflict at SAME time or decrease unexpectedly

{chr(10).join(pairs_text)}

Return JSON array with format:
[{{"pair": 1, "relation": "updates", "reason": "death toll increased from 5 to 36"}}]
"""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            data = json.loads(response.choices[0].message.content)
            results = {}

            # Handle both array and object with 'results' key
            items = data if isinstance(data, list) else data.get('results', data.get('pairs', []))

            for item in items:
                idx = item['pair'] - 1
                if 0 <= idx < len(pairs_for_llm):
                    c1, c2 = pairs_for_llm[idx]
                    key = tuple(sorted([c1.id, c2.id]))
                    results[key] = item['relation']
                    self.relation_cache[key] = item['relation']

            # Combine pre-classified and LLM-classified
            results.update(pre_classified)
            return results

        except Exception as e:
            print(f"  ⚠️ Batch classification failed: {e}")
            return pre_classified  # Return at least the pre-classified ones

    def temporal_likelihood(self, claim_new: ClaimWithMetadata, claim_old: ClaimWithMetadata) -> float:
        """
        P(new_claim | old_claim, temporal_ordering, domain_knowledge)

        In disasters, casualty counts follow predictable patterns:
        - Initial undercount (chaos, missing persons)
        - Gradual increase (bodies found, hospital deaths)
        - Final stabilization

        Returns likelihood factor (0.3 - 0.9)
        """
        if not claim_new.event_time or not claim_old.event_time:
            return 0.5  # No temporal info - neutral

        time_delta = (claim_new.event_time - claim_old.event_time).total_seconds()

        # Check for numeric metrics
        nums_new = claim_new.extracted_numbers
        nums_old = claim_old.extracted_numbers

        for key in set(nums_new.keys()) & set(nums_old.keys()):
            val_new = nums_new[key]
            val_old = nums_old[key]

            if key == 'deaths':
                if time_delta > 0 and val_new > val_old:
                    # Later, higher death count - EXPECTED in disasters
                    # This is evidence FOR the new claim being accurate update
                    return 0.80
                elif time_delta > 0 and val_new < val_old:
                    # Later, LOWER count - unusual, might be correction
                    return 0.45
                elif time_delta > 0 and val_new == val_old:
                    # Same count, later time - corroboration
                    return 0.85
                elif time_delta < 0 and val_new > val_old:
                    # Earlier claim has higher count than later - suspicious
                    return 0.35

            elif key in ['injured', 'missing']:
                # Missing typically decreases (found alive or dead)
                # Injured can go either way
                if time_delta > 0:
                    return 0.70  # Later info generally more reliable

        return 0.5  # No temporal pattern detected

    def compute_posterior(self, claim: ClaimWithMetadata, neighbors: List[Tuple[str, float, str]]) -> BayesianResult:
        """
        Compute posterior probability using Bayes' theorem.

        P(claim accurate | evidence) ∝ P(evidence | claim accurate) × P(claim accurate)

        neighbors: [(claim_id, similarity, relation_type)]
        """
        prior = claim.prior

        evidence_for = []
        evidence_against = []
        temporal_factors = []

        log_likelihood = 0.0  # Work in log space for numerical stability

        for neighbor_id, similarity, relation in neighbors:
            neighbor = self.claims[neighbor_id]

            if relation == 'corroborates':
                # Corroboration INCREASES plausibility
                # factor > 1 means positive evidence
                factor = 1.2 + 0.3 * similarity * neighbor.prior
                log_likelihood += math.log(factor)
                evidence_for.append(neighbor_id)

            elif relation == 'contradicts':
                # Contradiction DECREASES plausibility
                # But weight by relative source strength
                relative_strength = neighbor.prior / max(prior, 0.1)
                factor = 0.7 - 0.2 * similarity * min(relative_strength, 1.5)
                factor = max(0.3, factor)  # Floor
                log_likelihood += math.log(factor)
                evidence_against.append(neighbor_id)

            elif relation == 'updates':
                # Updates: if this claim is the LATER one updating earlier, boost it
                # If this claim is the EARLIER one being updated, slight decrease
                temporal_factor = self.temporal_likelihood(claim, neighbor)
                temporal_factors.append(temporal_factor)

                if temporal_factor > 0.6:
                    # We are the newer claim with expected progression - boost
                    factor = 1.1 + 0.2 * temporal_factor
                    log_likelihood += math.log(factor)
                    evidence_for.append(neighbor_id)
                else:
                    # We are being superseded or unexpected pattern
                    factor = 0.8
                    log_likelihood += math.log(factor)
                    evidence_against.append(neighbor_id)

            elif relation == 'complements':
                # Complementary claims - mild positive signal (factor > 1)
                factor = 1.0 + 0.1 * similarity
                log_likelihood += math.log(factor)

            # 'unrelated' contributes nothing (likelihood = 1, log = 0)

        # Convert back from log space
        likelihood = math.exp(log_likelihood) if log_likelihood != 0 else 1.0

        # Posterior (unnormalized - we'll normalize across all claims)
        posterior_unnorm = prior * likelihood

        # Confidence based on amount of evidence
        n_evidence = len(evidence_for) + len(evidence_against)
        confidence = min(0.95, 0.3 + 0.1 * n_evidence)  # More evidence = more confident

        avg_temporal = np.mean(temporal_factors) if temporal_factors else 1.0

        return BayesianResult(
            claim_id=claim.id,
            prior=prior,
            likelihood=likelihood,
            posterior=posterior_unnorm,  # Will be normalized later
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            temporal_factor=avg_temporal,
            confidence=confidence
        )

    async def analyze(self, claims: List[ClaimWithMetadata]) -> TopologyAnalysis:
        """
        Full Bayesian analysis of claim network.
        """
        print(f"\n  🧮 Bayesian Analysis of {len(claims)} claims")

        # Store claims
        for claim in claims:
            self.claims[claim.id] = claim

        # Generate embeddings if needed
        print(f"  📊 Generating embeddings...")
        for i, claim in enumerate(claims):
            if claim.id not in self.embeddings:
                self.embeddings[claim.id] = await self.generate_embedding(claim.text)
            if (i + 1) % 20 == 0:
                print(f"      {i+1}/{len(claims)}...")

        # Build similarity network
        print(f"  🕸️  Building similarity network...")
        high_similarity_pairs = []

        for i, c1 in enumerate(claims):
            for c2 in claims[i+1:]:
                sim = self.cosine_similarity(self.embeddings[c1.id], self.embeddings[c2.id])
                if sim > 0.4:
                    self.network.setdefault(c1.id, []).append((c2.id, sim))
                    self.network.setdefault(c2.id, []).append((c1.id, sim))

                    # High similarity pairs need LLM classification
                    if sim > 0.6:
                        high_similarity_pairs.append((c1, c2))

        total_edges = sum(len(n) for n in self.network.values()) // 2
        print(f"      {total_edges} edges, {len(high_similarity_pairs)} high-similarity pairs")

        # Batch classify high-similarity pairs
        if high_similarity_pairs:
            print(f"  🤖 Classifying {len(high_similarity_pairs)} claim relationships...")
            # Process in batches of 20
            for batch_start in range(0, len(high_similarity_pairs), 20):
                batch = high_similarity_pairs[batch_start:batch_start+20]
                await self.batch_classify_relations(batch)

        # Compute posteriors for each claim
        print(f"  📈 Computing posteriors...")
        results = {}

        for claim in claims:
            neighbors = []
            for neighbor_id, similarity in self.network.get(claim.id, []):
                cache_key = tuple(sorted([claim.id, neighbor_id]))
                relation = self.relation_cache.get(cache_key, 'complements')
                neighbors.append((neighbor_id, similarity, relation))

            result = self.compute_posterior(claim, neighbors)
            results[claim.id] = result

        # Compute date consensus - dates are factual claims too
        date_counts = defaultdict(int)
        for claim in claims:
            if claim.event_time:
                # Use date (not time) for consensus
                date_key = claim.event_time.strftime('%Y-%m-%d')
                date_counts[date_key] += 1

        consensus_date = None
        if date_counts:
            consensus_date = max(date_counts.keys(), key=lambda d: date_counts[d])
            consensus_count = date_counts[consensus_date]
            total_dated = sum(date_counts.values())
            print(f"      Date consensus: {consensus_date} ({consensus_count}/{total_dated} claims)")

            # Penalize claims with outlier dates
            for claim in claims:
                if claim.event_time:
                    claim_date = claim.event_time.strftime('%Y-%m-%d')
                    if claim_date != consensus_date:
                        # Different date - check how different
                        claim_year = claim.event_time.year
                        consensus_year = int(consensus_date[:4])
                        if claim_year != consensus_year:
                            # Wrong year - significant penalty
                            penalty = 0.3
                            print(f"      ⚠️ Date outlier: {claim_date} (claim: {claim.text[:40]}...)")
                        else:
                            # Same year, different day - minor penalty
                            penalty = 0.9

                        results[claim.id].posterior *= penalty

        # Normalize posteriors using log-odds transformation
        # This preserves relative ordering while making them interpretable
        log_posteriors = [math.log(r.posterior + 1e-10) for r in results.values()]
        mean_log = np.mean(log_posteriors)
        std_log = np.std(log_posteriors) if len(log_posteriors) > 1 else 1.0

        for result in results.values():
            # Z-score in log space, then sigmoid to [0,1]
            z = (math.log(result.posterior + 1e-10) - mean_log) / max(std_log, 0.1)
            # Sigmoid with temperature to spread values
            result.posterior = 1 / (1 + math.exp(-z * 0.8))
            # Clamp to reasonable range
            result.posterior = min(0.95, max(0.10, result.posterior))

        # Find contradictions (claims with 'contradicts' relations)
        contradictions = []
        seen_pairs = set()
        for claim_id, result in results.items():
            for contra_id in result.evidence_against:
                pair = tuple(sorted([claim_id, contra_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    cache_key = tuple(sorted([claim_id, contra_id]))
                    if self.relation_cache.get(cache_key) == 'contradicts':
                        contradictions.append({
                            'claim1': claim_id,
                            'claim2': contra_id,
                            'text1': self.claims[claim_id].text[:100],
                            'text2': self.claims[contra_id].text[:100],
                            'posterior1': results[claim_id].posterior,
                            'posterior2': results[contra_id].posterior,
                        })

        # Find consensus (high posterior + multiple corroborations)
        consensus_claims = [
            cid for cid, r in results.items()
            if r.posterior > 0.7 and len(r.evidence_for) >= 2
        ]

        # Compute overall entropy
        posteriors = [r.posterior for r in results.values()]
        entropy = -sum(p * math.log(p + 1e-10) for p in posteriors) / len(posteriors)

        # Determine pattern
        n_updates = sum(1 for key, rel in self.relation_cache.items() if rel == 'updates')
        n_contradicts = len(contradictions)
        n_corroborates = sum(1 for key, rel in self.relation_cache.items() if rel == 'corroborates')

        if n_updates > n_contradicts and n_updates > 3:
            pattern = 'progressive'
        elif n_contradicts > n_corroborates * 0.3:
            pattern = 'contradictory'
        elif n_corroborates > len(claims) * 0.5:
            pattern = 'consensus'
        else:
            pattern = 'mixed'

        print(f"      Pattern: {pattern}")
        print(f"      Contradictions: {len(contradictions)}, Consensus claims: {len(consensus_claims)}")

        return TopologyAnalysis(
            results=results,
            contradictions=contradictions,
            consensus_claims=consensus_claims,
            entropy=entropy,
            pattern=pattern,
            consensus_date=consensus_date
        )

    async def generate_narrative(self, analysis: TopologyAnalysis) -> str:
        """
        Generate narrative that reflects our actual epistemic state.

        - High posterior + high confidence → state as fact
        - High posterior + low confidence → state with source attribution
        - Contested (similar posteriors) → acknowledge uncertainty
        - Low posterior → omit or mark as unverified
        """
        self.llm_calls += 1

        # Categorize claims by epistemic status
        facts = []  # High posterior, good confidence
        attributed = []  # High posterior, lower confidence
        uncertain = []  # Medium posterior
        contested = []  # In contradiction groups with similar posteriors

        contested_ids = set()
        for contra in analysis.contradictions:
            diff = abs(contra['posterior1'] - contra['posterior2'])
            if diff < 0.2:  # Similar posteriors - genuinely contested
                contested_ids.add(contra['claim1'])
                contested_ids.add(contra['claim2'])
                contested.append(contra)

        for claim_id, result in analysis.results.items():
            if claim_id in contested_ids:
                continue

            claim = self.claims[claim_id]

            if result.posterior >= 0.70 and result.confidence >= 0.6:
                facts.append((claim, result))
            elif result.posterior >= 0.60:
                attributed.append((claim, result))
            elif result.posterior >= 0.40:
                uncertain.append((claim, result))

        # Sort by posterior
        facts.sort(key=lambda x: x[1].posterior, reverse=True)
        attributed.sort(key=lambda x: x[1].posterior, reverse=True)

        # Format for prompt - include dates
        def format_claim(c, r):
            date_str = c.event_time.strftime('%Y-%m-%d') if c.event_time else 'no date'
            return f"  [{r.posterior:.2f}] [{date_str}] {c.text}"

        facts_text = "\n".join(format_claim(c, r) for c, r in facts[:15])
        attributed_text = "\n".join(format_claim(c, r) for c, r in attributed[:10])

        contested_text = "\n".join(
            f"  • {c['text1'][:80]}... (posterior: {c['posterior1']:.2f})\n"
            f"    vs {c['text2'][:80]}... (posterior: {c['posterior2']:.2f})"
            for c in contested[:5]
        )

        # Include consensus date if available
        date_info = f"EVENT DATE: {analysis.consensus_date}" if analysis.consensus_date else "EVENT DATE: Unknown"

        prompt = f"""Generate a factual news narrative based ONLY on the claims provided below.

{date_info}

CRITICAL RULES:
- ONLY use information explicitly stated in the claims
- Use the EVENT DATE above as the authoritative date
- NEVER fabricate dates, names, numbers, or details not in the claims

ESTABLISHED FACTS (posterior ≥0.70) - format: [score] [date] text:
{facts_text or "  (none)"}

SUPPORTING CLAIMS (posterior ≥0.60):
{attributed_text or "  (none)"}

CONTESTED (similar evidence for both sides):
{contested_text or "  (none)"}

INSTRUCTIONS:
1. State high-posterior facts directly
2. Use EVENT DATE ({analysis.consensus_date or 'unknown'}) for when the incident occurred
3. For contested claims: "Reports vary, with some indicating X and others Y"
4. Structure: What happened → Casualties → Response → Investigation
5. Keep it concise - 3-4 paragraphs max
6. NO speculation or fabrication

Generate the narrative:"""

        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content


async def main():
    """Run Bayesian topology test"""
    print("=" * 80)
    print("🧬 BAYESIAN CLAIM TOPOLOGY TEST (Jaynes-Informed)")
    print("=" * 80)

    print("\n📋 Source Priors (our state of knowledge):")
    for source_type, prior in SOURCE_PRIORS.items():
        print(f"   {source_type}: {prior.base} (+{prior.byline_bonus} byline) - {prior.reason}")

    # Connect to DB
    neo4j = Neo4jService()
    await neo4j.connect()

    # Get fire-related pages with URLs
    print("\n📄 Finding fire-related pages...")
    fire_pages = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE c.text CONTAINS 'Wang Fuk Court' OR c.text CONTAINS 'Tai Po'
        RETURN DISTINCT p.id as page_id, p.url as url
    """)

    page_ids = [p['page_id'] for p in fire_pages]
    page_urls = {p['page_id']: p['url'] for p in fire_pages}

    print(f"   Found {len(page_ids)} pages:")
    for pid, url in page_urls.items():
        print(f"   - {url[:70]}...")

    # Get all claims
    print(f"\n📑 Getting all claims from {len(page_ids)} pages...")
    raw_claims = await neo4j._execute_read("""
        MATCH (p:Page)-[:CONTAINS]->(c:Claim)
        WHERE p.id IN $page_ids
        RETURN c.id as id, c.text as text, c.confidence as confidence,
               c.event_time as event_time, p.id as page_id, p.url as url
        ORDER BY c.event_time, c.id
    """, {'page_ids': page_ids})

    await neo4j.close()

    # Process claims with metadata
    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    analyzer = BayesianTopologyAnalyzer(openai_client)

    claims = []
    print(f"\n🔍 Processing {len(raw_claims)} claims with source priors...")

    source_type_counts = defaultdict(int)

    for raw in raw_claims:
        url = raw['url']
        source_type, has_byline = analyzer.classify_source(url)
        prior = analyzer.compute_source_prior(source_type, has_byline)

        source_type_counts[source_type] += 1

        # Parse event_time
        event_time = None
        if raw['event_time']:
            try:
                if isinstance(raw['event_time'], str):
                    event_time = datetime.fromisoformat(raw['event_time'].replace('Z', '+00:00'))
                else:
                    event_time = raw['event_time']
            except:
                pass

        claim = ClaimWithMetadata(
            id=raw['id'],
            text=raw['text'],
            page_id=raw['page_id'],
            page_url=url,
            event_time=event_time,
            source_type=source_type,
            has_byline=has_byline,
            prior=prior,
            extracted_numbers=analyzer.extract_numbers(raw['text'])
        )
        claims.append(claim)

    print(f"\n   Source type distribution:")
    for stype, count in sorted(source_type_counts.items(), key=lambda x: -x[1]):
        print(f"   - {stype}: {count} claims (prior: {SOURCE_PRIORS[stype].base})")

    # Show claims with extracted numbers
    numeric_claims = [c for c in claims if c.extracted_numbers]
    print(f"\n   Claims with numeric values: {len(numeric_claims)}")
    for c in numeric_claims[:10]:
        print(f"   - {c.extracted_numbers}: {c.text[:60]}...")

    # Run Bayesian analysis
    print(f"\n{'='*80}")
    print("BAYESIAN ANALYSIS")
    print(f"{'='*80}")

    analysis = await analyzer.analyze(claims)

    # Results summary
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")

    posteriors = [r.posterior for r in analysis.results.values()]
    print(f"\n   Posterior distribution:")
    print(f"   - Min: {min(posteriors):.3f}")
    print(f"   - Max: {max(posteriors):.3f}")
    print(f"   - Mean: {np.mean(posteriors):.3f}")
    print(f"   - Std: {np.std(posteriors):.3f}")

    print(f"\n   Pattern: {analysis.pattern}")
    print(f"   Entropy: {analysis.entropy:.3f}")
    print(f"   Contradictions: {len(analysis.contradictions)}")
    print(f"   Consensus claims: {len(analysis.consensus_claims)}")
    print(f"   LLM calls: {analyzer.llm_calls}")

    # Show top claims
    sorted_results = sorted(analysis.results.items(), key=lambda x: x[1].posterior, reverse=True)

    print(f"\n🏆 TOP 10 (highest posterior):")
    for claim_id, result in sorted_results[:10]:
        claim = analyzer.claims[claim_id]
        print(f"   [{result.posterior:.2f}] (prior:{result.prior:.2f}, conf:{result.confidence:.2f}) {claim.source_type}")
        print(f"       {claim.text[:70]}...")
        if result.evidence_for:
            print(f"       ✓ Supported by {len(result.evidence_for)} claims")

    print(f"\n⚠️ BOTTOM 5 (lowest posterior):")
    for claim_id, result in sorted_results[-5:]:
        claim = analyzer.claims[claim_id]
        print(f"   [{result.posterior:.2f}] (prior:{result.prior:.2f}) {claim.source_type}")
        print(f"       {claim.text[:70]}...")
        if result.evidence_against:
            print(f"       ✗ Contradicted by {len(result.evidence_against)} claims")

    # Show contradictions
    if analysis.contradictions:
        print(f"\n⚔️ CONTRADICTIONS:")
        for contra in analysis.contradictions[:5]:
            c1 = analyzer.claims[contra['claim1']]
            c2 = analyzer.claims[contra['claim2']]
            print(f"\n   {contra['text1'][:60]}...")
            print(f"      Posterior: {contra['posterior1']:.2f}, Source: {c1.source_type}")
            print(f"   vs")
            print(f"   {contra['text2'][:60]}...")
            print(f"      Posterior: {contra['posterior2']:.2f}, Source: {c2.source_type}")

            # Show which "wins"
            if abs(contra['posterior1'] - contra['posterior2']) > 0.15:
                winner = 'claim1' if contra['posterior1'] > contra['posterior2'] else 'claim2'
                print(f"   → Resolution: {winner} has higher posterior")
            else:
                print(f"   → Genuinely contested (similar posteriors)")

    # Generate narrative
    print(f"\n{'='*80}")
    print("📖 GENERATING BAYESIAN NARRATIVE")
    print(f"{'='*80}")

    narrative = await analyzer.generate_narrative(analysis)
    print(narrative)

    # Save results
    output = {
        'timestamp': datetime.utcnow().isoformat(),
        'method': 'bayesian_topology',
        'priors': {k: {'base': v.base, 'byline_bonus': v.byline_bonus} for k, v in SOURCE_PRIORS.items()},
        'summary': {
            'total_claims': len(claims),
            'total_pages': len(page_ids),
            'llm_calls': analyzer.llm_calls,
            'pattern': analysis.pattern,
            'entropy': analysis.entropy,
            'posterior_stats': {
                'min': float(min(posteriors)),
                'max': float(max(posteriors)),
                'mean': float(np.mean(posteriors)),
                'std': float(np.std(posteriors)),
            },
            'contradictions': len(analysis.contradictions),
            'consensus_claims': len(analysis.consensus_claims),
        },
        'source_distribution': dict(source_type_counts),
        'top_claims': [
            {
                'id': cid,
                'text': analyzer.claims[cid].text,
                'posterior': r.posterior,
                'prior': r.prior,
                'source_type': analyzer.claims[cid].source_type,
                'evidence_for': len(r.evidence_for),
                'evidence_against': len(r.evidence_against),
            }
            for cid, r in sorted_results[:20]
        ],
        'contradictions': analysis.contradictions,
        'narrative': narrative,
    }

    output_file = '/tmp/bayesian_topology_result.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
