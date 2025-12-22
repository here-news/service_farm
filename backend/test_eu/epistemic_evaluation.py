"""
Epistemic Evaluation with Embeddings and LLM

Evaluates the epistemic quality of our topology using:
1. Embeddings for semantic similarity (not just Jaccard)
2. LLM for relationship classification
3. Real claim data from Neo4j

Key questions:
- Does entropy correlate with actual uncertainty?
- Do corroboration relationships make epistemic sense?
- Can we detect dependent vs independent sources?

Run inside container:
    docker exec herenews-app python /app/test_eu/epistemic_evaluation.py
"""

import json
import os
import sys
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/app/backend')

from openai import OpenAI
from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# EMBEDDING SERVICE
# =============================================================================

class EmbeddingService:
    """Compute embeddings for claims"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.cache: Dict[str, List[float]] = {}
        self.model = "text-embedding-3-small"

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, with caching"""
        if text in self.cache:
            return self.cache[text]

        response = self.client.embeddings.create(
            model=self.model,
            input=text[:8000]  # Truncate if too long
        )
        embedding = response.data[0].embedding
        self.cache[text] = embedding
        return embedding

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts"""
        # Check cache first
        uncached = [(i, t) for i, t in enumerate(texts) if t not in self.cache]

        if uncached:
            # Batch API call
            indices, uncached_texts = zip(*uncached)
            response = self.client.embeddings.create(
                model=self.model,
                input=[t[:8000] for t in uncached_texts]
            )

            for i, emb_data in zip(indices, response.data):
                self.cache[texts[i]] = emb_data.embedding

        return [self.cache[t] for t in texts]

    def cosine_similarity(self, emb_a: List[float], emb_b: List[float]) -> float:
        """Compute cosine similarity between embeddings"""
        a = np.array(emb_a)
        b = np.array(emb_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# LLM CLASSIFIER
# =============================================================================

class LLMClassifier:
    """Use LLM for epistemic relationship classification"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"

    def classify_relationship(self, claim_a: str, claim_b: str) -> Dict:
        """
        Classify the epistemic relationship between two claims.

        Returns:
        - relationship: CORROBORATES | CONTRADICTS | INDEPENDENT | UPDATES
        - confidence: 0.0-1.0
        - reasoning: explanation
        """
        prompt = f"""Analyze the epistemic relationship between these two claims:

CLAIM A: {claim_a}

CLAIM B: {claim_b}

Classify their relationship as one of:
- CORROBORATES: Claims support each other, saying essentially the same thing
- CONTRADICTS: Claims conflict with each other
- UPDATES: Claim B provides new/updated information about what Claim A describes
- INDEPENDENT: Claims are about different things, no epistemic relationship

Also assess:
- Are these claims from the same source (likely copying)?
- Do they add independent epistemic value?

Respond in JSON format:
{{
  "relationship": "CORROBORATES|CONTRADICTS|UPDATES|INDEPENDENT",
  "confidence": 0.0-1.0,
  "same_source_likely": true/false,
  "adds_epistemic_value": true/false,
  "reasoning": "brief explanation"
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "relationship": "INDEPENDENT",
                "confidence": 0.0,
                "reasoning": "Failed to parse response"
            }

    def assess_claim_entropy(self, claim: str, context: List[str]) -> Dict:
        """
        Assess the epistemic uncertainty of a claim given context.

        Returns estimated entropy and reasoning.
        """
        context_str = "\n".join([f"- {c}" for c in context[:10]])

        prompt = f"""Assess the epistemic uncertainty of this claim:

CLAIM: {claim}

CONTEXT (related claims from other sources):
{context_str if context else "(No corroborating claims found)"}

Consider:
1. Is the claim well-supported by multiple independent sources?
2. Are there contradicting claims?
3. Is this a verifiable fact or speculation?
4. How much uncertainty remains?

Rate the epistemic entropy (uncertainty) from 0.0 to 1.0:
- 0.0 = Certain, well-established fact with strong evidence
- 0.5 = Moderate uncertainty, some support but not conclusive
- 1.0 = Maximum uncertainty, unverified claim with no support

Respond in JSON:
{{
  "entropy": 0.0-1.0,
  "support_quality": "strong|moderate|weak|none",
  "has_contradiction": true/false,
  "is_verifiable": true/false,
  "reasoning": "explanation"
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"entropy": 1.0, "reasoning": "Failed to parse"}


# =============================================================================
# EPISTEMIC TOPOLOGY
# =============================================================================

@dataclass
class EpistemicNode:
    """A claim in the epistemic topology"""
    id: str
    text: str
    embedding: Optional[List[float]] = None

    # Relationships
    corroborates: Set[str] = field(default_factory=set)
    contradicts: Set[str] = field(default_factory=set)
    updates: Set[str] = field(default_factory=set)

    # Computed values
    computed_entropy: float = 1.0      # Our formula
    llm_assessed_entropy: float = 1.0  # LLM assessment
    level: int = 0


class EpistemicTopology:
    """Epistemic topology with embedding-based similarity"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embeddings = embedding_service
        self.nodes: Dict[str, EpistemicNode] = {}

    def add_claim(self, claim: ClaimData) -> EpistemicNode:
        """Add a claim and compute its embedding"""
        node = EpistemicNode(
            id=claim.id,
            text=claim.text,
        )

        # Get embedding
        node.embedding = self.embeddings.get_embedding(claim.text)

        # Add existing relationships from graph
        node.corroborates.update(claim.corroborates_ids)
        node.contradicts.update(claim.contradicts_ids)
        node.updates.update(claim.updates_ids)

        self.nodes[claim.id] = node
        return node

    def find_similar(self, claim_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find semantically similar claims using embeddings"""
        if claim_id not in self.nodes:
            return []

        node = self.nodes[claim_id]
        if not node.embedding:
            return []

        similar = []
        for other_id, other in self.nodes.items():
            if other_id == claim_id or not other.embedding:
                continue

            sim = self.embeddings.cosine_similarity(node.embedding, other.embedding)
            if sim >= threshold:
                similar.append((other_id, sim))

        return sorted(similar, key=lambda x: -x[1])

    def compute_entropy(self, claim_id: str) -> float:
        """Compute entropy using our formula"""
        if claim_id not in self.nodes:
            return 1.0

        node = self.nodes[claim_id]
        base = 1.0

        # Corroboration reduces entropy
        n_corr = len(node.corroborates)
        corr_reduction = 0.15 * np.log1p(n_corr) if n_corr > 0 else 0

        # Contradiction adds entropy
        n_contra = len(node.contradicts)
        contra_addition = 0.1 * np.log1p(n_contra) if n_contra > 0 else 0

        entropy = base - corr_reduction + contra_addition
        node.computed_entropy = max(0.0, min(1.0, entropy))

        return node.computed_entropy


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_embedding_vs_jaccard(
    snapshot: GraphSnapshot,
    embedding_service: EmbeddingService,
    sample_size: int = 50
) -> Dict:
    """
    Compare embedding similarity vs Jaccard similarity.
    Which better captures epistemic relationships?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Embedding vs Jaccard Similarity")
    print("=" * 70)

    claims = list(snapshot.claims.values())[:sample_size]

    # Get embeddings for all claims
    print(f"Getting embeddings for {len(claims)} claims...")
    texts = [c.text for c in claims]
    embeddings = embedding_service.batch_embed(texts)

    # Compare similarity methods
    results = {
        'pairs_analyzed': 0,
        'embedding_correlates_better': 0,
        'jaccard_correlates_better': 0,
        'sample_comparisons': [],
    }

    # For claims with known relationships
    for i, claim in enumerate(claims):
        if not claim.corroborates_ids and not claim.contradicts_ids:
            continue

        # Compare with related claims
        for related_id in claim.corroborates_ids[:3]:  # Sample
            if related_id not in snapshot.claims:
                continue

            related = snapshot.claims[related_id]

            # Jaccard similarity
            words_a = set(claim.text.lower().split())
            words_b = set(related.text.lower().split())
            jaccard = len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0

            # Embedding similarity
            emb_related = embedding_service.get_embedding(related.text)
            emb_sim = embedding_service.cosine_similarity(embeddings[i], emb_related)

            results['pairs_analyzed'] += 1
            results['sample_comparisons'].append({
                'claim_a': claim.text[:50],
                'claim_b': related.text[:50],
                'relationship': 'CORROBORATES',
                'jaccard': round(jaccard, 3),
                'embedding': round(emb_sim, 3),
            })

    # Analyze results
    if results['sample_comparisons']:
        jaccards = [c['jaccard'] for c in results['sample_comparisons']]
        embeds = [c['embedding'] for c in results['sample_comparisons']]

        results['avg_jaccard_for_corr'] = np.mean(jaccards)
        results['avg_embedding_for_corr'] = np.mean(embeds)

        print(f"\nFor CORROBORATING claims:")
        print(f"  Avg Jaccard similarity: {results['avg_jaccard_for_corr']:.3f}")
        print(f"  Avg Embedding similarity: {results['avg_embedding_for_corr']:.3f}")

        print(f"\nSample pairs:")
        for comp in results['sample_comparisons'][:5]:
            print(f"  Jaccard={comp['jaccard']:.2f}, Embed={comp['embedding']:.2f}")
            print(f"    A: {comp['claim_a']}...")
            print(f"    B: {comp['claim_b']}...")

    return results


def experiment_llm_relationship_validation(
    snapshot: GraphSnapshot,
    classifier: LLMClassifier,
    sample_size: int = 20
) -> Dict:
    """
    Validate our detected relationships using LLM.
    Do the relationships make epistemic sense?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LLM Relationship Validation")
    print("=" * 70)

    results = {
        'validated_pairs': 0,
        'llm_agrees': 0,
        'llm_disagrees': 0,
        'validations': [],
    }

    # Find claims with relationships
    claims_with_rels = [
        c for c in snapshot.claims.values()
        if c.corroborates_ids or c.contradicts_ids
    ][:sample_size]

    print(f"Validating {len(claims_with_rels)} claims with relationships...")

    for claim in claims_with_rels:
        # Check corroborations
        for rel_id in claim.corroborates_ids[:2]:  # Limit to save API calls
            if rel_id not in snapshot.claims:
                continue

            related = snapshot.claims[rel_id]

            # Get LLM assessment
            assessment = classifier.classify_relationship(claim.text, related.text)

            expected = "CORROBORATES"
            actual = assessment.get('relationship', 'UNKNOWN')
            agrees = (actual == expected)

            results['validated_pairs'] += 1
            if agrees:
                results['llm_agrees'] += 1
            else:
                results['llm_disagrees'] += 1

            results['validations'].append({
                'claim_a': claim.text[:60],
                'claim_b': related.text[:60],
                'expected': expected,
                'llm_says': actual,
                'agrees': agrees,
                'confidence': assessment.get('confidence', 0),
                'reasoning': assessment.get('reasoning', ''),
                'adds_epistemic_value': assessment.get('adds_epistemic_value', True),
            })

            print(f"  {'✓' if agrees else '✗'} Expected {expected}, LLM says {actual}")

        # Check contradictions
        for rel_id in claim.contradicts_ids[:2]:
            if rel_id not in snapshot.claims:
                continue

            related = snapshot.claims[rel_id]
            assessment = classifier.classify_relationship(claim.text, related.text)

            expected = "CONTRADICTS"
            actual = assessment.get('relationship', 'UNKNOWN')
            agrees = (actual == expected)

            results['validated_pairs'] += 1
            if agrees:
                results['llm_agrees'] += 1
            else:
                results['llm_disagrees'] += 1

            results['validations'].append({
                'claim_a': claim.text[:60],
                'claim_b': related.text[:60],
                'expected': expected,
                'llm_says': actual,
                'agrees': agrees,
                'confidence': assessment.get('confidence', 0),
                'reasoning': assessment.get('reasoning', ''),
            })

            print(f"  {'✓' if agrees else '✗'} Expected {expected}, LLM says {actual}")

    # Summary
    if results['validated_pairs'] > 0:
        agreement_rate = results['llm_agrees'] / results['validated_pairs']
        results['agreement_rate'] = agreement_rate
        print(f"\nAgreement rate: {agreement_rate:.1%}")

    return results


def experiment_entropy_correlation(
    snapshot: GraphSnapshot,
    classifier: LLMClassifier,
    embedding_service: EmbeddingService,
    sample_size: int = 30
) -> Dict:
    """
    Compare our computed entropy vs LLM-assessed entropy.
    Does our formula match human/LLM judgment?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Entropy Formula Validation")
    print("=" * 70)

    topology = EpistemicTopology(embedding_service)

    # Build topology - prioritize claims with relationships
    all_claims = list(snapshot.claims.values())
    claims_with_corr = [c for c in all_claims if c.corroborated_by_ids]
    claims_with_contra = [c for c in all_claims if c.contradicted_by_ids]
    other_claims = [c for c in all_claims if not c.corroborated_by_ids and not c.contradicted_by_ids]

    # Mix: claims with relationships + some without
    claims = (
        claims_with_corr[:sample_size//2] +
        claims_with_contra[:sample_size//4] +
        other_claims[:sample_size//4]
    )[:sample_size]

    for claim in claims:
        topology.add_claim(claim)

    results = {
        'claims_assessed': 0,
        'entropy_comparisons': [],
    }

    print(f"Assessing entropy for {len(claims)} claims...")

    for claim in claims:
        node = topology.nodes[claim.id]

        # Our computed entropy
        computed = topology.compute_entropy(claim.id)

        # Get context (corroborating claims)
        context = []
        for corr_id in claim.corroborated_by_ids[:5]:
            if corr_id in snapshot.claims:
                context.append(snapshot.claims[corr_id].text)

        # LLM assessment
        assessment = classifier.assess_claim_entropy(claim.text, context)
        llm_entropy = assessment.get('entropy', 1.0)

        node.llm_assessed_entropy = llm_entropy

        results['claims_assessed'] += 1
        results['entropy_comparisons'].append({
            'claim': claim.text[:60],
            'computed_entropy': round(computed, 3),
            'llm_entropy': round(llm_entropy, 3),
            'difference': round(abs(computed - llm_entropy), 3),
            'n_corroborations': len(claim.corroborated_by_ids),
            'n_contradictions': len(claim.contradicted_by_ids),
            'llm_reasoning': assessment.get('reasoning', ''),
            'support_quality': assessment.get('support_quality', 'unknown'),
        })

        print(f"  Computed={computed:.2f}, LLM={llm_entropy:.2f}, diff={abs(computed-llm_entropy):.2f}")
        print(f"    ({len(claim.corroborated_by_ids)} corr, {len(claim.contradicted_by_ids)} contra)")

    # Analyze correlation
    if results['entropy_comparisons']:
        computed = [c['computed_entropy'] for c in results['entropy_comparisons']]
        llm = [c['llm_entropy'] for c in results['entropy_comparisons']]

        if len(computed) > 2:
            correlation = np.corrcoef(computed, llm)[0, 1]
            results['correlation'] = correlation
            results['mean_absolute_error'] = np.mean([c['difference'] for c in results['entropy_comparisons']])

            print(f"\nCorrelation (computed ↔ LLM entropy): {correlation:.3f}")
            print(f"Mean absolute error: {results['mean_absolute_error']:.3f}")

    return results


def experiment_source_independence(
    snapshot: GraphSnapshot,
    classifier: LLMClassifier,
    embedding_service: EmbeddingService,
    sample_size: int = 20
) -> Dict:
    """
    Detect which sources are independent vs copying.
    Use LLM to assess epistemic independence.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Source Independence Detection")
    print("=" * 70)

    results = {
        'claim_groups_analyzed': 0,
        'independence_assessments': [],
    }

    # Find claims with multiple corroborations
    claims_with_corr = [
        c for c in snapshot.claims.values()
        if len(c.corroborated_by_ids) >= 2
    ][:sample_size]

    print(f"Analyzing {len(claims_with_corr)} claims with multiple corroborations...")

    for claim in claims_with_corr:
        corr_claims = [
            snapshot.claims[cid] for cid in claim.corroborated_by_ids[:5]
            if cid in snapshot.claims
        ]

        if len(corr_claims) < 2:
            continue

        results['claim_groups_analyzed'] += 1

        # Get embeddings
        embeddings = embedding_service.batch_embed([c.text for c in corr_claims])

        # Check pairwise similarity
        high_similarity_pairs = 0
        for i in range(len(corr_claims)):
            for j in range(i + 1, len(corr_claims)):
                sim = embedding_service.cosine_similarity(embeddings[i], embeddings[j])
                if sim > 0.85:  # Very high similarity = likely copying
                    high_similarity_pairs += 1

        # Use LLM to assess one pair
        if len(corr_claims) >= 2:
            assessment = classifier.classify_relationship(
                corr_claims[0].text,
                corr_claims[1].text
            )

            total_pairs = len(corr_claims) * (len(corr_claims) - 1) // 2
            independence_ratio = 1 - (high_similarity_pairs / total_pairs) if total_pairs > 0 else 1

            results['independence_assessments'].append({
                'main_claim': claim.text[:60],
                'n_corroborations': len(corr_claims),
                'high_similarity_pairs': high_similarity_pairs,
                'independence_ratio': round(independence_ratio, 3),
                'llm_same_source': assessment.get('same_source_likely', False),
                'llm_adds_value': assessment.get('adds_epistemic_value', True),
            })

            print(f"  {len(corr_claims)} sources, {high_similarity_pairs} likely copies")
            print(f"    Independence ratio: {independence_ratio:.2f}")

    # Summary
    if results['independence_assessments']:
        avg_ratio = np.mean([a['independence_ratio'] for a in results['independence_assessments']])
        results['avg_independence_ratio'] = avg_ratio
        print(f"\nAverage independence ratio: {avg_ratio:.2f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EPISTEMIC EVALUATION WITH EMBEDDINGS AND LLM")
    print("=" * 70)

    # Initialize services
    print("\nInitializing services...")
    embedding_service = EmbeddingService()
    classifier = LLMClassifier()

    # Load data
    snapshot = load_snapshot()
    print(f"Loaded: {len(snapshot.claims)} claims")

    # Run experiments
    results = {}

    # Experiment 1: Embedding vs Jaccard
    results['embedding_vs_jaccard'] = experiment_embedding_vs_jaccard(
        snapshot, embedding_service, sample_size=50
    )

    # Experiment 2: LLM relationship validation
    results['llm_validation'] = experiment_llm_relationship_validation(
        snapshot, classifier, sample_size=20
    )

    # Experiment 3: Entropy correlation
    results['entropy_correlation'] = experiment_entropy_correlation(
        snapshot, classifier, embedding_service, sample_size=30
    )

    # Experiment 4: Source independence
    results['source_independence'] = experiment_source_independence(
        snapshot, classifier, embedding_service, sample_size=20
    )

    # Summary
    print("\n" + "=" * 70)
    print("EPISTEMIC EVALUATION SUMMARY")
    print("=" * 70)

    exp1 = results['embedding_vs_jaccard']
    exp2 = results['llm_validation']
    exp3 = results['entropy_correlation']
    exp4 = results['source_independence']

    avg_jaccard = exp1.get('avg_jaccard_for_corr', 0)
    avg_embed = exp1.get('avg_embedding_for_corr', 0)
    agreement = exp2.get('agreement_rate', 0)
    correlation = exp3.get('correlation', 0)
    if correlation != correlation:  # NaN check
        correlation = 0.0
    mae = exp3.get('mean_absolute_error', 0)
    indep_ratio = exp4.get('avg_independence_ratio', 0)

    print(f"""
1. EMBEDDING VS JACCARD
   - Avg Jaccard for corroborations: {avg_jaccard:.3f}
   - Avg Embedding for corroborations: {avg_embed:.3f}
   → Embeddings capture semantic similarity better

2. LLM RELATIONSHIP VALIDATION
   - Agreement rate: {agreement:.1%}
   - Validated pairs: {exp2.get('validated_pairs', 0)}
   → Our relationship detection is {'✓ accurate' if agreement > 0.7 else '⚠️ needs improvement'}

3. ENTROPY FORMULA VALIDATION
   - Correlation with LLM judgment: {correlation:.3f}
   - Mean absolute error: {mae:.3f}
   → Our entropy formula {'✓ aligns' if correlation > 0.5 else '⚠️ diverges'} from LLM assessment

4. SOURCE INDEPENDENCE
   - Avg independence ratio: {indep_ratio:.2f}
   → {'✓ Sources are mostly independent' if indep_ratio > 0.7 else '⚠️ Significant copying detected'}
""")

    # Recommendations
    print("\n--- RECOMMENDATIONS ---")

    if exp1.get('avg_embedding_for_corr', 0) > exp1.get('avg_jaccard_for_corr', 0):
        print("✓ Use embeddings for similarity (better semantic capture)")

    if exp2.get('agreement_rate', 0) < 0.7:
        print("⚠️ Consider LLM-based relationship classification")

    if exp3.get('correlation', 0) < 0.5:
        print("⚠️ Entropy formula needs calibration")

    if exp4.get('avg_independence_ratio', 0) < 0.7:
        print("⚠️ Apply dependency discount to corroboration count")

    # Save results
    output_path = Path("/app/test_eu/results/epistemic_evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {'claims': len(snapshot.claims)},
        'experiments': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
