"""
Source Dependency Detection

The problem: 100 outlets copying AP ≠ 100 independent sources.
We need to detect when claims are dependent (copying) vs independent.

Signals:
1. Temporal: Later publication suggests copying
2. Textual: High similarity suggests copying
3. Explicit citation: "according to X" reveals dependency
4. Structural: Same numbers, same order suggests copying
5. Unique detail propagation: Unusual details appearing in both

Run inside container:
    docker exec herenews-app python /app/test_eu/source_dependency_detection.py
"""

import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# DEPENDENCY DETECTION
# =============================================================================

def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text"""
    return re.findall(r'\d+', text)


def extract_proper_nouns(text: str) -> Set[str]:
    """Extract capitalized words (rough proper noun detection)"""
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return set(words)


def text_similarity(text_a: str, text_b: str) -> float:
    """Simple Jaccard similarity on words"""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)

    return intersection / union if union > 0 else 0.0


def structural_similarity(text_a: str, text_b: str) -> float:
    """Check if claims have same structure (numbers, entities)"""
    nums_a = extract_numbers(text_a)
    nums_b = extract_numbers(text_b)

    entities_a = extract_proper_nouns(text_a)
    entities_b = extract_proper_nouns(text_b)

    # Number match
    num_match = 0
    if nums_a and nums_b:
        shared_nums = set(nums_a) & set(nums_b)
        num_match = len(shared_nums) / max(len(nums_a), len(nums_b))

    # Entity match
    entity_match = 0
    if entities_a and entities_b:
        shared_entities = entities_a & entities_b
        entity_match = len(shared_entities) / max(len(entities_a), len(entities_b))

    return 0.5 * num_match + 0.5 * entity_match


def detect_explicit_citation(text: str) -> List[str]:
    """Detect explicit citations like 'according to X'"""
    patterns = [
        r'according to (?:the )?([A-Z][a-zA-Z\s]+)',
        r'(?:the )?([A-Z][a-zA-Z\s]+) reported',
        r'(?:the )?([A-Z][a-zA-Z\s]+) said',
        r'citing (?:the )?([A-Z][a-zA-Z\s]+)',
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return [c.strip() for c in citations if c.strip()]


def compute_dependency_score(
    claim_a: ClaimData,
    claim_b: ClaimData,
    snapshot: GraphSnapshot
) -> Tuple[float, Dict]:
    """
    Compute probability that claim_b depends on claim_a.

    Returns: (score, breakdown)
    """
    breakdown = {}

    # 1. Text similarity
    text_sim = text_similarity(claim_a.text, claim_b.text)
    breakdown['text_similarity'] = text_sim

    # 2. Structural similarity (same numbers, entities)
    struct_sim = structural_similarity(claim_a.text, claim_b.text)
    breakdown['structural_similarity'] = struct_sim

    # 3. Explicit citation detection
    citations_b = detect_explicit_citation(claim_b.text)
    # Check if any citation matches claim_a's source
    # (This is a simplification - in real system we'd match publisher names)
    explicit_cite = 0.0
    if citations_b:
        breakdown['explicit_citations'] = citations_b
        # For now, presence of explicit citation is informative
        explicit_cite = 0.3  # Partial credit for having any citation
    breakdown['explicit_citation_score'] = explicit_cite

    # 4. Same page = definitely related (possibly duplicate extraction)
    same_page = 1.0 if claim_a.page_id == claim_b.page_id else 0.0
    breakdown['same_page'] = same_page

    # 5. Entity overlap (shared context suggests potential dependency)
    if claim_a.entity_ids and claim_b.entity_ids:
        entity_a = set(claim_a.entity_ids)
        entity_b = set(claim_b.entity_ids)
        entity_overlap = len(entity_a & entity_b) / len(entity_a | entity_b)
    else:
        entity_overlap = 0.0
    breakdown['entity_overlap'] = entity_overlap

    # Compute overall dependency score
    # High text + structural similarity = likely copy
    # Same page = definitely dependent
    # Explicit citation = known dependency

    score = (
        0.30 * text_sim +
        0.25 * struct_sim +
        0.15 * explicit_cite +
        0.20 * same_page +
        0.10 * entity_overlap
    )

    # Boost if very high similarity (likely copy/paste)
    if text_sim > 0.8:
        score = min(1.0, score + 0.2)

    breakdown['final_score'] = score

    return score, breakdown


# =============================================================================
# EFFECTIVE CORROBORATION
# =============================================================================

def compute_effective_corroboration(
    claim: ClaimData,
    corroborating_claims: List[ClaimData],
    snapshot: GraphSnapshot
) -> Tuple[float, Dict]:
    """
    Compute effective corroboration accounting for dependencies.

    100 dependent sources ≈ 1 source
    5 independent sources = 5 sources
    """
    if not corroborating_claims:
        return 0.0, {'raw_count': 0, 'effective': 0.0}

    analysis = {
        'raw_count': len(corroborating_claims),
        'claim_dependencies': [],
        'source_groups': [],
    }

    # Build dependency matrix
    n = len(corroborating_claims)
    dependency_matrix = np.zeros((n, n))

    for i, claim_i in enumerate(corroborating_claims):
        for j, claim_j in enumerate(corroborating_claims):
            if i != j:
                score, _ = compute_dependency_score(claim_i, claim_j, snapshot)
                dependency_matrix[i, j] = score

    # Cluster into dependency groups using simple threshold
    DEPENDENCY_THRESHOLD = 0.4

    groups = []
    assigned = set()

    for i in range(n):
        if i in assigned:
            continue

        group = {i}
        for j in range(n):
            if j not in assigned and (dependency_matrix[i, j] > DEPENDENCY_THRESHOLD or
                                       dependency_matrix[j, i] > DEPENDENCY_THRESHOLD):
                group.add(j)

        groups.append(group)
        assigned.update(group)

    # Add any unassigned as singletons
    for i in range(n):
        if i not in assigned:
            groups.append({i})

    analysis['dependency_groups'] = len(groups)
    analysis['largest_group'] = max(len(g) for g in groups) if groups else 0

    # Effective corroboration:
    # Each independent group counts as 1
    # Within a group, additional sources add diminishing returns
    effective = 0.0
    for group in groups:
        group_size = len(group)
        # First source in group = 1.0
        # Additional sources add log diminishing returns
        group_contribution = 1.0 + 0.3 * np.log1p(group_size - 1)
        effective += group_contribution

    analysis['effective_corroboration'] = effective
    analysis['independence_ratio'] = effective / len(corroborating_claims) if corroborating_claims else 0

    return effective, analysis


# =============================================================================
# EXPERIMENT: ANALYZE REAL DATA
# =============================================================================

def analyze_claim_dependencies(snapshot: GraphSnapshot) -> Dict:
    """
    Analyze dependency patterns in real claim data.
    """
    print("\n" + "="*70)
    print("SOURCE DEPENDENCY ANALYSIS")
    print("="*70)

    results = {
        'claims_analyzed': 0,
        'high_dependency_pairs': [],
        'effective_corroboration_analysis': [],
        'summary': {}
    }

    # Find claims with multiple corroborations
    claims_with_corr = []
    for cid, claim in snapshot.claims.items():
        if claim.corroborated_by_ids and len(claim.corroborated_by_ids) >= 2:
            corr_claims = [snapshot.claims.get(c) for c in claim.corroborated_by_ids]
            corr_claims = [c for c in corr_claims if c]
            if len(corr_claims) >= 2:
                claims_with_corr.append((claim, corr_claims))

    print(f"\nClaims with 2+ corroborations: {len(claims_with_corr)}")

    # Analyze each
    all_dependency_scores = []
    all_independence_ratios = []

    for claim, corr_claims in claims_with_corr:
        effective, analysis = compute_effective_corroboration(claim, corr_claims, snapshot)

        raw = analysis['raw_count']
        eff = analysis['effective_corroboration']
        ratio = analysis['independence_ratio']

        all_independence_ratios.append(ratio)

        results['effective_corroboration_analysis'].append({
            'claim': claim.text[:60],
            'raw_corroborations': raw,
            'effective_corroboration': round(eff, 2),
            'independence_ratio': round(ratio, 2),
            'dependency_groups': analysis['dependency_groups']
        })

    # Summary statistics
    if all_independence_ratios:
        avg_independence = sum(all_independence_ratios) / len(all_independence_ratios)
        print(f"\nAverage independence ratio: {avg_independence:.2f}")
        print(f"  (1.0 = all sources independent, <1.0 = some dependencies)")

        results['summary']['avg_independence_ratio'] = avg_independence

    # Find pairs with highest dependency
    print("\n--- Analyzing Pairwise Dependencies ---")

    high_dependency_pairs = []
    sample_claims = list(snapshot.claims.values())[:200]  # Sample for speed

    for i, claim_i in enumerate(sample_claims):
        for j, claim_j in enumerate(sample_claims[i+1:], i+1):
            score, breakdown = compute_dependency_score(claim_i, claim_j, snapshot)

            if score > 0.5:  # High dependency threshold
                high_dependency_pairs.append({
                    'claim_a': claim_i.text[:50],
                    'claim_b': claim_j.text[:50],
                    'dependency_score': round(score, 3),
                    'text_similarity': round(breakdown['text_similarity'], 3),
                    'structural_similarity': round(breakdown['structural_similarity'], 3),
                    'same_page': breakdown['same_page']
                })

    # Sort by dependency score
    high_dependency_pairs.sort(key=lambda x: x['dependency_score'], reverse=True)

    print(f"\nHigh dependency pairs (>0.5): {len(high_dependency_pairs)}")

    print("\n--- Top Dependency Pairs ---")
    for pair in high_dependency_pairs[:10]:
        print(f"\n  [{pair['dependency_score']:.2f}] text_sim={pair['text_similarity']:.2f}, struct_sim={pair['structural_similarity']:.2f}")
        print(f"    A: {pair['claim_a']}...")
        print(f"    B: {pair['claim_b']}...")

    results['high_dependency_pairs'] = high_dependency_pairs[:20]
    results['summary']['high_dependency_count'] = len(high_dependency_pairs)

    # Analyze by event
    print("\n--- Effective Corroboration by Claim ---")

    sorted_by_ratio = sorted(
        results['effective_corroboration_analysis'],
        key=lambda x: x['independence_ratio']
    )

    print("\nLeast Independent (most copying):")
    for item in sorted_by_ratio[:5]:
        print(f"  [{item['independence_ratio']:.2f}] raw={item['raw_corroborations']}, eff={item['effective_corroboration']:.1f}")
        print(f"       {item['claim']}...")

    print("\nMost Independent (diverse sources):")
    for item in sorted_by_ratio[-5:]:
        print(f"  [{item['independence_ratio']:.2f}] raw={item['raw_corroborations']}, eff={item['effective_corroboration']:.1f}")
        print(f"       {item['claim']}...")

    return results


# =============================================================================
# CASE STUDY: HISTORICAL CLAIM
# =============================================================================

def case_study_historical_claim(snapshot: GraphSnapshot) -> Dict:
    """
    Case study: Historical claims like "1948 fire killed 176"

    These often have single primary source but get widely copied.
    How should we handle entropy?
    """
    print("\n" + "="*70)
    print("CASE STUDY: Historical/Derived Claims")
    print("="*70)

    results = {
        'findings': []
    }

    # Find claims that mention historical data (years before 2020)
    historical_claims = []
    for cid, claim in snapshot.claims.items():
        text = claim.text.lower()
        years = re.findall(r'\b(19\d{2}|20[01]\d)\b', text)
        if years:
            historical_claims.append({
                'claim': claim,
                'years_mentioned': years
            })

    print(f"\nClaims mentioning historical years: {len(historical_claims)}")

    # Find claims with "according to" (explicit dependency)
    explicit_citations = []
    for cid, claim in snapshot.claims.items():
        citations = detect_explicit_citation(claim.text)
        if citations:
            explicit_citations.append({
                'claim': claim.text[:80],
                'citations': citations
            })

    print(f"Claims with explicit citations ('according to'): {len(explicit_citations)}")

    print("\n--- Sample Explicit Citations ---")
    for item in explicit_citations[:10]:
        print(f"  Citations: {item['citations']}")
        print(f"  Claim: {item['claim']}...")
        print()

    results['historical_claims_count'] = len(historical_claims)
    results['explicit_citations_count'] = len(explicit_citations)
    results['sample_citations'] = explicit_citations[:10]

    # Analysis: Claims that cite other sources should have:
    # 1. Their entropy based on the CITED source's reliability
    # 2. Not reduced entropy just because many outlets repeat

    print("\n--- Entropy Implications ---")
    print("""
    For a claim like "1948 fire killed 176 (according to SCMP)":

    WRONG approach:
    - 50 outlets report this
    - Entropy = LOW (many corroborations)

    CORRECT approach:
    - 50 outlets report this
    - But ALL cite SCMP as primary source
    - Effective independent sources = 1
    - Entropy = HIGH (single primary source, historical)

    The system should detect:
    1. Explicit citations ("according to X")
    2. Implicit copying (high text similarity)
    3. Source chains (A cites B cites C)

    And compute entropy based on PRIMARY sources, not total mentions.
    """)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SOURCE DEPENDENCY DETECTION")
    print("="*70)
    print("\nProblem: 100 outlets copying AP ≠ 100 independent sources")
    print("Solution: Detect dependencies, compute EFFECTIVE corroboration\n")

    snapshot = load_snapshot()
    print(f"Loaded: {len(snapshot.claims)} claims from {len(snapshot.pages)} pages")

    # Run analysis
    dependency_results = analyze_claim_dependencies(snapshot)
    case_study_results = case_study_historical_claim(snapshot)

    # Combined results
    results = {
        'dependency_analysis': dependency_results,
        'case_study': case_study_results
    }

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    avg_independence = dependency_results.get('summary', {}).get('avg_independence_ratio', 0)
    high_dep_count = dependency_results.get('summary', {}).get('high_dependency_count', 0)

    print(f"""
    1. Average independence ratio: {avg_independence:.2f}
       - Below 1.0 means sources are NOT fully independent
       - Naive corroboration count OVERESTIMATES certainty

    2. High dependency pairs found: {high_dep_count}
       - These are likely copies or derived from same source
       - Should be grouped, not counted separately

    3. Explicit citations detected: {case_study_results.get('explicit_citations_count', 0)}
       - "According to X" reveals source chain
       - Can trace back to primary source

    IMPLICATION FOR ENTROPY:
    - Must compute EFFECTIVE corroboration (discounting dependencies)
    - Historical claims often have SINGLE primary source
    - Widely-reported ≠ well-corroborated
    """)

    # Save
    output_path = Path("/app/test_eu/results/source_dependency_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
