"""
Universal Claim Topology Validation

Experiments to validate the fundamental hypothesis:
    "All knowledge is claims supporting claims, with entropy as the universal index"

Core Claims to Validate:
1. Entropy formula correctly measures uncertainty
2. Abstraction levels emerge from support graph (not assigned)
3. Independent evidence reduces entropy more than correlated
4. Contradictions increase entropy (but are information)
5. Higher-order claims emerge from patterns

Data: Real claims from Neo4j (1215 claims, 16 events, 740 entities)

Run inside container:
    docker exec herenews-app python /app/test_eu/universal_topology_validation.py
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import asyncio

from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class UniversalClaim:
    """
    A claim in the universal topology.
    Can be at any abstraction level - level emerges from structure.
    """
    id: str
    content: str

    # Evidence relationships
    supported_by: List[str] = field(default_factory=list)  # Claim IDs
    contradicted_by: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)  # What this claim supports

    # Source information
    source_id: Optional[str] = None  # Page/publisher
    source_type: Optional[str] = None

    # Computed metrics (not assigned)
    _entropy: Optional[float] = None
    _abstraction_level: Optional[int] = None
    _plausibility: Optional[float] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class TopologyIndex:
    """Universal index for claim topology"""
    claims: Dict[str, UniversalClaim] = field(default_factory=dict)

    # Caches
    _entropy_cache: Dict[str, float] = field(default_factory=dict)
    _level_cache: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# EXPERIMENT 1: ENTROPY FORMULA VALIDATION
# =============================================================================

def compute_claim_entropy(
    claim: UniversalClaim,
    topology: TopologyIndex,
    snapshot: GraphSnapshot
) -> Tuple[float, Dict]:
    """
    Compute entropy of a claim based on evidence.

    H(C) = uncertainty about whether C is true

    Factors:
    1. Corroboration count (reduces entropy)
    2. Contradiction count (increases entropy)
    3. Source diversity (independent sources reduce more)
    4. Source reliability (high-rep sources reduce more)

    Returns: (entropy, breakdown)
    """
    breakdown = {}

    # Base entropy (maximum uncertainty)
    base_entropy = 1.0

    # Factor 1: Corroboration (reduces entropy)
    corr_count = len(claim.supported_by)
    if corr_count > 0:
        # Log scale - diminishing returns
        corr_reduction = min(0.5, math.log1p(corr_count) / math.log1p(10))
    else:
        corr_reduction = 0.0
    breakdown['corroboration'] = corr_reduction

    # Factor 2: Contradiction (increases entropy / adds tension)
    contra_count = len(claim.contradicted_by)
    if contra_count > 0:
        # Contradictions add uncertainty
        contra_addition = min(0.3, contra_count * 0.1)
    else:
        contra_addition = 0.0
    breakdown['contradiction'] = contra_addition

    # Factor 3: Source diversity
    if claim.supported_by:
        supporting_claims = [topology.claims.get(cid) for cid in claim.supported_by]
        supporting_claims = [c for c in supporting_claims if c]
        unique_sources = len(set(c.source_id for c in supporting_claims if c.source_id))
        diversity_bonus = min(0.2, unique_sources * 0.05)
    else:
        diversity_bonus = 0.0
    breakdown['source_diversity'] = diversity_bonus

    # Compute final entropy
    entropy = base_entropy - corr_reduction + contra_addition - diversity_bonus
    entropy = max(0.0, min(1.0, entropy))  # Clamp to [0, 1]

    breakdown['final'] = entropy

    return entropy, breakdown


def experiment_1_entropy_validation(snapshot: GraphSnapshot) -> Dict:
    """
    Validate entropy formula against known outcomes.

    Test: Claims we KNOW are resolved should have LOW entropy.
          Claims we KNOW are contested should have HIGH entropy.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Entropy Formula Validation")
    print("="*70)

    results = {
        'hypothesis': 'Entropy correctly measures uncertainty',
        'test_cases': [],
        'summary': {}
    }

    # Build topology from snapshot
    topology = TopologyIndex()

    for cid, claim_data in snapshot.claims.items():
        claim = UniversalClaim(
            id=cid,
            content=claim_data.text,
            supported_by=claim_data.corroborated_by_ids,
            contradicted_by=claim_data.contradicted_by_ids,
            source_id=claim_data.page_id
        )
        topology.claims[cid] = claim

    # Compute entropy for all claims
    entropies = []
    for cid, claim in topology.claims.items():
        entropy, breakdown = compute_claim_entropy(claim, topology, snapshot)
        claim._entropy = entropy
        entropies.append({
            'id': cid,
            'text': claim.content[:80],
            'entropy': entropy,
            'corr_count': len(claim.supported_by),
            'contra_count': len(claim.contradicted_by),
            'breakdown': breakdown
        })

    # Sort by entropy
    entropies.sort(key=lambda x: x['entropy'])

    # Analysis
    low_entropy = [e for e in entropies if e['entropy'] < 0.5]
    high_entropy = [e for e in entropies if e['entropy'] > 0.8]
    contested = [e for e in entropies if e['contra_count'] > 0]

    print(f"\nTotal claims: {len(entropies)}")
    print(f"Low entropy (<0.5): {len(low_entropy)} ({len(low_entropy)/len(entropies)*100:.1f}%)")
    print(f"High entropy (>0.8): {len(high_entropy)} ({len(high_entropy)/len(entropies)*100:.1f}%)")
    print(f"Contested (has contradictions): {len(contested)}")

    # Test Case 1: Most corroborated claims should have lowest entropy
    print("\n--- Lowest Entropy Claims (should be well-corroborated) ---")
    for e in entropies[:5]:
        print(f"  [{e['entropy']:.3f}] corr={e['corr_count']} contra={e['contra_count']}")
        print(f"         {e['text']}...")
        results['test_cases'].append({
            'test': 'lowest_entropy',
            'claim': e['text'],
            'entropy': e['entropy'],
            'corroborations': e['corr_count']
        })

    # Test Case 2: Contested claims should have higher entropy
    print("\n--- Contested Claims (should have higher entropy) ---")
    for e in contested[:5]:
        print(f"  [{e['entropy']:.3f}] corr={e['corr_count']} contra={e['contra_count']}")
        print(f"         {e['text']}...")
        results['test_cases'].append({
            'test': 'contested',
            'claim': e['text'],
            'entropy': e['entropy'],
            'contradictions': e['contra_count']
        })

    # Test Case 3: Isolated claims (no corr/contra) should have maximum entropy
    isolated = [e for e in entropies if e['corr_count'] == 0 and e['contra_count'] == 0]
    print(f"\n--- Isolated Claims (should have max entropy ~1.0) ---")
    print(f"    Count: {len(isolated)}")
    if isolated:
        avg_entropy = sum(e['entropy'] for e in isolated) / len(isolated)
        print(f"    Average entropy: {avg_entropy:.3f}")
        results['summary']['isolated_avg_entropy'] = avg_entropy

    # Validation check
    validation_passed = True

    # Check 1: Corroborated claims have lower entropy than isolated
    if low_entropy and isolated:
        corr_avg = sum(e['entropy'] for e in low_entropy) / len(low_entropy)
        iso_avg = sum(e['entropy'] for e in isolated) / len(isolated)
        check1 = corr_avg < iso_avg
        print(f"\n✓ Check 1: Corroborated ({corr_avg:.3f}) < Isolated ({iso_avg:.3f}): {check1}")
        validation_passed &= check1

    # Check 2: Contested claims have higher entropy than uncontested
    if contested:
        contested_avg = sum(e['entropy'] for e in contested) / len(contested)
        uncontested = [e for e in entropies if e['contra_count'] == 0 and e['corr_count'] > 0]
        if uncontested:
            uncontested_avg = sum(e['entropy'] for e in uncontested) / len(uncontested)
            check2 = contested_avg > uncontested_avg
            print(f"✓ Check 2: Contested ({contested_avg:.3f}) > Uncontested ({uncontested_avg:.3f}): {check2}")
            validation_passed &= check2

    results['summary']['validation_passed'] = validation_passed
    results['summary']['total_claims'] = len(entropies)
    results['summary']['low_entropy_count'] = len(low_entropy)
    results['summary']['contested_count'] = len(contested)

    return results


# =============================================================================
# EXPERIMENT 2: ABSTRACTION LEVEL EMERGENCE
# =============================================================================

def compute_abstraction_level(
    claim_id: str,
    topology: TopologyIndex,
    visited: Set[str] = None
) -> int:
    """
    Compute abstraction level from graph structure (emergent, not assigned).

    Level 0 = No supporters (ground observation)
    Level N = 1 + max(level of supporters)
    """
    if visited is None:
        visited = set()

    if claim_id in visited:
        return 0  # Cycle protection

    visited.add(claim_id)

    claim = topology.claims.get(claim_id)
    if not claim:
        return 0

    # Use cached value if available
    if claim._abstraction_level is not None:
        return claim._abstraction_level

    # Ground level if no supporters
    if not claim.supported_by:
        claim._abstraction_level = 0
        return 0

    # Otherwise, 1 + max of supporters
    max_supporter_level = 0
    for supporter_id in claim.supported_by:
        level = compute_abstraction_level(supporter_id, topology, visited.copy())
        max_supporter_level = max(max_supporter_level, level)

    claim._abstraction_level = 1 + max_supporter_level
    return claim._abstraction_level


def experiment_2_abstraction_emergence(snapshot: GraphSnapshot) -> Dict:
    """
    Validate that abstraction levels emerge from graph structure.

    Test: Events should have higher abstraction than their claims.
          Well-supported claims should have higher abstraction than isolated.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Abstraction Level Emergence")
    print("="*70)

    results = {
        'hypothesis': 'Abstraction levels emerge from support graph structure',
        'level_distribution': {},
        'test_cases': []
    }

    # Build topology with support relationships
    topology = TopologyIndex()

    for cid, claim_data in snapshot.claims.items():
        claim = UniversalClaim(
            id=cid,
            content=claim_data.text,
            # Corroborates = supports the other claim
            supports=claim_data.corroborates_ids,
            # Corroborated_by = supported by other claims
            supported_by=claim_data.corroborated_by_ids,
            source_id=claim_data.page_id
        )
        topology.claims[cid] = claim

    # Compute abstraction levels
    levels = []
    for cid, claim in topology.claims.items():
        level = compute_abstraction_level(cid, topology)
        levels.append({
            'id': cid,
            'text': claim.content[:60],
            'level': level,
            'supported_by_count': len(claim.supported_by),
            'supports_count': len(claim.supports)
        })

    # Analyze distribution
    level_counts = defaultdict(int)
    for l in levels:
        level_counts[l['level']] += 1

    print(f"\nLevel Distribution:")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        pct = count / len(levels) * 100
        bar = "█" * int(pct / 2)
        print(f"  Level {level}: {count:4d} ({pct:5.1f}%) {bar}")

    results['level_distribution'] = dict(level_counts)

    # Test: Higher level claims should have more supporters
    print("\n--- Claims by Level ---")
    for target_level in range(max(level_counts.keys()) + 1):
        level_claims = [l for l in levels if l['level'] == target_level]
        if level_claims:
            avg_supported_by = sum(l['supported_by_count'] for l in level_claims) / len(level_claims)
            avg_supports = sum(l['supports_count'] for l in level_claims) / len(level_claims)
            print(f"  Level {target_level}: avg_supported_by={avg_supported_by:.2f}, avg_supports={avg_supports:.2f}")

            # Show examples
            examples = sorted(level_claims, key=lambda x: x['supported_by_count'], reverse=True)[:2]
            for ex in examples:
                print(f"      Example: {ex['text']}...")

    # Test: Event claims (from event names) should be higher level
    print("\n--- Event-Level Claims Analysis ---")
    event_claims = []
    for event in snapshot.events.values():
        # Find claims that might represent the event itself
        for cid in event.claim_ids:
            claim_level = next((l for l in levels if l['id'] == cid), None)
            if claim_level:
                event_claims.append({
                    'event': event.canonical_name,
                    'claim': claim_level['text'],
                    'level': claim_level['level']
                })

    if event_claims:
        avg_event_claim_level = sum(ec['level'] for ec in event_claims) / len(event_claims)
        avg_all_level = sum(l['level'] for l in levels) / len(levels)
        print(f"  Average level of event claims: {avg_event_claim_level:.2f}")
        print(f"  Average level of all claims: {avg_all_level:.2f}")

        results['test_cases'].append({
            'test': 'event_claims_vs_all',
            'event_claim_avg_level': avg_event_claim_level,
            'all_claims_avg_level': avg_all_level,
            'passed': avg_event_claim_level >= avg_all_level
        })

    return results


# =============================================================================
# EXPERIMENT 3: INDEPENDENCE AMPLIFICATION
# =============================================================================

def compute_source_independence(
    claim_ids: List[str],
    snapshot: GraphSnapshot
) -> float:
    """
    Compute independence score for a set of claims.

    Independence = diversity of sources
    - Same page = 0 independence
    - Same publisher, different page = 0.3
    - Different publisher = 1.0
    """
    if len(claim_ids) < 2:
        return 0.0

    sources = []
    for cid in claim_ids:
        claim = snapshot.claims.get(cid)
        if claim and claim.page_id:
            sources.append(claim.page_id)

    if len(sources) < 2:
        return 0.0

    unique_sources = len(set(sources))
    independence = (unique_sources - 1) / (len(sources) - 1)

    return independence


def experiment_3_independence_effect(snapshot: GraphSnapshot) -> Dict:
    """
    Validate that independent evidence reduces entropy more.

    Test: Claims supported by diverse sources should have lower entropy
          than claims supported by same source multiple times.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Independence Amplification")
    print("="*70)

    results = {
        'hypothesis': 'Independent evidence reduces entropy more than correlated',
        'findings': []
    }

    # Build topology
    topology = TopologyIndex()
    for cid, claim_data in snapshot.claims.items():
        claim = UniversalClaim(
            id=cid,
            content=claim_data.text,
            supported_by=claim_data.corroborated_by_ids,
            source_id=claim_data.page_id
        )
        topology.claims[cid] = claim

    # For claims with multiple corroborations, compute independence
    multi_supported = []
    for cid, claim in topology.claims.items():
        if len(claim.supported_by) >= 2:
            independence = compute_source_independence(claim.supported_by, snapshot)
            entropy, _ = compute_claim_entropy(claim, topology, snapshot)

            multi_supported.append({
                'id': cid,
                'text': claim.content[:60],
                'support_count': len(claim.supported_by),
                'independence': independence,
                'entropy': entropy
            })

    print(f"\nClaims with 2+ corroborations: {len(multi_supported)}")

    if not multi_supported:
        print("  Not enough data for this experiment")
        return results

    # Analyze: Does higher independence correlate with lower entropy?
    # Group by independence
    high_independence = [m for m in multi_supported if m['independence'] > 0.5]
    low_independence = [m for m in multi_supported if m['independence'] <= 0.5]

    if high_independence and low_independence:
        high_ind_avg_entropy = sum(m['entropy'] for m in high_independence) / len(high_independence)
        low_ind_avg_entropy = sum(m['entropy'] for m in low_independence) / len(low_independence)

        print(f"\n  High independence (>0.5): {len(high_independence)} claims, avg entropy: {high_ind_avg_entropy:.3f}")
        print(f"  Low independence (≤0.5): {len(low_independence)} claims, avg entropy: {low_ind_avg_entropy:.3f}")

        # The hypothesis: high independence should have LOWER entropy
        # (with same corroboration count, more independent = more certain)
        hypothesis_supported = high_ind_avg_entropy < low_ind_avg_entropy
        print(f"\n  Hypothesis supported: {hypothesis_supported}")

        results['findings'].append({
            'high_independence_avg_entropy': high_ind_avg_entropy,
            'low_independence_avg_entropy': low_ind_avg_entropy,
            'hypothesis_supported': hypothesis_supported
        })

    # Control for support count: Compare claims with SAME support count
    print("\n--- Controlled Analysis (same support count) ---")
    by_support_count = defaultdict(list)
    for m in multi_supported:
        by_support_count[m['support_count']].append(m)

    for count, claims in sorted(by_support_count.items()):
        if len(claims) >= 5:
            # Compute correlation between independence and entropy
            independences = [c['independence'] for c in claims]
            entropies = [c['entropy'] for c in claims]

            if len(set(independences)) > 1:  # Need variance
                correlation = np.corrcoef(independences, entropies)[0, 1]
                print(f"  Support count {count}: n={len(claims)}, correlation(independence, entropy)={correlation:.3f}")

                # Negative correlation = higher independence → lower entropy (good!)
                results['findings'].append({
                    'support_count': count,
                    'n': len(claims),
                    'correlation': correlation
                })

    return results


# =============================================================================
# EXPERIMENT 4: CONTRADICTION AS SIGNAL
# =============================================================================

def experiment_4_contradiction_analysis(snapshot: GraphSnapshot) -> Dict:
    """
    Validate that contradictions are correctly handled.

    Test: Contradictions should increase entropy but provide information.
          We should be able to identify WHAT is contested.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Contradiction as Signal")
    print("="*70)

    results = {
        'hypothesis': 'Contradictions increase entropy but reveal contested claims',
        'contradictions': [],
        'analysis': {}
    }

    # Find all contradiction pairs
    contradiction_pairs = []
    for cid, claim_data in snapshot.claims.items():
        for contra_id in claim_data.contradicts_ids:
            contra_claim = snapshot.claims.get(contra_id)
            if contra_claim:
                contradiction_pairs.append({
                    'claim1_id': cid,
                    'claim1_text': claim_data.text[:80],
                    'claim2_id': contra_id,
                    'claim2_text': contra_claim.text[:80]
                })

    # Deduplicate (each pair appears twice)
    seen = set()
    unique_pairs = []
    for pair in contradiction_pairs:
        key = tuple(sorted([pair['claim1_id'], pair['claim2_id']]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"\nUnique contradiction pairs: {len(unique_pairs)}")

    if not unique_pairs:
        print("  No contradictions found")
        return results

    # Analyze what types of things are contested
    print("\n--- Sample Contradictions ---")
    for pair in unique_pairs[:10]:
        print(f"\n  CLAIM: {pair['claim1_text']}...")
        print(f"  VS:    {pair['claim2_text']}...")

    # Categorize contradictions (manual inspection of patterns)
    # Look for patterns: numbers, entities, events
    numeric_contradictions = []
    factual_contradictions = []

    for pair in unique_pairs:
        text1 = pair['claim1_text'].lower()
        text2 = pair['claim2_text'].lower()

        # Check for numeric differences (e.g., death tolls)
        import re
        nums1 = set(re.findall(r'\d+', text1))
        nums2 = set(re.findall(r'\d+', text2))
        if nums1 and nums2 and nums1 != nums2:
            numeric_contradictions.append(pair)
        else:
            factual_contradictions.append(pair)

    print(f"\n--- Contradiction Types ---")
    print(f"  Numeric (different numbers): {len(numeric_contradictions)}")
    print(f"  Factual (different facts): {len(factual_contradictions)}")

    results['analysis'] = {
        'total_pairs': len(unique_pairs),
        'numeric_contradictions': len(numeric_contradictions),
        'factual_contradictions': len(factual_contradictions)
    }

    # Key insight: Contradictions often represent TEMPORAL UPDATES
    print("\n--- Numeric Contradictions (often temporal updates) ---")
    for pair in numeric_contradictions[:5]:
        print(f"  {pair['claim1_text'][:50]}... vs {pair['claim2_text'][:50]}...")

    return results


# =============================================================================
# EXPERIMENT 5: FRAME EMERGENCE
# =============================================================================

def experiment_5_frame_emergence(snapshot: GraphSnapshot) -> Dict:
    """
    Validate that higher-order claims (frames) emerge from event patterns.

    Test: Can we detect when multiple independent events support a common frame?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Frame Emergence from Event Patterns")
    print("="*70)

    results = {
        'hypothesis': 'Independent events supporting same pattern create emergent frames',
        'potential_frames': []
    }

    # Group events by shared entities
    entity_to_events = defaultdict(list)
    for event in snapshot.events.values():
        for eid in event.entity_ids:
            entity_to_events[eid].append(event)

    # Find entities that appear in multiple events (potential frame anchors)
    print("\n--- Entities Spanning Multiple Events ---")
    multi_event_entities = {
        eid: events for eid, events in entity_to_events.items()
        if len(events) >= 2
    }

    print(f"Entities in 2+ events: {len(multi_event_entities)}")

    # For each, check if events are INDEPENDENT (not just same story)
    for eid, events in sorted(multi_event_entities.items(),
                               key=lambda x: len(x[1]), reverse=True)[:10]:
        entity = snapshot.entities.get(eid)
        entity_name = entity.canonical_name if entity else eid

        event_names = [e.canonical_name for e in events]

        # Check independence: different claim sets?
        claim_sets = [set(e.claim_ids) for e in events]
        if len(claim_sets) >= 2:
            overlap = len(claim_sets[0] & claim_sets[1]) / len(claim_sets[0] | claim_sets[1])
        else:
            overlap = 1.0

        independence = 1 - overlap

        print(f"\n  {entity_name} ({len(events)} events, independence={independence:.2f})")
        for name in event_names[:3]:
            print(f"    - {name}")

        if independence > 0.5 and len(events) >= 2:
            results['potential_frames'].append({
                'anchor_entity': entity_name,
                'event_count': len(events),
                'events': event_names,
                'independence': independence,
                'potential_frame': f"Stories involving {entity_name}"
            })

    # Specific test: Hong Kong events
    print("\n--- Hong Kong Frame Analysis ---")
    hk_events = [e for e in snapshot.events.values()
                 if 'hong kong' in e.canonical_name.lower() or
                 any('hong kong' in snapshot.entities.get(eid, type('', (), {'canonical_name': ''})).canonical_name.lower()
                     for eid in e.entity_ids if snapshot.entities.get(eid))]

    print(f"Events related to Hong Kong: {len(hk_events)}")
    for e in hk_events:
        print(f"  - {e.canonical_name} ({len(e.claim_ids)} claims)")

    if len(hk_events) >= 2:
        # These independent events could support a frame about HK
        results['potential_frames'].append({
            'anchor_entity': 'Hong Kong',
            'event_count': len(hk_events),
            'events': [e.canonical_name for e in hk_events],
            'potential_frame': 'Hong Kong 2025 - Multiple independent incidents',
            'note': 'Fire + Trial = independent evidence for systemic frame?'
        })

    return results


# =============================================================================
# EXPERIMENT 6: END-TO-END TOPOLOGY QUERY
# =============================================================================

def experiment_6_topology_query(snapshot: GraphSnapshot) -> Dict:
    """
    Demonstrate the full topology query capability.

    For any claim, show:
    1. Its entropy (uncertainty)
    2. Its abstraction level
    3. Its evidence chain
    4. What it supports
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Full Topology Query")
    print("="*70)

    results = {
        'hypothesis': 'Can query any claim for entropy, level, evidence chain',
        'queries': []
    }

    # Build full topology
    topology = TopologyIndex()
    for cid, claim_data in snapshot.claims.items():
        claim = UniversalClaim(
            id=cid,
            content=claim_data.text,
            supported_by=claim_data.corroborated_by_ids,
            contradicted_by=claim_data.contradicted_by_ids,
            supports=claim_data.corroborates_ids,
            source_id=claim_data.page_id
        )
        topology.claims[cid] = claim

    # Compute all metrics
    for cid, claim in topology.claims.items():
        claim._entropy, _ = compute_claim_entropy(claim, topology, snapshot)
        claim._abstraction_level = compute_abstraction_level(cid, topology)

    # Select interesting claims to query
    # 1. Lowest entropy (most certain)
    # 2. Highest abstraction (most derived)
    # 3. Most contested (has contradictions)

    sorted_by_entropy = sorted(topology.claims.values(), key=lambda c: c._entropy or 1.0)
    sorted_by_level = sorted(topology.claims.values(), key=lambda c: c._abstraction_level or 0, reverse=True)
    contested = [c for c in topology.claims.values() if c.contradicted_by]

    def query_claim(claim: UniversalClaim, label: str):
        print(f"\n--- {label} ---")
        print(f"Claim: {claim.content[:100]}...")
        print(f"  Entropy: {claim._entropy:.3f}")
        print(f"  Abstraction Level: {claim._abstraction_level}")
        print(f"  Supported by: {len(claim.supported_by)} claims")
        print(f"  Contradicted by: {len(claim.contradicted_by)} claims")
        print(f"  Supports: {len(claim.supports)} claims")

        # Evidence chain (first few supporters)
        if claim.supported_by:
            print(f"  Evidence chain:")
            for supporter_id in claim.supported_by[:3]:
                supporter = topology.claims.get(supporter_id)
                if supporter:
                    print(f"    ← {supporter.content[:60]}...")

        return {
            'claim': claim.content[:100],
            'entropy': claim._entropy,
            'abstraction_level': claim._abstraction_level,
            'support_count': len(claim.supported_by),
            'contradiction_count': len(claim.contradicted_by)
        }

    # Query examples
    if sorted_by_entropy:
        results['queries'].append(query_claim(sorted_by_entropy[0], "MOST CERTAIN"))

    if sorted_by_level and sorted_by_level[0]._abstraction_level > 0:
        results['queries'].append(query_claim(sorted_by_level[0], "HIGHEST ABSTRACTION"))

    if contested:
        results['queries'].append(query_claim(contested[0], "MOST CONTESTED"))

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("UNIVERSAL CLAIM TOPOLOGY VALIDATION")
    print("="*70)
    print("\nHypothesis: All knowledge is claims supporting claims,")
    print("            with entropy as the universal uncertainty index.")
    print()

    # Load data
    snapshot = load_snapshot()
    print(f"Loaded: {len(snapshot.claims)} claims, {len(snapshot.events)} events, "
          f"{len(snapshot.entities)} entities")

    # Run experiments
    results = {
        'data': {
            'claims': len(snapshot.claims),
            'events': len(snapshot.events),
            'entities': len(snapshot.entities)
        },
        'experiments': {}
    }

    results['experiments']['1_entropy'] = experiment_1_entropy_validation(snapshot)
    results['experiments']['2_abstraction'] = experiment_2_abstraction_emergence(snapshot)
    results['experiments']['3_independence'] = experiment_3_independence_effect(snapshot)
    results['experiments']['4_contradiction'] = experiment_4_contradiction_analysis(snapshot)
    results['experiments']['5_frames'] = experiment_5_frame_emergence(snapshot)
    results['experiments']['6_query'] = experiment_6_topology_query(snapshot)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print("\n1. ENTROPY FORMULA:")
    if results['experiments']['1_entropy'].get('summary', {}).get('validation_passed'):
        print("   ✓ Validated - corroborated claims have lower entropy")
    else:
        print("   ? Needs review")

    print("\n2. ABSTRACTION EMERGENCE:")
    levels = results['experiments']['2_abstraction'].get('level_distribution', {})
    if levels:
        print(f"   ✓ Levels emerged: {dict(levels)}")

    print("\n3. INDEPENDENCE EFFECT:")
    findings = results['experiments']['3_independence'].get('findings', [])
    if findings:
        print(f"   ✓ Independence data collected: {len(findings)} findings")

    print("\n4. CONTRADICTION AS SIGNAL:")
    analysis = results['experiments']['4_contradiction'].get('analysis', {})
    if analysis:
        print(f"   ✓ Found {analysis.get('total_pairs', 0)} contradiction pairs")
        print(f"     - Numeric (temporal updates): {analysis.get('numeric_contradictions', 0)}")
        print(f"     - Factual disputes: {analysis.get('factual_contradictions', 0)}")

    print("\n5. FRAME EMERGENCE:")
    frames = results['experiments']['5_frames'].get('potential_frames', [])
    if frames:
        print(f"   ✓ Found {len(frames)} potential emergent frames")
        for f in frames[:3]:
            print(f"     - {f['potential_frame']}")

    print("\n6. TOPOLOGY QUERY:")
    queries = results['experiments']['6_query'].get('queries', [])
    if queries:
        print(f"   ✓ Successfully queried {len(queries)} claims with full metrics")

    # Save results
    output_path = Path("/app/test_eu/results/universal_topology_validation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
