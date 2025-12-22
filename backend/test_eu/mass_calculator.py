"""
Experiment 1: Mass Derivation

Can we compute meaningful "mass" from existing graph data?

Mass components:
- corroboration: count(CORROBORATES relationships)
- tension: count(CONTRADICTS relationships)
- entity_weight: sum of entity mention counts (normalized)
- source_diversity: count(distinct pages that emit similar claims)
- downstream: count(claims that reference this via CORROBORATES/CONTRADICTS/UPDATES)

Run inside container:
    docker exec herenews-app python /app/test_eu/mass_calculator.py
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

from load_graph import load_snapshot, GraphSnapshot, ClaimData, EntityData


@dataclass
class MassVector:
    """Multi-dimensional mass score"""
    corroboration: float = 0.0    # How many claims support this
    tension: float = 0.0          # Contradiction intensity
    entity_weight: float = 0.0    # Importance of entities involved
    downstream: float = 0.0       # How many claims reference this

    def total(self, weights: Dict[str, float] = None) -> float:
        """Weighted sum of mass components"""
        w = weights or {
            'corroboration': 0.30,
            'tension': 0.25,
            'entity_weight': 0.25,
            'downstream': 0.20
        }
        return (
            self.corroboration * w['corroboration'] +
            self.tension * w['tension'] +
            self.entity_weight * w['entity_weight'] +
            self.downstream * w['downstream']
        )

    def __repr__(self):
        return f"Mass(corr={self.corroboration:.2f}, tens={self.tension:.2f}, ent={self.entity_weight:.2f}, down={self.downstream:.2f}, total={self.total():.2f})"


def compute_mass(
    claim: ClaimData,
    snapshot: GraphSnapshot,
    max_entity_mentions: int
) -> MassVector:
    """
    Compute mass vector for a single claim.

    All values normalized to 0-1 range.
    """
    mass = MassVector()

    # Corroboration: log scale (diminishing returns)
    corr_count = len(claim.corroborated_by_ids)
    if corr_count > 0:
        mass.corroboration = min(1.0, math.log1p(corr_count) / math.log1p(10))  # 10+ corr = 1.0

    # Tension: contradictions are rarer, so weight them higher
    contra_count = len(claim.contradicted_by_ids)
    if contra_count > 0:
        mass.tension = min(1.0, contra_count / 3.0)  # 3+ contradictions = 1.0

    # Entity weight: average importance of mentioned entities
    if claim.entity_ids:
        entity_scores = []
        for eid in claim.entity_ids:
            entity = snapshot.entities.get(eid)
            if entity:
                # Normalize by max mentions
                score = entity.mention_count / max_entity_mentions
                entity_scores.append(score)
        if entity_scores:
            mass.entity_weight = sum(entity_scores) / len(entity_scores)

    # Downstream: how many claims reference this one
    downstream_count = (
        len(claim.corroborated_by_ids) +
        len(claim.contradicted_by_ids) +
        len([c for c in snapshot.claims.values() if claim.id in c.updates_ids])
    )
    if downstream_count > 0:
        mass.downstream = min(1.0, math.log1p(downstream_count) / math.log1p(15))

    return mass


def compute_all_masses(snapshot: GraphSnapshot) -> Dict[str, MassVector]:
    """Compute mass for all claims"""
    # Find max entity mentions for normalization
    max_mentions = max(e.mention_count for e in snapshot.entities.values()) if snapshot.entities else 1

    masses = {}
    for cid, claim in snapshot.claims.items():
        masses[cid] = compute_mass(claim, snapshot, max_mentions)

    return masses


def analyze_mass_distribution(
    masses: Dict[str, MassVector],
    snapshot: GraphSnapshot
) -> Dict:
    """Analyze mass distribution and patterns"""
    results = {
        'total_claims': len(masses),
        'mass_stats': {},
        'top_by_total': [],
        'top_by_component': {},
        'event_mass_distribution': {},
        'browsable_threshold_analysis': {}
    }

    # Basic stats
    totals = [m.total() for m in masses.values()]
    results['mass_stats'] = {
        'min': min(totals),
        'max': max(totals),
        'mean': sum(totals) / len(totals),
        'median': sorted(totals)[len(totals) // 2],
        'zero_mass': sum(1 for t in totals if t == 0),
        'high_mass': sum(1 for t in totals if t > 0.3),
    }

    # Top claims by total mass
    sorted_by_total = sorted(masses.items(), key=lambda x: x[1].total(), reverse=True)
    results['top_by_total'] = [
        {
            'id': cid,
            'text': snapshot.claims[cid].text[:100],
            'mass': mass.total(),
            'vector': {
                'corroboration': mass.corroboration,
                'tension': mass.tension,
                'entity_weight': mass.entity_weight,
                'downstream': mass.downstream
            }
        }
        for cid, mass in sorted_by_total[:20]
    ]

    # Top by each component
    for component in ['corroboration', 'tension', 'entity_weight', 'downstream']:
        sorted_by_comp = sorted(masses.items(), key=lambda x: getattr(x[1], component), reverse=True)
        results['top_by_component'][component] = [
            {
                'id': cid,
                'text': snapshot.claims[cid].text[:80],
                'value': getattr(mass, component)
            }
            for cid, mass in sorted_by_comp[:5]
            if getattr(mass, component) > 0
        ]

    # Mass distribution by event
    for eid, event in snapshot.events.items():
        event_masses = [masses[cid].total() for cid in event.claim_ids if cid in masses]
        if event_masses:
            results['event_mass_distribution'][event.canonical_name] = {
                'claim_count': len(event_masses),
                'mean_mass': sum(event_masses) / len(event_masses),
                'max_mass': max(event_masses),
                'high_mass_claims': sum(1 for m in event_masses if m > 0.3)
            }

    # Browsable threshold analysis
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        browsable = sum(1 for t in totals if t >= threshold)
        results['browsable_threshold_analysis'][str(threshold)] = {
            'count': browsable,
            'percentage': browsable / len(totals) * 100
        }

    return results


def main():
    print("=" * 60)
    print("EU Experiment 1: Mass Calculation")
    print("=" * 60)

    # Load snapshot
    snapshot = load_snapshot()
    print(f"\nLoaded: {len(snapshot.claims)} claims, {len(snapshot.entities)} entities")

    # Compute masses
    print("\nComputing mass vectors...")
    masses = compute_all_masses(snapshot)

    # Analyze
    print("Analyzing distribution...")
    results = analyze_mass_distribution(masses, snapshot)

    # Print results
    print("\n" + "=" * 60)
    print("Mass Distribution Stats")
    print("=" * 60)
    stats = results['mass_stats']
    print(f"  Min:        {stats['min']:.4f}")
    print(f"  Max:        {stats['max']:.4f}")
    print(f"  Mean:       {stats['mean']:.4f}")
    print(f"  Median:     {stats['median']:.4f}")
    print(f"  Zero mass:  {stats['zero_mass']} ({stats['zero_mass']/len(masses)*100:.1f}%)")
    print(f"  High mass:  {stats['high_mass']} ({stats['high_mass']/len(masses)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Top 10 Claims by Mass")
    print("=" * 60)
    for i, item in enumerate(results['top_by_total'][:10], 1):
        v = item['vector']
        print(f"\n{i}. [{item['mass']:.3f}] {item['text']}...")
        print(f"   corr={v['corroboration']:.2f} tens={v['tension']:.2f} ent={v['entity_weight']:.2f} down={v['downstream']:.2f}")

    print("\n" + "=" * 60)
    print("Top Claims by Component")
    print("=" * 60)
    for comp, items in results['top_by_component'].items():
        print(f"\n{comp.upper()}:")
        for item in items[:3]:
            print(f"  [{item['value']:.2f}] {item['text']}...")

    print("\n" + "=" * 60)
    print("Mass Distribution by Event")
    print("=" * 60)
    sorted_events = sorted(
        results['event_mass_distribution'].items(),
        key=lambda x: x[1]['mean_mass'],
        reverse=True
    )
    for name, stats in sorted_events:
        print(f"  {name}")
        print(f"    claims={stats['claim_count']}, mean={stats['mean_mass']:.3f}, max={stats['max_mass']:.3f}, high_mass={stats['high_mass_claims']}")

    print("\n" + "=" * 60)
    print("Browsable Threshold Analysis")
    print("=" * 60)
    print("  If we only show claims above threshold:")
    for thresh, data in results['browsable_threshold_analysis'].items():
        print(f"    >{thresh}: {data['count']} claims ({data['percentage']:.1f}%)")

    # Save results
    output_path = Path("/app/test_eu/results/mass_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
