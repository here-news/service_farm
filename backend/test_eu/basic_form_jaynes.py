"""
Basic Form of the Universal Epistemic Engine - JAYNESIAN FOUNDATION

Jaynes: Probability is extended logic. It quantifies plausibility given information.
Bayes: P(H|E) = P(E|H) × P(H) / P(E) - beliefs update with evidence.
Entropy: H = -Σ p log p - uncertainty about state.

The epistemic engine is not just graph theory.
It's a PROBABILITY MACHINE operating on claims.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from math import log2, exp
from typing import Optional


@dataclass
class Claim:
    id: str
    text: str
    source: str
    # Prior credibility of this source (0-1)
    source_credibility: float = 0.5


@dataclass
class Edge:
    c1_id: str
    c2_id: str
    relation: str  # CORROBORATES | CONTRADICTS
    # Probability that this relation is correct (0-1)
    confidence: float = 1.0


@dataclass
class Event:
    claim_ids: list[str]
    # Derived quantities
    entropy: float = 0.0  # Uncertainty about the event's state
    coherence: float = 0.0  # Internal agreement (0-1)
    mass: float = 0.0  # Total evidence weight


def relate(c1: Claim, c2: Claim) -> Optional[Edge]:
    """
    The atomic epistemic operation.

    Returns an Edge with PROBABILISTIC confidence, not just a label.

    Jaynes would ask: "Given c1 and c2, what is P(same_event)?"
    """
    # For now: oracle. In production: embedding similarity → probability
    return None


def compute_event_entropy(event: Event, claims: dict, edges: list[Edge]) -> float:
    """
    Entropy measures UNCERTAINTY about the event's state.

    Low entropy = consensus (all claims agree)
    High entropy = contested (claims contradict each other)

    H = -Σ p_i log(p_i) where p_i is probability of each "version"
    """
    event_set = set(event.claim_ids)

    # Find contradictions within event
    contradictions = [e for e in edges
                     if e.relation == 'CONTRADICTS'
                     and e.c1_id in event_set
                     and e.c2_id in event_set]

    if not contradictions:
        # No contradictions = certainty = zero entropy
        return 0.0

    # Each contradiction creates uncertainty
    # Simple model: each contested fact adds ~1 bit of entropy
    # More sophisticated: weight by confidence and source credibility

    contested_claims = set()
    for e in contradictions:
        contested_claims.add(e.c1_id)
        contested_claims.add(e.c2_id)

    # Entropy contribution from contested facts
    # Using source credibility as probability weights
    total_cred = sum(claims[cid].source_credibility for cid in contested_claims)

    if total_cred == 0:
        return log2(len(contested_claims))  # Maximum entropy

    H = 0.0
    for cid in contested_claims:
        p = claims[cid].source_credibility / total_cred
        if p > 0:
            H -= p * log2(p)

    return H


def compute_event_mass(event: Event, claims: dict, edges: list[Edge]) -> float:
    """
    Mass = total evidence weight.

    Bayesian: More independent sources corroborating = higher posterior.

    m = Σ credibility_i × (1 + corroboration_bonus)
    """
    event_set = set(event.claim_ids)

    # Count corroborations for each claim
    corr_count = defaultdict(int)
    for e in edges:
        if e.relation == 'CORROBORATES':
            if e.c1_id in event_set:
                corr_count[e.c1_id] += 1
            if e.c2_id in event_set:
                corr_count[e.c2_id] += 1

    mass = 0.0
    for cid in event.claim_ids:
        base = claims[cid].source_credibility
        # Corroboration multiplier (diminishing returns)
        bonus = 1 + 0.5 * log2(1 + corr_count[cid])
        mass += base * bonus

    return mass


def compute_event_coherence(event: Event, edges: list[Edge]) -> float:
    """
    Coherence = internal agreement ratio.

    coherence = corroborations / (corroborations + contradictions)

    1.0 = perfect agreement
    0.5 = balanced conflict
    0.0 = total contradiction
    """
    event_set = set(event.claim_ids)

    corr = sum(1 for e in edges
               if e.relation == 'CORROBORATES'
               and e.c1_id in event_set and e.c2_id in event_set)

    cont = sum(1 for e in edges
               if e.relation == 'CONTRADICTS'
               and e.c1_id in event_set and e.c2_id in event_set)

    if corr + cont == 0:
        return 1.0  # No edges = no conflict = coherent

    return corr / (corr + cont)


def find_events(claims: list[Claim], edges: list[Edge]) -> list[Event]:
    """Events emerge from connected components."""
    parent = {c.id: c.id for c in claims}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for e in edges:
        union(e.c1_id, e.c2_id)

    components = defaultdict(list)
    for c in claims:
        components[find(c.id)].append(c.id)

    return [Event(claim_ids=cids) for cids in components.values()]


def main():
    """
    Jaynesian test case with probabilistic quantities.
    """

    # Claims with source credibility
    claims = [
        # Event A: Hong Kong Fire - different sources with different credibility
        Claim("a1", "Fire breaks out in Hong Kong apartment", "BBC", 0.9),
        Claim("a2", "12 people killed in Hong Kong fire", "Reuters", 0.85),
        Claim("a3", "Death toll confirmed at 12", "AP", 0.85),
        Claim("a4", "Only 8 dead in Hong Kong fire", "tabloid.hk", 0.3),

        # Event B: Japan Earthquake
        Claim("b1", "6.5 magnitude earthquake hits Japan", "NHK", 0.95),
        Claim("b2", "Strong quake strikes Japan coast", "CNN", 0.8),

        # Singleton
        Claim("c1", "Markets close at record high", "Bloomberg", 0.9),
    ]
    claim_map = {c.id: c for c in claims}

    # Edges with confidence
    edges = [
        Edge("a1", "a2", "CORROBORATES", 0.9),
        Edge("a2", "a3", "CORROBORATES", 0.95),  # Same number = high confidence
        Edge("a2", "a4", "CONTRADICTS", 0.99),   # 12 ≠ 8, very confident contradiction
        Edge("b1", "b2", "CORROBORATES", 0.85),
    ]

    print("=" * 60)
    print("JAYNESIAN EPISTEMIC ENGINE")
    print("=" * 60)

    print("\n--- Claims with Source Credibility ---")
    for c in claims:
        print(f"  [{c.id}] {c.text}")
        print(f"       source: {c.source}, credibility: {c.source_credibility}")

    print("\n--- Edges with Confidence ---")
    for e in edges:
        print(f"  {e.c1_id} --{e.relation}({e.confidence:.2f})--> {e.c2_id}")

    # Find events
    events = find_events(claims, edges)

    # Compute Jaynesian quantities
    for event in events:
        event.entropy = compute_event_entropy(event, claim_map, edges)
        event.mass = compute_event_mass(event, claim_map, edges)
        event.coherence = compute_event_coherence(event, edges)

    # Sort by mass (most evidence first)
    events.sort(key=lambda e: e.mass, reverse=True)

    print("\n--- Emergent Events with Jaynesian Quantities ---")
    for i, event in enumerate(events):
        print(f"\nEvent {i+1}: {len(event.claim_ids)} claims")
        print(f"  Mass (evidence weight): {event.mass:.2f}")
        print(f"  Entropy (uncertainty):  {event.entropy:.2f} bits")
        print(f"  Coherence (agreement):  {event.coherence:.0%}")
        print(f"  Claims:")
        for cid in event.claim_ids:
            print(f"    • {claim_map[cid].text}")

    print("\n" + "=" * 60)
    print("JAYNESIAN INTERPRETATION")
    print("=" * 60)
    print("""
    MASS = Bayesian posterior weight
      - More credible sources → higher mass
      - Corroboration → multiplier (independent evidence)
      - P(event|claims) ∝ Π P(claim_i|event) × P(source_i)

    ENTROPY = Epistemic uncertainty
      - H = -Σ p log p over contested versions
      - Low H → consensus, we know what happened
      - High H → contested, truth unclear
      - Credibility weights the probability distribution

    COHERENCE = Internal consistency
      - corr / (corr + cont)
      - 1.0 = all claims agree
      - 0.0 = total disagreement
      - Events can have high mass but low coherence (many sources, but they disagree)

    THE JAYNESIAN MOVE:
      Instead of: relate() → {CORROBORATES, CONTRADICTS}
      We have:    relate() → P(same_event | c1, c2), P(agree | c1, c2)

      Everything becomes probabilistic. No hard categories.
    """)


if __name__ == "__main__":
    main()
