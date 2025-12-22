"""
Basic Form of the Universal Epistemic Engine

First principles: What is the MINIMAL structure?

Claims → relate() → Edges → Connected Components = Events

No embeddings, no entity index, no heuristics.
Just claims, edges, and the emergence of events.
"""

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Claim:
    id: str
    text: str
    source: str


def relate(c1: Claim, c2: Claim) -> str:
    """
    The atomic epistemic operation.

    Given two claims, what is their relationship?

    This is the ONLY place where epistemic logic lives.
    Everything else is just graph operations.
    """
    # For now: manual/oracle - we define the truth
    # In production: LLM or embedding-based
    return 'INDEPENDENT'


def find_events(claims: list[Claim], edges: list[tuple[str, str, str]]) -> list[list[str]]:
    """
    Events EMERGE from connected components.

    This is pure graph theory - no epistemic logic here.
    """
    parent = {c.id: c.id for c in claims}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Only CORROBORATES and UPDATES connect claims into events
    # CONTRADICTS keeps them in same event (they're about the same thing!)
    for c1, c2, rel in edges:
        if rel in ('CORROBORATES', 'CONTRADICTS', 'UPDATES'):
            union(c1, c2)

    components = defaultdict(list)
    for c in claims:
        components[find(c.id)].append(c.id)

    return sorted(components.values(), key=len, reverse=True)


def main():
    """
    Minimal synthetic test case.

    Scenario: Two events + contradiction
    - Event A: Fire in Hong Kong (4 claims, including 1 contradiction)
    - Event B: Earthquake in Japan (2 claims)
    - 1 unrelated claim (singleton)
    """

    # Define claims
    claims = [
        # Event A: Hong Kong Fire
        Claim("a1", "Fire breaks out in Hong Kong apartment building", "BBC"),
        Claim("a2", "12 people killed in Hong Kong high-rise fire", "Reuters"),
        Claim("a3", "Hong Kong fire death toll rises to 12", "AP"),
        Claim("a4", "Only 8 confirmed dead in Hong Kong fire", "SCMP"),  # Contradiction!

        # Event B: Japan Earthquake
        Claim("b1", "Magnitude 6.5 earthquake hits Japan", "NHK"),
        Claim("b2", "Strong earthquake strikes off coast of Japan", "CNN"),

        # Unrelated
        Claim("c1", "Stock market closes at record high", "Bloomberg"),
    ]

    # Define edges (oracle/ground truth)
    # In production, these would be computed by relate()
    edges = [
        # a1 ↔ a2: Same event (fire), different facts (a2 adds death count)
        ("a1", "a2", "CORROBORATES"),

        # a2 ↔ a3: Same fact (death toll = 12)
        ("a2", "a3", "CORROBORATES"),

        # a2 ↔ a4: Different death counts! Contradiction within same event
        ("a2", "a4", "CONTRADICTS"),

        # b1 ↔ b2: Same event (earthquake)
        ("b1", "b2", "CORROBORATES"),

        # c1 is independent of everything
    ]

    print("=" * 50)
    print("BASIC FORM: Universal Epistemic Engine")
    print("=" * 50)

    print("\n--- Claims ---")
    for c in claims:
        print(f"  [{c.id}] {c.text} ({c.source})")

    print("\n--- Edges ---")
    for c1, c2, rel in edges:
        print(f"  {c1} --{rel}--> {c2}")

    print("\n--- Emergent Events ---")
    events = find_events(claims, edges)

    claim_map = {c.id: c for c in claims}
    for i, event in enumerate(events):
        print(f"\nEvent {i+1} ({len(event)} claims):")
        for cid in event:
            print(f"  • {claim_map[cid].text}")

    # Show internal structure of events
    print("\n--- Event Internal Structure ---")
    edge_map = defaultdict(list)
    for c1, c2, rel in edges:
        edge_map[c1].append((c2, rel))
        edge_map[c2].append((c1, rel))

    for i, event in enumerate(events):
        if len(event) < 2:
            continue
        print(f"\nEvent {i+1} structure:")

        # Find edges within this event
        event_set = set(event)
        internal_edges = [(c1, c2, r) for c1, c2, r in edges
                         if c1 in event_set and c2 in event_set]

        corr = [e for e in internal_edges if e[2] == 'CORROBORATES']
        cont = [e for e in internal_edges if e[2] == 'CONTRADICTS']

        print(f"  Corroborations: {len(corr)}")
        for c1, c2, _ in corr:
            print(f"    {c1} ⟷ {c2}")

        if cont:
            print(f"  Contradictions: {len(cont)} ⚠️")
            for c1, c2, _ in cont:
                print(f"    {c1} ⟷ {c2}  [CONTESTED]")

    print("\n--- Analysis ---")
    print(f"Total claims: {len(claims)}")
    print(f"Total edges: {len(edges)}")
    print(f"Emergent events: {len(events)}")
    print(f"  Multi-claim events: {sum(1 for e in events if len(e) > 1)}")
    print(f"  Singleton claims: {sum(1 for e in events if len(e) == 1)}")

    print("\n" + "=" * 50)
    print("THE BASIC FORM")
    print("=" * 50)
    print("""
    claim₁ ─relate()─→ edge ←─relate()─ claim₂
                         │
                         ▼
               connected components
                         │
                         ▼
                      EVENTS

    That's it. Everything else is optimization.
    """)

    print("=" * 50)
    print("WHAT IS relate()?")
    print("=" * 50)
    print("""
    The atomic epistemic operation. Three implementations:

    1. ORACLE (this test)
       - Human defines the truth
       - relate(a, b) → manually specified

    2. RULE-BASED
       - Numeric matching: "12 dead" == "12 dead" → CORROBORATES
       - Numeric conflict: "12 dead" ≠ "8 dead" → CONTRADICTS
       - Semantic similarity: embedding cosine > 0.85 → CORROBORATES

    3. LLM
       - "Do these claims support or contradict each other?"
       - Most powerful, most expensive

    The basic form is agnostic to implementation.
    It only cares about the OUTPUT: CORROBORATES | CONTRADICTS | INDEPENDENT
    """)

    print("=" * 50)
    print("THE SCALING PROBLEM")
    print("=" * 50)
    print("""
    With N claims, there are N×(N-1)/2 pairs.

       N=7      →       21 pairs  ✓
       N=100    →    4,950 pairs  ✓
       N=1000   →  499,500 pairs  ⚠️
       N=10000  → 49,995,000 pairs  ✗

    Solution: ENTITY ROUTING (optimization, not core logic)

    1. Build entity index: entity_id → [claim_ids]
    2. For each claim, only relate() to claims sharing entities
    3. Reduces O(N²) to O(N × avg_claims_per_entity)

    But this is OPTIMIZATION. The basic form is still:
    claims → edges → connected components → events
    """)


if __name__ == "__main__":
    main()
