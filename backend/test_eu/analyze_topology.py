"""Analyze topology weights, confidence, and plausibility structure."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_eu.core.kernel import EpistemicKernel
from openai import AsyncOpenAI


async def analyze():
    llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    k = EpistemicKernel(llm_client=llm)

    claims = [
        ("Fire kills 11 in Tai Po", "BBC"),
        ("Blaze at Wang Fuk Court leaves 11 dead", "Reuters"),
        ("Death toll rises to 36 as rescue continues", "SCMP"),
        ("36 confirmed dead in Hong Kong fire", "HK Gov"),
        ("Fire broke out on 14th floor of Wang Fuk Court", "BBC"),
        ("Blaze started around 3am on 14th floor", "Reuters"),
        ("Over 100 firefighters responded to the scene", "SCMP"),
        ("Jimmy Lai's trial continues in Hong Kong", "NYT"),
        ("Lai denies sedition charges", "Guardian"),
        ("Trial enters fourth week", "AP"),
    ]

    print("Processing claims...")
    for text, src in claims:
        await k.process(text, src)

    t = k.topology()

    print("\n" + "=" * 70)
    print("TOPOLOGY WEIGHT/CONFIDENCE/PLAUSIBILITY ANALYSIS")
    print("=" * 70)

    # 1. Node Plausibility
    print("\n┌" + "─" * 68 + "┐")
    print("│ 1. BELIEF PLAUSIBILITY (Jaynes-aligned entropy)                    │")
    print("└" + "─" * 68 + "┘")
    print()
    print("  Plausibility = 1 - Entropy")
    print("  Higher sources → Lower entropy → Higher plausibility")
    print()

    for n in t["nodes"]:
        plausibility = 1.0 - n['entropy']
        bar = "█" * int(plausibility * 20) + "░" * (20 - int(plausibility * 20))

        conf_color = {"confirmed": "★★★", "likely": "★★☆", "reported": "★☆☆"}[n['confidence']]

        print(f"  B{n['id']} {conf_color} [{n['source_count']} src]")
        print(f"     Text: {n['text'][:50]}")
        print(f"     Entropy:      H = {n['entropy']:.2f}")
        print(f"     Plausibility: [{bar}] {plausibility:.0%}")
        print()

    # 2. Edge Weights
    print("\n┌" + "─" * 68 + "┐")
    print("│ 2. EDGE WEIGHTS (connection strength)                              │")
    print("└" + "─" * 68 + "┘")
    print()
    print("  Weight = shared_sources / max(sources_A, sources_B)")
    print("  Higher weight → Stronger connection → More likely same event")
    print()

    if t["edges"]:
        for e in t["edges"]:
            weight = e.get('weight', 0)
            bar = "═" * int(weight * 20) + "─" * (20 - int(weight * 20))
            shared = e.get('shared_sources', [])

            src_node = t["nodes"][e['source']]
            tgt_node = t["nodes"][e['target']]

            print(f"  B{e['source']} ←──────────→ B{e['target']}")
            print(f"     Relation: {e['relation']}")
            print(f"     Shared:   {', '.join(shared)}")
            print(f"     Weight:   [{bar}] {weight:.0%}")
            print()
    else:
        print("  (no edges)")

    # 3. Surface Analysis
    print("\n┌" + "─" * 68 + "┐")
    print("│ 3. SURFACE COHERENCE (event clusters)                              │")
    print("└" + "─" * 68 + "┘")
    print()

    for i, s in enumerate(t["surfaces"]):
        belief_ids = s['beliefs']
        size = s['size']

        # Calculate aggregate metrics
        entropies = [t["nodes"][bid]["entropy"] for bid in belief_ids if bid < len(t["nodes"])]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 1.0
        avg_plausibility = 1.0 - avg_entropy

        # Count confirmed/likely/reported
        confs = [t["nodes"][bid]["confidence"] for bid in belief_ids if bid < len(t["nodes"])]
        confirmed = confs.count("confirmed")
        likely = confs.count("likely")
        reported = confs.count("reported")

        status = "CONNECTED" if size > 1 else "ISOLATED"
        bar = "█" * int(avg_plausibility * 20) + "░" * (20 - int(avg_plausibility * 20))

        print(f"  Surface {i+1} [{status}]")
        print(f"     Beliefs: {', '.join(f'B{b}' for b in belief_ids)}")
        print(f"     Sources: {s['total_sources']} unique")
        print(f"     Composition: {confirmed} confirmed, {likely} likely, {reported} reported")
        print(f"     Aggregate Plausibility: [{bar}] {avg_plausibility:.0%}")
        print()

        # Show beliefs
        for bid in belief_ids:
            if bid < len(k.beliefs):
                b = k.beliefs[bid]
                n = t["nodes"][bid]
                plaus = 1.0 - n['entropy']
                mini_bar = "█" * int(plaus * 5) + "░" * (5 - int(plaus * 5))
                print(f"       └─ B{bid} [{mini_bar}] {n['confidence']:>9} | {b.text[:40]}")
        print()

    # 4. ASCII Graph
    print("\n┌" + "─" * 68 + "┐")
    print("│ 4. TOPOLOGY GRAPH                                                  │")
    print("└" + "─" * 68 + "┘")
    print()

    # Group by surface
    for i, s in enumerate(t["surfaces"]):
        belief_ids = s['beliefs']
        status = "CONNECTED" if s['size'] > 1 else "ISOLATED"

        print(f"  ┌{'─' * 60}┐")
        print(f"  │ Surface {i+1} ({status})".ljust(62) + "│")
        print(f"  └{'─' * 60}┘")

        for bid in belief_ids:
            if bid < len(t["nodes"]):
                n = t["nodes"][bid]
                conf_sym = {"confirmed": "●●●", "likely": "●●○", "reported": "●○○"}[n['confidence']]
                plaus = 1.0 - n['entropy']

                print(f"       ┌──────────────────────────────────────────────┐")
                print(f"       │ B{bid} {conf_sym} P={plaus:.0%}".ljust(49) + "│")
                print(f"       │ {n['text'][:44]}".ljust(49) + "│")
                print(f"       │ Sources: {', '.join(n['sources'][:3])}".ljust(49) + "│")
                print(f"       └──────────────────────────────────────────────┘")

                # Show edges from this node
                for e in t["edges"]:
                    if e['source'] == bid:
                        w = e.get('weight', 0)
                        arrow = "═══" if w >= 0.5 else "───"
                        print(f"              │")
                        print(f"              │ {e['relation']} (w={w:.0%})")
                        print(f"              ▼")
        print()

    # 5. Summary
    print("\n┌" + "─" * 68 + "┐")
    print("│ 5. PLAUSIBILITY SUMMARY                                            │")
    print("└" + "─" * 68 + "┘")

    confirmed = [n for n in t["nodes"] if n['confidence'] == 'confirmed']
    likely = [n for n in t["nodes"] if n['confidence'] == 'likely']
    reported = [n for n in t["nodes"] if n['confidence'] == 'reported']

    print(f"""
  Confidence Distribution:
    ●●● CONFIRMED (3+ src, H≤0.35): {len(confirmed)} beliefs
    ●●○ LIKELY    (2 src, H=0.50):  {len(likely)} beliefs
    ●○○ REPORTED  (1 src, H=0.80):  {len(reported)} beliefs

  Topology Stats:
    Total Beliefs: {len(t["nodes"])}
    Total Edges:   {len(t["edges"])}
    Connected Surfaces: {t["stats"]["connected_surfaces"]}
    Isolated Beliefs:   {t["stats"]["isolated_beliefs"]}
""")

    if t["edges"]:
        weights = [e.get('weight', 0) for e in t["edges"]]
        print(f"""  Edge Weight Distribution:
    Min: {min(weights):.2f}
    Max: {max(weights):.2f}
    Avg: {sum(weights)/len(weights):.2f}
""")

    # Jaynes check
    single_ent = [n['entropy'] for n in t["nodes"] if n['source_count'] == 1]
    multi_ent = [n['entropy'] for n in t["nodes"] if n['source_count'] >= 2]

    if single_ent and multi_ent:
        single_avg = sum(single_ent) / len(single_ent)
        multi_avg = sum(multi_ent) / len(multi_ent)
        jaynes_pass = single_avg > multi_avg

        print(f"""  Jaynes Maximum Entropy Check:
    Single-source avg entropy: {single_avg:.2f}
    Multi-source avg entropy:  {multi_avg:.2f}
    {"✓ PASS" if jaynes_pass else "✗ FAIL"}: {"Single > Multi (correct uncertainty)" if jaynes_pass else "Entropy not aligned"}
""")


if __name__ == "__main__":
    asyncio.run(analyze())
