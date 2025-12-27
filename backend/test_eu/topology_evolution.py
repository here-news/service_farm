"""Watch the topology evolve as claims are fed in."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_eu.core.kernel import EpistemicKernel
from openai import AsyncOpenAI


def print_mini_topology(k, claim_num, claim_text, source, relation):
    """Print a compact topology snapshot after each claim."""
    t = k.topology()

    # Header
    print(f"\n{'─' * 70}")
    print(f"CLAIM {claim_num}: \"{claim_text[:45]}...\" ({source})")
    print(f"Result: {relation.upper()}")
    print(f"{'─' * 70}")

    # Stats bar
    n_beliefs = len(t['nodes'])
    n_edges = len(t['edges'])
    n_surfaces = len(t['surfaces'])
    connected = t['stats']['connected_surfaces']

    print(f"\n  Beliefs: {n_beliefs}  |  Edges: {n_edges}  |  Surfaces: {n_surfaces} ({connected} connected)")

    # Compact belief list with plausibility bars
    print(f"\n  {'ID':>3} {'Plaus':>8} {'Conf':>10} {'Src':>4}  Text")
    print(f"  {'─' * 60}")

    for n in t['nodes']:
        plaus = 1.0 - n['entropy']
        bar = "█" * int(plaus * 5) + "░" * (5 - int(plaus * 5))
        conf_sym = {"confirmed": "★★★", "likely": "★★○", "reported": "★○○"}[n['confidence']]
        print(f"  B{n['id']:>2} [{bar}] {conf_sym:>10} {n['source_count']:>4}  {n['text'][:35]}")

    # Show edges if any
    if t['edges']:
        print(f"\n  Connections:")
        for e in t['edges']:
            shared = e.get('shared_sources', [])
            print(f"    B{e['source']}──({e['relation']})──B{e['target']} via {shared}")

    # Show surfaces
    print(f"\n  Surfaces:")
    for i, s in enumerate(t['surfaces']):
        beliefs = ', '.join(f"B{b}" for b in s['beliefs'])
        status = "●" if s['size'] > 1 else "○"
        print(f"    {status} S{i+1}: [{beliefs}] ({s['total_sources']} sources)")


async def evolve():
    llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    k = EpistemicKernel(llm_client=llm)

    claims = [
        # Fire event - watch death toll converge
        ("Fire kills 11 in Tai Po", "BBC"),
        ("Blaze at Wang Fuk Court leaves 11 dead", "Reuters"),
        ("Death toll rises to 36 as rescue continues", "SCMP"),
        ("36 confirmed dead in Hong Kong fire", "HK Gov"),

        # Fire event - other aspects
        ("Fire broke out on 14th floor of Wang Fuk Court", "BBC"),
        ("Blaze started around 3am on 14th floor", "Reuters"),
        ("Over 100 firefighters responded to the scene", "SCMP"),

        # Separate event - watch it stay isolated
        ("Jimmy Lai's trial continues in Hong Kong", "NYT"),
        ("Lai denies sedition charges", "Guardian"),
        ("Trial enters fourth week", "AP"),
    ]

    print("=" * 70)
    print("TOPOLOGY EVOLUTION - Watch beliefs form and connect")
    print("=" * 70)
    print("\nLegend:")
    print("  Plausibility: [█████] = high (multi-source), [█░░░░] = low (single-source)")
    print("  Confidence:   ★★★ = confirmed, ★★○ = likely, ★○○ = reported")
    print("  Surfaces:     ● = connected, ○ = isolated")

    for i, (text, src) in enumerate(claims, 1):
        result = await k.process(text, src)
        rel = result['relation']
        print_mini_topology(k, i, text, src, rel)

        # Pause to let user see evolution
        print("\n  [Press Enter for next claim...]", end="")
        # In non-interactive mode, just continue
        await asyncio.sleep(0.1)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TOPOLOGY")
    print("=" * 70)

    t = k.topology()

    print(f"\n  Input:  {len(claims)} claims")
    print(f"  Output: {len(t['nodes'])} beliefs ({len(claims)/len(t['nodes']):.1f}x compression)")
    print(f"  Edges:  {len(t['edges'])}")
    print(f"  Surfaces: {len(t['surfaces'])} ({t['stats']['connected_surfaces']} connected, {t['stats']['isolated_beliefs']} isolated)")

    # Jaynes check
    single_ent = [n['entropy'] for n in t['nodes'] if n['source_count'] == 1]
    multi_ent = [n['entropy'] for n in t['nodes'] if n['source_count'] >= 2]

    if single_ent and multi_ent:
        print(f"\n  Jaynes Check:")
        print(f"    Single-source entropy: {sum(single_ent)/len(single_ent):.2f}")
        print(f"    Multi-source entropy:  {sum(multi_ent)/len(multi_ent):.2f}")
        print(f"    {'✓ PASS' if sum(single_ent)/len(single_ent) > sum(multi_ent)/len(multi_ent) else '✗ FAIL'}")


if __name__ == "__main__":
    asyncio.run(evolve())
