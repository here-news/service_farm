"""Test the topology method of the unified kernel."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_eu.core.kernel import EpistemicKernel
from openai import AsyncOpenAI


async def test():
    llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    k = EpistemicKernel(llm_client=llm)

    # Realistic scenario: sources report multiple aspects of same event
    # This should create source overlap → edges → surfaces
    claims = [
        # Fire event claims - sources report multiple aspects
        ("Fire kills 11 in Tai Po", "BBC"),
        ("Blaze at Wang Fuk Court leaves 11 dead", "Reuters"),
        ("Death toll rises to 36 as rescue continues", "SCMP"),
        ("36 confirmed dead in Hong Kong fire", "HK Gov"),
        ("Fire broke out on 14th floor of Wang Fuk Court", "BBC"),
        ("Blaze started around 3am on 14th floor", "Reuters"),
        ("Over 100 firefighters responded to the scene", "SCMP"),

        # Separate event (Jimmy Lai trial) - different sources
        ("Jimmy Lai's trial continues in Hong Kong", "NYT"),
        ("Lai denies sedition charges", "Guardian"),
        ("Trial enters fourth week", "AP"),
    ]

    print("Processing claims...")
    for text, src in claims:
        result = await k.process(text, src)
        rel = result['relation'].upper()
        sym = {'NOVEL': '+', 'CONFIRMS': '=', 'REFINES': '↑',
               'SUPERSEDES': '→', 'CONFLICTS': '!'}[rel]
        print(f"  [{sym}] {rel:10s} | {src:10s} | {text[:40]}")

    print(f"\n{len(claims)} claims -> {len(k.beliefs)} beliefs\n")

    t = k.topology()

    print("=" * 60)
    print("TOPOLOGY")
    print("=" * 60)

    print("\nStats:", t["stats"])

    print("\nNodes (beliefs):")
    for n in t["nodes"]:
        srcs = ", ".join(n['sources'])
        conf = n['confidence'].upper()
        print(f"  B{n['id']}: [{conf}, {n['source_count']} src] {n['text'][:45]}")

    print("\nEdges (connections):")
    if t["edges"]:
        for e in t["edges"]:
            extra = f" via {e.get('shared_sources', [])}" if 'shared_sources' in e else ""
            print(f"  B{e['source']} --{e['relation']}--> B{e['target']}{extra}")
    else:
        print("  (no edges)")

    print("\nSurfaces (event clusters):")
    for i, s in enumerate(t["surfaces"]):
        belief_ids = ", ".join(f"B{b}" for b in s['beliefs'])
        status = "CONNECTED" if s['size'] > 1 else "ISOLATED"
        print(f"  Surface {i+1} [{status}]: {belief_ids} ({s['total_sources']} unique sources)")
        for bid in s['beliefs']:
            if bid < len(k.beliefs):
                b = k.beliefs[bid]
                src_count = len(b.sources)
                print(f"    - [{src_count} src] {b.text[:50]}")

    # Save topology to JSON for visualization
    output_path = os.path.join(os.path.dirname(__file__), "results", "topology_demo.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "input": {
            "claims": [{"text": c[0], "source": c[1]} for c in claims],
            "claim_count": len(claims)
        },
        "output": {
            "belief_count": len(k.beliefs),
            "compression_ratio": len(claims) / len(k.beliefs)
        },
        "topology": t,
        "summary": k.summary()
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nTopology saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(test())
