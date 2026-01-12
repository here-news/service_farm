# Set up path for container environment - MUST be before any local imports
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

"""
Tier-B Ablation: Compiler vs Heuristic Baseline

Compares three approaches on WFC + distractors subset:
1. Heuristic overlap (baseline)
2. Artifacts-only compiler (new approach)
3. Old DF-filtered approach (for comparison)

Metrics tracked:
- WFC coverage: what fraction of true WFC incidents are in the WFC case
- WFC purity: what fraction of the WFC case are actually WFC incidents
- Largest case size
- Total cases formed
- Spine/metabolic edge counts
- DEFER count
- InquirySeed count
- LLM calls per incident
"""

import os
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Set, Any

import asyncpg
from openai import AsyncOpenAI
from services.neo4j_service import Neo4jService

# Import compiler components
from reee.compiler.membrane import (
    Action,
    EdgeType,
    ReferentRole,
    Referent,
    IncidentArtifact,
    MembraneDecision,
    CompilerParams,
    DEFAULT_PARAMS,
    compile_pair,
    assert_invariants,
)
from reee.compiler.artifacts.extractor import (
    ReferentType,
    extract_artifact,
    ExtractionResult,
)
from reee.compiler.weaver_compiler import UnionFind


@dataclass
class AblationMetrics:
    """Metrics for a single ablation run."""
    name: str
    total_cases: int
    largest_case: int
    spine_edges: int
    metabolic_edges: int
    deferred: int
    inquiry_seeds: int
    wfc_coverage: float
    wfc_purity: float
    wfc_case_size: int
    contaminants: int
    llm_calls: int


async def load_tier_b_dataset(neo4j, n_distractors: int = 100):
    """Load WFC incidents + distractors from Neo4j."""
    # Get all WFC incidents
    wfc_query = """
    MATCH (i:Incident)
    WHERE 'Wang Fuk Court' IN i.anchor_entities
    RETURN i.id as id, i.title as title, i.anchor_entities as anchors,
           i.time_start as time_start
    ORDER BY i.id
    """
    wfc_results = await neo4j._execute_read(wfc_query)

    # Get non-WFC distractors
    distractor_query = f"""
    MATCH (i:Incident)
    WHERE NOT 'Wang Fuk Court' IN i.anchor_entities
    WITH i ORDER BY i.id LIMIT {n_distractors}
    RETURN i.id as id, i.title as title, i.anchor_entities as anchors,
           i.time_start as time_start
    """
    distractor_results = await neo4j._execute_read(distractor_query)

    # Combine into incidents dict
    incidents = {}
    wfc_ids = set()

    for r in wfc_results:
        incidents[r['id']] = {
            'id': r['id'],
            'title': r['title'] or '',
            'anchor_entities': set(r['anchors'] or []),
            'time_start': r['time_start'],
        }
        wfc_ids.add(r['id'])

    for r in distractor_results:
        incidents[r['id']] = {
            'id': r['id'],
            'title': r['title'] or '',
            'anchor_entities': set(r['anchors'] or []),
            'time_start': r['time_start'],
        }

    return incidents, wfc_ids


async def run_heuristic_baseline(incidents: Dict, wfc_ground_truth: Set[str]) -> AblationMetrics:
    """Ablation 1: Heuristic overlap baseline (no LLM)."""
    # Generate edges from any entity overlap
    edges = []
    incident_list = list(incidents.keys())

    for i, id1 in enumerate(incident_list):
        for id2 in incident_list[i + 1:]:
            ents1 = incidents[id1]['anchor_entities']
            ents2 = incidents[id2]['anchor_entities']
            overlap = ents1 & ents2
            if overlap:
                edges.append((id1, id2, overlap))

    # Union-find for case formation
    uf = UnionFind(set(incidents.keys()))
    for id1, id2, _ in edges:
        uf.union(id1, id2)

    # Get components
    components = uf.get_components()
    cases = [c for c in components.values() if len(c) >= 2]

    # Find WFC case
    wfc_case = None
    for c in sorted(cases, key=len, reverse=True):
        if c & wfc_ground_truth:
            wfc_case = c
            break

    # Compute metrics
    wfc_case = wfc_case or set()
    wfc_in_case = len(wfc_case & wfc_ground_truth)
    coverage = wfc_in_case / len(wfc_ground_truth) if wfc_ground_truth else 0
    purity = wfc_in_case / len(wfc_case) if wfc_case else 0

    return AblationMetrics(
        name="A1: Heuristic Overlap",
        total_cases=len(cases),
        largest_case=max((len(c) for c in cases), default=0),
        spine_edges=len(edges),
        metabolic_edges=0,
        deferred=0,
        inquiry_seeds=0,
        wfc_coverage=coverage,
        wfc_purity=purity,
        wfc_case_size=len(wfc_case),
        contaminants=len(wfc_case - wfc_ground_truth),
        llm_calls=0,
    )


async def run_compiler_ablation(
    incidents: Dict,
    wfc_ground_truth: Set[str],
    llm_client: Any,
    params: CompilerParams = DEFAULT_PARAMS,
) -> AblationMetrics:
    """Ablation 2: Artifacts-only compiler."""
    # Step 1: Extract artifacts for all incidents
    print("    Extracting artifacts...")
    artifacts: Dict[str, ExtractionResult] = {}
    llm_calls = 0

    for iid, inc in incidents.items():
        result = await extract_artifact(
            incident_id=iid,
            title=inc['title'],
            anchor_entities=inc['anchor_entities'],
            entity_lookup={},
            llm_client=llm_client,
        )
        artifacts[iid] = result
        llm_calls += 1

        if llm_calls % 20 == 0:
            print(f"      {llm_calls}/{len(incidents)} artifacts extracted")

    print(f"    {len(artifacts)} artifacts extracted")

    # Step 2: Generate candidates (referent overlap)
    print("    Generating candidates...")
    candidates = []
    incident_ids = list(artifacts.keys())

    for i, id1 in enumerate(incident_ids):
        for id2 in incident_ids[i + 1:]:
            art1 = artifacts[id1].artifact
            art2 = artifacts[id2].artifact

            refs1 = {r.entity_id for r in art1.referents}
            refs2 = {r.entity_id for r in art2.referents}

            if refs1 & refs2:
                candidates.append((id1, id2))
            elif art1.contexts & art2.contexts:
                candidates.append((id1, id2))

    print(f"    {len(candidates)} candidates")

    # Step 3: Compile each pair through membrane
    print("    Compiling pairs...")
    spine_edges = []
    metabolic_edges = []
    deferred = []
    all_inquiries = set()

    for id1, id2 in candidates:
        art1 = artifacts[id1].artifact
        art2 = artifacts[id2].artifact

        decision = compile_pair(art1, art2, params)
        assert_invariants(decision)

        if decision.action == Action.MERGE:
            spine_edges.append((id1, id2, decision))
        elif decision.action == Action.PERIPHERY:
            metabolic_edges.append((id1, id2, decision))
        elif decision.action == Action.DEFER:
            deferred.append((id1, id2, decision))

    # Collect inquiries
    for result in artifacts.values():
        all_inquiries.update(result.inquiries)

    print(f"    {len(spine_edges)} spine, {len(metabolic_edges)} metabolic, {len(deferred)} deferred")

    # Step 4: Form cases via union-find on spine edges
    uf = UnionFind(set(incidents.keys()))
    for id1, id2, _ in spine_edges:
        uf.union(id1, id2)

    components = uf.get_components()
    cases = [c for c in components.values() if len(c) >= 2]

    # Find WFC case
    wfc_case = None
    for c in sorted(cases, key=len, reverse=True):
        # Check if any incident in case has WFC referent
        has_wfc = False
        for iid in c:
            art = artifacts[iid].artifact
            for ref in art.referents:
                if 'wang_fuk_court' in ref.entity_id.lower():
                    has_wfc = True
                    break
            if has_wfc:
                break
        if has_wfc:
            wfc_case = c
            break

    # Fallback: check by original anchor entities
    if not wfc_case:
        for c in sorted(cases, key=len, reverse=True):
            if c & wfc_ground_truth:
                wfc_case = c
                break

    # Compute metrics
    wfc_case = wfc_case or set()
    wfc_in_case = len(wfc_case & wfc_ground_truth)
    coverage = wfc_in_case / len(wfc_ground_truth) if wfc_ground_truth else 0
    purity = wfc_in_case / len(wfc_case) if wfc_case else 0

    return AblationMetrics(
        name="A2: Compiler (artifacts-only)",
        total_cases=len(cases),
        largest_case=max((len(c) for c in cases), default=0),
        spine_edges=len(spine_edges),
        metabolic_edges=len(metabolic_edges),
        deferred=len(deferred),
        inquiry_seeds=len(all_inquiries),
        wfc_coverage=coverage,
        wfc_purity=purity,
        wfc_case_size=len(wfc_case),
        contaminants=len(wfc_case - wfc_ground_truth),
        llm_calls=llm_calls,
    )


def print_results(metrics_list: list[AblationMetrics], wfc_count: int, distractor_count: int):
    """Print comparison table."""
    print()
    print("=" * 80)
    print("TIER-B ABLATION RESULTS")
    print("=" * 80)
    print()
    print(f"Dataset: {wfc_count} WFC + {distractor_count} distractors = {wfc_count + distractor_count} total")
    print()
    print("| Ablation | Cases | Largest | Spine | Metabolic | DEFER | Seeds | WFC Cov | WFC Pur | Contam | LLM |")
    print("|----------|-------|---------|-------|-----------|-------|-------|---------|---------|--------|-----|")

    for m in metrics_list:
        print(f"| {m.name[:15]:<15} | {m.total_cases:>5} | {m.largest_case:>7} | {m.spine_edges:>5} | "
              f"{m.metabolic_edges:>9} | {m.deferred:>5} | {m.inquiry_seeds:>5} | {m.wfc_coverage:>6.1%} | "
              f"{m.wfc_purity:>6.1%} | {m.contaminants:>6} | {m.llm_calls:>3} |")

    print()

    # Highlight improvements
    if len(metrics_list) >= 2:
        baseline = metrics_list[0]
        for m in metrics_list[1:]:
            if m.wfc_purity > baseline.wfc_purity:
                print(f"PURITY IMPROVEMENT ({m.name}): {baseline.wfc_purity:.1%} -> {m.wfc_purity:.1%} "
                      f"(+{(m.wfc_purity - baseline.wfc_purity) * 100:.1f}pp)")
            if m.wfc_coverage > baseline.wfc_coverage:
                print(f"COVERAGE IMPROVEMENT ({m.name}): {baseline.wfc_coverage:.1%} -> {m.wfc_coverage:.1%} "
                      f"(+{(m.wfc_coverage - baseline.wfc_coverage) * 100:.1f}pp)")


async def main():
    timestamp = datetime.utcnow().isoformat()
    print("=" * 80)
    print("TIER-B ABLATION: Compiler vs Heuristic Baseline")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print()

    # Connect to services
    neo4j = Neo4jService()
    await neo4j.connect()

    llm_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        # Load dataset
        print("Loading Tier-B dataset...")
        incidents, wfc_ground_truth = await load_tier_b_dataset(neo4j, n_distractors=100)
        print(f"  {len(wfc_ground_truth)} WFC incidents")
        print(f"  {len(incidents) - len(wfc_ground_truth)} distractor incidents")
        print(f"  {len(incidents)} total incidents")
        print()

        results = []

        # Ablation 1: Heuristic baseline
        print("Running A1: Heuristic Overlap (baseline)...")
        m1 = await run_heuristic_baseline(incidents, wfc_ground_truth)
        results.append(m1)
        print(f"  WFC coverage: {m1.wfc_coverage:.1%}, purity: {m1.wfc_purity:.1%}")
        print()

        # Ablation 2: Compiler
        print("Running A2: Compiler (artifacts-only)...")
        m2 = await run_compiler_ablation(incidents, wfc_ground_truth, llm_client)
        results.append(m2)
        print(f"  WFC coverage: {m2.wfc_coverage:.1%}, purity: {m2.wfc_purity:.1%}")
        print()

        # Print comparison
        print_results(results, len(wfc_ground_truth), len(incidents) - len(wfc_ground_truth))

        # Save results
        output = {
            'timestamp': timestamp,
            'dataset': {
                'wfc_count': len(wfc_ground_truth),
                'distractor_count': len(incidents) - len(wfc_ground_truth),
                'total': len(incidents),
            },
            'results': [asdict(m) for m in results],
        }
        print()
        print("JSON output:")
        print(json.dumps(output, indent=2))

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
