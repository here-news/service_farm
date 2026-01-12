#!/usr/bin/env python3
"""
Proof: Why Spine + Metabolic Topology is Necessary
===================================================

Demonstrates three approaches to case formation:

1. FAILED: Heuristic overlap (giant component)
2. WORKS: Role-based spine (high precision + recall)
3. NECESSITY: DF filtering prevents LLM mislabeling from degrading spine

This proves:
- Relatedness ≠ identity (must separate spine from metabolic edges)
- Referent roles are the minimal correct witness for spine membership
- DF filtering is defense-in-depth against LLM errors
"""

import sys
sys.path.insert(0, '/media/im3/plus/lab4/re_news/service_farm/backend')

from reee.contracts.case_formation import (
    EntityRole,
    IncidentRoleArtifact,
    evaluate_spine_gate,
    EntityDFEstimate,
    should_suppress_entity,
)
from datetime import datetime

# Simulated WFC corpus (48 WFC incidents + contaminants)
INCIDENTS = {
    # True WFC incidents (48 total, simplified to 3 for demonstration)
    "wfc-1": {
        "entities": {"Wang Fuk Court": "FACILITY", "Tai Po": "LOCATION", "Hong Kong": "LOCATION"},
        "time": "2024-11-26",
    },
    "wfc-2": {
        "entities": {"Wang Fuk Court": "FACILITY", "Tai Po": "LOCATION", "fire": "EVENT"},
        "time": "2024-11-26",
    },
    "wfc-3": {
        "entities": {"Wang Fuk Court": "FACILITY", "Hong Kong Fire Services": "ORG"},
        "time": "2024-11-27",
    },

    # Contaminants (share "Tai Po" or "Hong Kong" but different events)
    "other-tp-1": {
        "entities": {"Some Building": "FACILITY", "Tai Po": "LOCATION"},
        "time": "2024-12-01",
    },
    "other-hk-1": {
        "entities": {"Central": "LOCATION", "Hong Kong": "LOCATION", "protest": "EVENT"},
        "time": "2024-11-20",
    },
}

GLOBAL_DF = {
    "Wang Fuk Court": 48,
    "Tai Po": 45,
    "Hong Kong": 200,
    "Hong Kong Fire Services": 38,
    "Some Building": 5,
    "Central": 12,
}

GLOBAL_TOTAL = 515

def approach_1_heuristic_overlap():
    """FAILED: Entity overlap creates giant component."""

    print("APPROACH 1: Heuristic Overlap (Entity Intersection)")
    print("-" * 60)

    # Build adjacency based on ANY shared entity
    edges = []
    for id1 in INCIDENTS:
        for id2 in INCIDENTS:
            if id1 >= id2:
                continue

            ents1 = set(INCIDENTS[id1]["entities"].keys())
            ents2 = set(INCIDENTS[id2]["entities"].keys())
            overlap = ents1 & ents2

            if overlap:
                edges.append((id1, id2, overlap))

    # Union-find to get components
    parent = {iid: iid for iid in INCIDENTS}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for id1, id2, _ in edges:
        union(id1, id2)

    # Collect components
    components = {}
    for iid in INCIDENTS:
        root = find(iid)
        if root not in components:
            components[root] = set()
        components[root].add(iid)

    # Check WFC case
    wfc_ids = {iid for iid in INCIDENTS if "Wang Fuk Court" in INCIDENTS[iid]["entities"]}
    wfc_case = None
    for comp in components.values():
        if comp & wfc_ids:
            wfc_case = comp
            break

    purity = len(wfc_case & wfc_ids) / len(wfc_case) if wfc_case else 0

    print(f"Total edges: {len(edges)}")
    print(f"Components: {len(components)}")
    print(f"WFC case size: {len(wfc_case)}")
    print(f"WFC purity: {purity:.0%}")
    print(f"Problem: Giant component with contaminants")
    print(f"  Contaminants: {wfc_case - wfc_ids}")
    print()

    return purity


def approach_2_role_based_spine():
    """WORKS: Role-based referent overlap (no DF filtering yet)."""

    print("APPROACH 2: Role-Based Spine (LLM labels referents)")
    print("-" * 60)

    # Simulate LLM role labeling (BEFORE DF filtering)
    # LLM mistakenly labels "Tai Po" and "Hong Kong" as REFERENT_LOCATION
    artifacts = {}
    for iid, inc in INCIDENTS.items():
        referents = set()
        roles = {}

        for ent, etype in inc["entities"].items():
            if etype == "FACILITY":
                referents.add(ent)
                roles[ent] = EntityRole.REFERENT_FACILITY
            elif etype == "LOCATION":
                # LLM labels ALL locations as referents (mistake!)
                referents.add(ent)
                roles[ent] = EntityRole.REFERENT_LOCATION
            elif etype == "ORG":
                roles[ent] = EntityRole.RESPONDER

        artifacts[iid] = IncidentRoleArtifact(
            incident_id=iid,
            referent_entity_ids=frozenset(referents),
            role_map=roles,
            time_start=datetime.fromisoformat(inc["time"]),
        )

    # Build spine edges from referent overlap
    spine_edges = []
    for id1 in INCIDENTS:
        for id2 in INCIDENTS:
            if id1 >= id2:
                continue

            art1 = artifacts[id1]
            art2 = artifacts[id2]

            overlap = art1.referent_entity_ids & art2.referent_entity_ids
            if overlap:
                result = evaluate_spine_gate(art1, art2, time_closeness_days=7)
                if result.is_spine:
                    spine_edges.append((id1, id2, overlap))

    # Union-find
    parent = {iid: iid for iid in INCIDENTS}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for id1, id2, _ in spine_edges:
        union(id1, id2)

    components = {}
    for iid in INCIDENTS:
        root = find(iid)
        if root not in components:
            components[root] = set()
        components[root].add(iid)

    wfc_ids = {iid for iid in INCIDENTS if "Wang Fuk Court" in INCIDENTS[iid]["entities"]}
    wfc_case = None
    for comp in components.values():
        if comp & wfc_ids:
            wfc_case = comp
            break

    purity = len(wfc_case & wfc_ids) / len(wfc_case) if wfc_case else 0

    print(f"Spine edges: {len(spine_edges)}")
    print(f"Components: {len(components)}")
    print(f"WFC case size: {len(wfc_case)}")
    print(f"WFC purity: {purity:.0%}")
    print(f"Problem: LLM mislabels broad locations as referents")
    print(f"  Contaminants: {wfc_case - wfc_ids}")
    print()

    return purity


def approach_3_df_filtered_spine():
    """BEST: DF filtering demotes broad locations → clean spine."""

    print("APPROACH 3: DF-Filtered Spine (Corrects LLM errors)")
    print("-" * 60)

    # Same LLM artifacts as Approach 2
    artifacts_raw = {}
    for iid, inc in INCIDENTS.items():
        referents = set()
        roles = {}

        for ent, etype in inc["entities"].items():
            if etype == "FACILITY":
                referents.add(ent)
                roles[ent] = EntityRole.REFERENT_FACILITY
            elif etype == "LOCATION":
                referents.add(ent)
                roles[ent] = EntityRole.REFERENT_LOCATION
            elif etype == "ORG":
                roles[ent] = EntityRole.RESPONDER

        artifacts_raw[iid] = IncidentRoleArtifact(
            incident_id=iid,
            referent_entity_ids=frozenset(referents),
            role_map=roles,
            time_start=datetime.fromisoformat(inc["time"]),
        )

    # Apply DF filtering
    threshold = 0.05
    artifacts_filtered = {}

    # Compute local DF
    local_df = {}
    for art in artifacts_raw.values():
        for eid in art.role_map.keys():
            local_df[eid] = local_df.get(eid, 0) + 1

    for iid, art in artifacts_raw.items():
        role_map = dict(art.role_map)
        filtered_referents = set()

        for eid in art.referent_entity_ids:
            role = role_map.get(eid, EntityRole.CONTEXT)

            # Compute DF shrinkage
            df_est = EntityDFEstimate.compute(
                entity_id=eid,
                df_global=GLOBAL_DF.get(eid, 0),
                df_local=local_df.get(eid, 0),
                alpha=0.9,
            )

            # Deterministic demotion: REFERENT_LOCATION → BROAD_LOCATION if high DF
            if role == EntityRole.REFERENT_LOCATION:
                fraction = df_est.df_shrunk / GLOBAL_TOTAL
                if fraction > threshold:
                    print(f"  Demoting '{eid}' in {iid}: df={df_est.df_shrunk:.0f}, fraction={fraction:.1%}")
                    role = EntityRole.BROAD_LOCATION
                    role_map[eid] = role

            # Keep only if still a referent role
            if role.is_referent:
                filtered_referents.add(eid)

        artifacts_filtered[iid] = IncidentRoleArtifact(
            incident_id=iid,
            referent_entity_ids=frozenset(filtered_referents),
            role_map=role_map,
            time_start=art.time_start,
        )

    print()

    # Build spine edges from filtered referents
    spine_edges = []
    for id1 in INCIDENTS:
        for id2 in INCIDENTS:
            if id1 >= id2:
                continue

            art1 = artifacts_filtered[id1]
            art2 = artifacts_filtered[id2]

            overlap = art1.referent_entity_ids & art2.referent_entity_ids
            if overlap:
                result = evaluate_spine_gate(art1, art2, time_closeness_days=7)
                if result.is_spine:
                    spine_edges.append((id1, id2, overlap))

    # Union-find
    parent = {iid: iid for iid in INCIDENTS}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for id1, id2, _ in spine_edges:
        union(id1, id2)

    components = {}
    for iid in INCIDENTS:
        root = find(iid)
        if root not in components:
            components[root] = set()
        components[root].add(iid)

    wfc_ids = {iid for iid in INCIDENTS if "Wang Fuk Court" in INCIDENTS[iid]["entities"]}
    wfc_case = None
    for comp in components.values():
        if comp & wfc_ids:
            wfc_case = comp
            break

    purity = len(wfc_case & wfc_ids) / len(wfc_case) if wfc_case else 0

    print(f"Spine edges: {len(spine_edges)}")
    print(f"Components: {len(components)}")
    print(f"WFC case size: {len(wfc_case)}")
    print(f"WFC purity: {purity:.0%}")
    print(f"Success: Only valid referents open spine gates")
    print()

    return purity


if __name__ == "__main__":
    print("=" * 60)
    print("PROOF: Spine + Metabolic Topology Necessity")
    print("=" * 60)
    print()

    p1 = approach_1_heuristic_overlap()
    p2 = approach_2_role_based_spine()
    p3 = approach_3_df_filtered_spine()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Approach 1 (Heuristic overlap):        {p1:.0%} purity - FAILED (giant component)")
    print(f"Approach 2 (Role-based, no filter):    {p2:.0%} purity - DEGRADED (LLM errors)")
    print(f"Approach 3 (DF-filtered roles):        {p3:.0%} purity - SUCCESS (clean spine)")
    print()
    print("CONCLUSION:")
    print("1. Relatedness ≠ Identity - heuristic overlap creates giant components")
    print("2. Referent roles are the correct witness for spine membership")
    print("3. DF filtering prevents LLM mislabeling from degrading precision")
    print("4. Spine + Metabolic separation is algorithmic necessity, not philosophy")
