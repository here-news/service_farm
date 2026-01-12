"""
Weaver Compiler: Orchestrates artifact extraction + membrane compilation.

This is the central compiler that:
1. Extracts artifacts from incidents (via artifacts/extractor)
2. Generates candidate pairs for comparison (embeddings for recall)
3. Compiles each pair through the membrane (deterministic decisions)
4. Forms cases via union-find on spine edges

The compiler NEVER makes topology decisions - only the membrane does.
Embeddings are used ONLY for recall (candidate generation), not identity.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Set, Dict, List, FrozenSet
import hashlib

from .membrane import (
    Action,
    EdgeType,
    MembraneDecision,
    CompilerParams,
    DEFAULT_PARAMS,
    compile_pair,
    assert_invariants,
)
from .artifacts.extractor import (
    extract_artifact,
    extract_artifacts_batch,
    ExtractionResult,
    InquirySeed,
)


# =============================================================================
# Compilation Output Schema
# =============================================================================

@dataclass
class CompiledEdge:
    """A single compiled edge between two incidents."""
    incident_a: str
    incident_b: str
    decision: MembraneDecision
    artifact_hash_a: str  # For reproducibility
    artifact_hash_b: str


@dataclass
class Case:
    """A case formed from spine edges."""
    case_id: str
    incident_ids: FrozenSet[str]
    spine_edges: List[CompiledEdge]
    metabolic_edges: List[CompiledEdge]


@dataclass
class CompilationResult:
    """Complete compilation result."""
    cases: List[Case]
    all_edges: List[CompiledEdge]
    deferred: List[CompiledEdge]  # DEFER edges need attention
    inquiries: List[InquirySeed]  # Human disambiguation needed
    stats: Dict[str, Any]


# =============================================================================
# Union-Find for Case Formation
# =============================================================================

class UnionFind:
    """Union-find data structure for case formation from spine edges."""

    def __init__(self, elements: Set[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """Union two elements. Returns True if they were in different sets."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_components(self) -> Dict[str, Set[str]]:
        """Get all connected components."""
        components = {}
        for x in self.parent:
            root = self.find(x)
            if root not in components:
                components[root] = set()
            components[root].add(x)
        return components


# =============================================================================
# Candidate Generation (Embeddings for Recall)
# =============================================================================

def generate_candidates_by_referent_overlap(
    artifacts: Dict[str, ExtractionResult],
) -> List[tuple[str, str]]:
    """
    Generate candidate pairs based on referent overlap.
    This is a simple O(n^2) approach for small datasets.
    For production, use embedding similarity for recall.
    """
    incident_ids = list(artifacts.keys())
    candidates = []

    for i, id1 in enumerate(incident_ids):
        for id2 in incident_ids[i + 1:]:
            art1 = artifacts[id1].artifact
            art2 = artifacts[id2].artifact

            # Check referent overlap
            refs1 = {r.entity_id for r in art1.referents}
            refs2 = {r.entity_id for r in art2.referents}
            if refs1 & refs2:
                candidates.append((id1, id2))
            # Also check context overlap for potential CONTEXT_FOR edges
            elif art1.contexts & art2.contexts:
                candidates.append((id1, id2))

    return candidates


async def generate_candidates_by_embedding(
    artifacts: Dict[str, ExtractionResult],
    embedding_client: Any,
    threshold: float = 0.7,
    model: str = "text-embedding-3-small",
) -> List[tuple[str, str]]:
    """
    Generate candidate pairs using embedding similarity.
    Uses embeddings ONLY for recall - never for identity decisions.

    Args:
        artifacts: Mapping of incident_id -> ExtractionResult
        embedding_client: OpenAI-compatible client
        threshold: Similarity threshold for candidate inclusion
        model: Embedding model to use

    Returns:
        List of (incident_a, incident_b) candidate pairs
    """
    # This is a placeholder - would integrate with actual embedding service
    # For now, fall back to referent overlap
    return generate_candidates_by_referent_overlap(artifacts)


# =============================================================================
# Main Compiler
# =============================================================================

async def compile_incidents(
    incidents: List[Dict],  # [{id, title, anchor_entities, time_start}, ...]
    entity_lookup: Dict[str, Dict],
    llm_client: Any,
    params: CompilerParams = DEFAULT_PARAMS,
    embedding_client: Any = None,
    embedding_threshold: float = 0.7,
) -> CompilationResult:
    """
    Compile a set of incidents into cases.

    This is the main entry point. It:
    1. Extracts artifacts from all incidents
    2. Generates candidate pairs
    3. Compiles each pair through the membrane
    4. Forms cases via union-find on MERGE decisions

    Args:
        incidents: List of incident dicts with id, title, anchor_entities
        entity_lookup: Mapping of entity_id -> metadata
        llm_client: OpenAI-compatible client for artifact extraction
        params: Compiler parameters
        embedding_client: Optional client for embedding-based recall
        embedding_threshold: Threshold for embedding similarity

    Returns:
        CompilationResult with cases, edges, deferred, inquiries, and stats
    """
    # Step 1: Extract artifacts
    artifacts = await extract_artifacts_batch(
        incidents=incidents,
        entity_lookup=entity_lookup,
        llm_client=llm_client,
    )

    # Step 2: Generate candidates
    if embedding_client:
        candidates = await generate_candidates_by_embedding(
            artifacts=artifacts,
            embedding_client=embedding_client,
            threshold=embedding_threshold,
        )
    else:
        candidates = generate_candidates_by_referent_overlap(artifacts)

    # Step 3: Compile each pair
    all_edges = []
    spine_edges = []
    metabolic_edges = []
    deferred = []

    for id1, id2 in candidates:
        if id1 not in artifacts or id2 not in artifacts:
            continue

        art1 = artifacts[id1].artifact
        art2 = artifacts[id2].artifact

        # Compile through membrane
        decision = compile_pair(art1, art2, params)

        # Validate invariants
        assert_invariants(decision)

        edge = CompiledEdge(
            incident_a=id1,
            incident_b=id2,
            decision=decision,
            artifact_hash_a=artifacts[id1].prompt_hash,
            artifact_hash_b=artifacts[id2].prompt_hash,
        )
        all_edges.append(edge)

        # Route by action
        if decision.action == Action.MERGE:
            spine_edges.append(edge)
        elif decision.action == Action.PERIPHERY:
            metabolic_edges.append(edge)
        elif decision.action == Action.DEFER:
            deferred.append(edge)

    # Step 4: Form cases via union-find on spine edges
    incident_ids = set(artifacts.keys())
    uf = UnionFind(incident_ids)

    for edge in spine_edges:
        uf.union(edge.incident_a, edge.incident_b)

    # Build case objects
    components = uf.get_components()
    cases = []
    for root, members in components.items():
        if len(members) < 2:
            continue  # Singleton, not a case

        # Gather edges for this case
        case_spine = [
            e for e in spine_edges
            if e.incident_a in members or e.incident_b in members
        ]
        case_metabolic = [
            e for e in metabolic_edges
            if e.incident_a in members or e.incident_b in members
        ]

        case = Case(
            case_id=f"case_{hashlib.sha256(root.encode()).hexdigest()[:12]}",
            incident_ids=frozenset(members),
            spine_edges=case_spine,
            metabolic_edges=case_metabolic,
        )
        cases.append(case)

    # Collect all inquiries
    all_inquiries = []
    for result in artifacts.values():
        all_inquiries.extend(result.inquiries)

    # Compute stats
    stats = {
        "total_incidents": len(incidents),
        "artifacts_extracted": len(artifacts),
        "candidates_generated": len(candidates),
        "spine_edges": len(spine_edges),
        "metabolic_edges": len(metabolic_edges),
        "deferred_edges": len(deferred),
        "cases_formed": len(cases),
        "inquiries_pending": len(all_inquiries),
        "largest_case": max((len(c.incident_ids) for c in cases), default=0),
    }

    return CompilationResult(
        cases=cases,
        all_edges=all_edges,
        deferred=deferred,
        inquiries=list(set(all_inquiries)),
        stats=stats,
    )


# =============================================================================
# Incremental Compilation (for new incidents)
# =============================================================================

async def compile_incremental(
    new_incidents: List[Dict],
    existing_artifacts: Dict[str, ExtractionResult],
    existing_cases: List[Case],
    entity_lookup: Dict[str, Dict],
    llm_client: Any,
    params: CompilerParams = DEFAULT_PARAMS,
) -> CompilationResult:
    """
    Incrementally compile new incidents against existing topology.

    This is used when new incidents arrive - we only need to:
    1. Extract artifacts for new incidents
    2. Compare new incidents against existing ones
    3. Update case membership

    Args:
        new_incidents: New incidents to add
        existing_artifacts: Already-extracted artifacts
        existing_cases: Current case structure
        entity_lookup: Entity metadata
        llm_client: LLM client
        params: Compiler parameters

    Returns:
        Updated CompilationResult
    """
    # Extract artifacts for new incidents only
    new_artifacts = await extract_artifacts_batch(
        incidents=new_incidents,
        entity_lookup=entity_lookup,
        llm_client=llm_client,
    )

    # Merge with existing
    all_artifacts = {**existing_artifacts, **new_artifacts}

    # Generate candidates: new × (new + existing)
    new_ids = set(new_artifacts.keys())
    existing_ids = set(existing_artifacts.keys())

    candidates = []
    for new_id in new_ids:
        # New vs new
        for other_new in new_ids:
            if new_id < other_new:  # Avoid duplicates
                candidates.append((new_id, other_new))
        # New vs existing
        for existing_id in existing_ids:
            candidates.append((new_id, existing_id))

    # Filter to those with potential overlap
    filtered_candidates = []
    for id1, id2 in candidates:
        if id1 not in all_artifacts or id2 not in all_artifacts:
            continue
        art1 = all_artifacts[id1].artifact
        art2 = all_artifacts[id2].artifact
        refs1 = {r.entity_id for r in art1.referents}
        refs2 = {r.entity_id for r in art2.referents}
        if refs1 & refs2 or art1.contexts & art2.contexts:
            filtered_candidates.append((id1, id2))

    # Compile each pair
    all_edges = []
    spine_edges = []
    metabolic_edges = []
    deferred = []

    for id1, id2 in filtered_candidates:
        art1 = all_artifacts[id1].artifact
        art2 = all_artifacts[id2].artifact

        decision = compile_pair(art1, art2, params)
        assert_invariants(decision)

        edge = CompiledEdge(
            incident_a=id1,
            incident_b=id2,
            decision=decision,
            artifact_hash_a=all_artifacts[id1].prompt_hash,
            artifact_hash_b=all_artifacts[id2].prompt_hash,
        )
        all_edges.append(edge)

        if decision.action == Action.MERGE:
            spine_edges.append(edge)
        elif decision.action == Action.PERIPHERY:
            metabolic_edges.append(edge)
        elif decision.action == Action.DEFER:
            deferred.append(edge)

    # Rebuild cases with all spine edges
    all_incident_ids = set(all_artifacts.keys())
    uf = UnionFind(all_incident_ids)

    # Add existing spine edges
    for case in existing_cases:
        for edge in case.spine_edges:
            uf.union(edge.incident_a, edge.incident_b)

    # Add new spine edges
    for edge in spine_edges:
        uf.union(edge.incident_a, edge.incident_b)

    # Build updated cases
    components = uf.get_components()
    cases = []
    for root, members in components.items():
        if len(members) < 2:
            continue

        case = Case(
            case_id=f"case_{hashlib.sha256(root.encode()).hexdigest()[:12]}",
            incident_ids=frozenset(members),
            spine_edges=[],  # Would need to gather all spine edges
            metabolic_edges=[],
        )
        cases.append(case)

    # Collect inquiries
    all_inquiries = []
    for result in new_artifacts.values():
        all_inquiries.extend(result.inquiries)

    stats = {
        "new_incidents": len(new_incidents),
        "new_artifacts": len(new_artifacts),
        "candidates_checked": len(filtered_candidates),
        "new_spine_edges": len(spine_edges),
        "new_metabolic_edges": len(metabolic_edges),
        "deferred_edges": len(deferred),
        "cases_after": len(cases),
        "inquiries_pending": len(all_inquiries),
    }

    return CompilationResult(
        cases=cases,
        all_edges=all_edges,
        deferred=deferred,
        inquiries=list(set(all_inquiries)),
        stats=stats,
    )
