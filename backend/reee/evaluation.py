"""
Epistemic Evaluation Harness
============================

Validates the multi-level epistemic architecture with rigorous metrics.

Levels:
- Level 0 (Claims): Confusion matrix, micro/macro F1 for 5 relations
- Level 1+ (Clustering): B³, V-measure, completeness, fragmentation

Coherence:
- Temporal coherence: % pairs violating time gates
- Anchor coherence: anchor overlap + specificity
- Semantic dispersion: embedding variance per cluster

Stability:
- Order shuffle: Jaccard stability across random orders
- Threshold sensitivity: precision/recall tradeoff curve
- Ablations: embeddings only, +entities, +time, +anchor gating
"""

import json
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum

from .epistemic_unit import Claim, Surface, EmergenceEngine, Relation, cosine_similarity


# =============================================================================
# GROUND TRUTH SCHEMA
# =============================================================================

@dataclass
class LabeledClaim:
    """A claim with ground truth labels."""
    claim: Claim
    event_label: str  # Ground truth event
    relation_to: Optional[str] = None  # claim_id this relates to
    relation_type: Optional[str] = None  # CONFIRMS/REFINES/...


@dataclass
class GroundTruth:
    """Complete ground truth dataset."""
    claims: List[LabeledClaim]
    events: Dict[str, str]  # event_id -> event_name
    confounders: List[str]  # Notes on hard cases

    def claim_to_event(self) -> Dict[str, str]:
        """Map claim_id -> event_label."""
        return {lc.claim.id: lc.event_label for lc in self.claims}

    def relation_pairs(self) -> List[Tuple[str, str, str]]:
        """Get labeled relation pairs: (claim_a, claim_b, relation)."""
        pairs = []
        for lc in self.claims:
            if lc.relation_to and lc.relation_type:
                pairs.append((lc.claim.id, lc.relation_to, lc.relation_type))
        return pairs


# =============================================================================
# LEVEL 0 EVALUATION: Relation Classification
# =============================================================================

@dataclass
class Level0Metrics:
    """Metrics for claim-to-claim relation classification."""
    confusion_matrix: Dict[str, Dict[str, int]]
    per_class: Dict[str, Dict[str, float]]  # precision, recall, f1
    micro_f1: float
    macro_f1: float
    total_pairs: int
    correct: int


def evaluate_level0(
    engine: EmergenceEngine,
    ground_truth: GroundTruth
) -> Level0Metrics:
    """
    Evaluate Level 0 relation classification.

    Compares predicted relations to ground truth pairs.
    """
    relations = ['confirms', 'refines', 'supersedes', 'conflicts', 'unrelated']

    # Build confusion matrix
    confusion = {r: {r2: 0 for r2 in relations} for r in relations}

    # Get predicted edges as dict
    predicted = {}
    for c1, c2, rel, conf in engine.claim_edges:
        key = tuple(sorted([c1, c2]))
        predicted[key] = rel.value

    # Compare to ground truth
    gt_pairs = ground_truth.relation_pairs()
    total = len(gt_pairs)
    correct = 0

    for c1, c2, gt_rel in gt_pairs:
        key = tuple(sorted([c1, c2]))
        pred_rel = predicted.get(key, 'unrelated')
        gt_rel_lower = gt_rel.lower()

        confusion[gt_rel_lower][pred_rel] += 1
        if pred_rel == gt_rel_lower:
            correct += 1

    # Per-class metrics
    per_class = {}
    for rel in relations:
        tp = confusion[rel][rel]
        fp = sum(confusion[other][rel] for other in relations if other != rel)
        fn = sum(confusion[rel][other] for other in relations if other != rel)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[rel] = {'precision': precision, 'recall': recall, 'f1': f1}

    # Micro F1
    micro_f1 = correct / total if total > 0 else 0.0

    # Macro F1
    macro_f1 = np.mean([per_class[r]['f1'] for r in relations])

    return Level0Metrics(
        confusion_matrix=confusion,
        per_class=per_class,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        total_pairs=total,
        correct=correct
    )


# =============================================================================
# LEVEL 1+ EVALUATION: Clustering
# =============================================================================

@dataclass
class ClusterMetrics:
    """Metrics for surface/event clustering."""
    b3_precision: float
    b3_recall: float
    b3_f1: float
    v_measure: float
    completeness: float  # Per-event: max fraction in one cluster
    fragmentation: float  # 1 - completeness (how split up)
    num_clusters: int
    num_ground_truth: int
    cluster_sizes: List[int]


def evaluate_clustering(
    surfaces: List[Surface],
    ground_truth: GroundTruth
) -> ClusterMetrics:
    """
    Evaluate Level 1+ clustering with B³, V-measure, completeness.
    """
    gt_map = ground_truth.claim_to_event()

    # Build cluster assignment: claim_id -> surface_id
    claim_to_surface = {}
    for surface in surfaces:
        for claim_id in surface.claim_ids:
            claim_to_surface[claim_id] = surface.id

    # B³ Score
    precision_sum = 0.0
    recall_sum = 0.0
    n = len(gt_map)

    gt_clusters = defaultdict(set)
    for claim_id, event_label in gt_map.items():
        gt_clusters[event_label].add(claim_id)

    for claim_id, event_label in gt_map.items():
        if claim_id not in claim_to_surface:
            continue

        surface_id = claim_to_surface[claim_id]
        surface = next((s for s in surfaces if s.id == surface_id), None)
        if not surface:
            continue

        # Items in same predicted cluster
        pred_cluster = surface.claim_ids & set(gt_map.keys())

        # Items in same ground truth cluster
        gt_cluster = gt_clusters[event_label]

        # Intersection
        correct = pred_cluster & gt_cluster

        if pred_cluster:
            precision_sum += len(correct) / len(pred_cluster)
        if gt_cluster:
            recall_sum += len(correct) / len(gt_cluster)

    b3_precision = precision_sum / n if n > 0 else 0.0
    b3_recall = recall_sum / n if n > 0 else 0.0
    b3_f1 = 2 * b3_precision * b3_recall / (b3_precision + b3_recall) if (b3_precision + b3_recall) > 0 else 0.0

    # V-measure (entropy-based)
    v_measure = compute_v_measure(surfaces, gt_map)

    # Completeness: for each GT event, what fraction in largest cluster?
    completeness_scores = []
    for event_label, gt_items in gt_clusters.items():
        cluster_counts = defaultdict(int)
        for claim_id in gt_items:
            if claim_id in claim_to_surface:
                cluster_counts[claim_to_surface[claim_id]] += 1

        if cluster_counts:
            max_in_one = max(cluster_counts.values())
            completeness_scores.append(max_in_one / len(gt_items))

    completeness = np.mean(completeness_scores) if completeness_scores else 0.0
    fragmentation = 1 - completeness

    return ClusterMetrics(
        b3_precision=b3_precision,
        b3_recall=b3_recall,
        b3_f1=b3_f1,
        v_measure=v_measure,
        completeness=completeness,
        fragmentation=fragmentation,
        num_clusters=len(surfaces),
        num_ground_truth=len(gt_clusters),
        cluster_sizes=[len(s.claim_ids) for s in surfaces]
    )


def compute_v_measure(surfaces: List[Surface], gt_map: Dict[str, str]) -> float:
    """Compute V-measure (harmonic mean of homogeneity and completeness)."""
    from math import log

    # Build contingency table
    cluster_ids = list(set(s.id for s in surfaces))
    class_ids = list(set(gt_map.values()))

    if len(cluster_ids) <= 1 or len(class_ids) <= 1:
        return 1.0 if len(cluster_ids) == len(class_ids) else 0.0

    # Count matrix
    n = len(gt_map)
    contingency = defaultdict(lambda: defaultdict(int))

    for surface in surfaces:
        for claim_id in surface.claim_ids:
            if claim_id in gt_map:
                contingency[surface.id][gt_map[claim_id]] += 1

    # Entropy calculations
    def entropy(counts):
        total = sum(counts)
        if total == 0:
            return 0
        return -sum((c/total) * log(c/total) for c in counts if c > 0)

    # H(C) - entropy of classes
    class_counts = defaultdict(int)
    for claim_id, label in gt_map.items():
        class_counts[label] += 1
    H_C = entropy(class_counts.values())

    # H(K) - entropy of clusters
    cluster_counts = [len(s.claim_ids & set(gt_map.keys())) for s in surfaces]
    H_K = entropy(cluster_counts)

    # H(C|K) - conditional entropy
    H_C_given_K = 0
    for surface in surfaces:
        k_count = len(surface.claim_ids & set(gt_map.keys()))
        if k_count == 0:
            continue
        class_dist = [contingency[surface.id][c] for c in class_ids]
        H_C_given_K += (k_count / n) * entropy(class_dist)

    # H(K|C)
    H_K_given_C = 0
    for label in class_ids:
        c_count = class_counts[label]
        if c_count == 0:
            continue
        cluster_dist = [contingency[s.id][label] for s in surfaces]
        H_K_given_C += (c_count / n) * entropy(cluster_dist)

    # Homogeneity and completeness
    homogeneity = 1 - (H_C_given_K / H_C) if H_C > 0 else 1.0
    completeness_v = 1 - (H_K_given_C / H_K) if H_K > 0 else 1.0

    # V-measure
    if homogeneity + completeness_v == 0:
        return 0.0
    return 2 * homogeneity * completeness_v / (homogeneity + completeness_v)


# =============================================================================
# COHERENCE EVALUATION
# =============================================================================

@dataclass
class CoherenceMetrics:
    """Coherence signals per cluster."""
    temporal_violations: float  # % pairs violating time gate
    anchor_coherence: float  # Average anchor overlap
    anchor_specificity: float  # Rare entity bonus
    semantic_dispersion: float  # Embedding variance


def evaluate_coherence(
    surfaces: List[Surface],
    claims: Dict[str, Claim],
    time_gate_days: int = 30
) -> Dict[str, CoherenceMetrics]:
    """Evaluate coherence signals for each surface."""
    results = {}

    for surface in surfaces:
        if len(surface.claim_ids) < 2:
            results[surface.id] = CoherenceMetrics(0, 1, 1, 0)
            continue

        surface_claims = [claims[cid] for cid in surface.claim_ids if cid in claims]

        # Temporal violations
        temporal_violations = 0
        temporal_pairs = 0
        for i, c1 in enumerate(surface_claims):
            for c2 in surface_claims[i+1:]:
                if c1.timestamp and c2.timestamp:
                    days_apart = abs((c1.timestamp - c2.timestamp).days)
                    temporal_pairs += 1
                    if days_apart > time_gate_days:
                        temporal_violations += 1

        temp_viol_rate = temporal_violations / temporal_pairs if temporal_pairs > 0 else 0.0

        # Anchor coherence: what fraction of pairs share an anchor?
        anchor_overlaps = []
        for i, c1 in enumerate(surface_claims):
            for c2 in surface_claims[i+1:]:
                if c1.anchor_entities and c2.anchor_entities:
                    overlap = len(c1.anchor_entities & c2.anchor_entities) / len(c1.anchor_entities | c2.anchor_entities)
                    anchor_overlaps.append(overlap)

        anchor_coherence = np.mean(anchor_overlaps) if anchor_overlaps else 0.0

        # Anchor specificity: bonus for rare entities (inverse document frequency)
        all_anchors = set()
        for c in surface_claims:
            all_anchors.update(c.anchor_entities)
        anchor_specificity = 1 / len(all_anchors) if all_anchors else 0.0

        # Semantic dispersion: variance of embeddings from centroid
        embeddings = [c.embedding for c in surface_claims if c.embedding]
        if len(embeddings) >= 2 and surface.centroid:
            distances = [1 - cosine_similarity(e, surface.centroid) for e in embeddings]
            semantic_dispersion = float(np.mean(distances))
        else:
            semantic_dispersion = 0.0

        results[surface.id] = CoherenceMetrics(
            temporal_violations=temp_viol_rate,
            anchor_coherence=anchor_coherence,
            anchor_specificity=anchor_specificity,
            semantic_dispersion=semantic_dispersion
        )

    return results


# =============================================================================
# STABILITY EVALUATION
# =============================================================================

@dataclass
class StabilityMetrics:
    """Stability across random orders and thresholds."""
    order_jaccard: float  # Cluster assignment stability
    order_std: float  # Standard deviation across runs
    threshold_curve: List[Dict]  # Precision/recall at different thresholds


async def evaluate_stability(
    claims: List[Claim],
    ground_truth: GroundTruth,
    llm,
    num_shuffles: int = 5,
    thresholds: List[float] = None
) -> StabilityMetrics:
    """
    Evaluate stability across random orders and threshold sweeps.
    """
    import random

    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
    gt_map = ground_truth.claim_to_event()

    # Order shuffle test
    assignments_per_run = []

    for run in range(num_shuffles):
        shuffled = claims.copy()
        random.shuffle(shuffled)

        engine = EmergenceEngine(llm=llm)
        for claim in shuffled:
            await engine.add_claim(claim)
        surfaces = engine.compute_surfaces()

        # Record assignments
        assignment = {}
        for surface in surfaces:
            for claim_id in surface.claim_ids:
                assignment[claim_id] = surface.id
        assignments_per_run.append(assignment)

    # Compute Jaccard stability
    jaccards = []
    for i in range(len(assignments_per_run)):
        for j in range(i + 1, len(assignments_per_run)):
            a1, a2 = assignments_per_run[i], assignments_per_run[j]
            # Same cluster if assigned together
            pairs1 = set()
            pairs2 = set()
            claim_ids = list(set(a1.keys()) & set(a2.keys()))
            for k, c1 in enumerate(claim_ids):
                for c2 in claim_ids[k+1:]:
                    if a1.get(c1) == a1.get(c2):
                        pairs1.add((c1, c2))
                    if a2.get(c1) == a2.get(c2):
                        pairs2.add((c1, c2))

            if pairs1 or pairs2:
                jaccard = len(pairs1 & pairs2) / len(pairs1 | pairs2)
                jaccards.append(jaccard)

    order_jaccard = np.mean(jaccards) if jaccards else 1.0
    order_std = np.std(jaccards) if jaccards else 0.0

    # Threshold sweep (simplified - just vary affinity threshold)
    threshold_curve = []
    # Note: full implementation would require modifying Claim.should_compare threshold

    return StabilityMetrics(
        order_jaccard=order_jaccard,
        order_std=order_std,
        threshold_curve=threshold_curve
    )


# =============================================================================
# ABLATION STUDY
# =============================================================================

@dataclass
class AblationResult:
    """Result of one ablation configuration."""
    config: str
    b3_f1: float
    completeness: float
    num_clusters: int


async def run_ablations(
    claims: List[Claim],
    ground_truth: GroundTruth,
    llm
) -> List[AblationResult]:
    """
    Run ablation study: embeddings only, +entities, +time, +anchors.
    """
    results = []
    gt_map = ground_truth.claim_to_event()

    configs = [
        ("embeddings_only", True, False, False, False),
        ("+entities", True, True, False, False),
        ("+time", True, True, True, False),
        ("+anchors (full)", True, True, True, True),
    ]

    for name, use_emb, use_ent, use_time, use_anchor in configs:
        # Modify claims for this config
        modified_claims = []
        for c in claims:
            mc = Claim(
                id=c.id,
                text=c.text,
                source=c.source,
                embedding=c.embedding if use_emb else None,
                entities=c.entities if use_ent else set(),
                anchor_entities=c.anchor_entities if use_anchor else set(),
                timestamp=c.timestamp if use_time else None
            )
            modified_claims.append(mc)

        # Run engine
        engine = EmergenceEngine(llm=llm)
        for claim in modified_claims:
            await engine.add_claim(claim)
        surfaces = engine.compute_surfaces()

        # Evaluate
        cluster_metrics = evaluate_clustering(surfaces, ground_truth)

        results.append(AblationResult(
            config=name,
            b3_f1=cluster_metrics.b3_f1,
            completeness=cluster_metrics.completeness,
            num_clusters=cluster_metrics.num_clusters
        ))

    return results


# =============================================================================
# LLM ROI EVALUATION
# =============================================================================

@dataclass
class LLMRoiMetrics:
    """ROI metrics for LLM calls."""
    total_claims: int
    llm_calls: int
    calls_per_100: float
    f1_base: float  # Without LLM
    f1_with_llm: float  # With LLM
    f1_gain: float
    f1_gain_per_call: float


async def evaluate_llm_roi(
    claims: List[Claim],
    ground_truth: GroundTruth,
    llm
) -> LLMRoiMetrics:
    """Compare base (no LLM) vs LLM-assisted clustering."""
    # Base: no LLM (affinity only)
    engine_base = EmergenceEngine(llm=None)
    for claim in claims:
        await engine_base.add_claim(claim)
    surfaces_base = engine_base.compute_surfaces()
    metrics_base = evaluate_clustering(surfaces_base, ground_truth)

    # With LLM
    engine_llm = EmergenceEngine(llm=llm)
    for claim in claims:
        await engine_llm.add_claim(claim)
    surfaces_llm = engine_llm.compute_surfaces()
    metrics_llm = evaluate_clustering(surfaces_llm, ground_truth)

    # Count LLM calls (edges = LLM calls for relation classification)
    llm_calls = len(engine_llm.claim_edges)

    f1_gain = metrics_llm.b3_f1 - metrics_base.b3_f1
    f1_gain_per_call = f1_gain / llm_calls if llm_calls > 0 else 0.0

    return LLMRoiMetrics(
        total_claims=len(claims),
        llm_calls=llm_calls,
        calls_per_100=llm_calls / len(claims) * 100 if claims else 0,
        f1_base=metrics_base.b3_f1,
        f1_with_llm=metrics_llm.b3_f1,
        f1_gain=f1_gain,
        f1_gain_per_call=f1_gain_per_call
    )


# =============================================================================
# FULL EVALUATION REPORT
# =============================================================================

@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    level0: Optional[Level0Metrics]
    clustering: ClusterMetrics
    coherence: Dict[str, CoherenceMetrics]
    stability: Optional[StabilityMetrics]
    ablations: Optional[List[AblationResult]]
    llm_roi: Optional[LLMRoiMetrics]


def print_report(report: EvaluationReport):
    """Pretty print evaluation report."""
    print("=" * 70)
    print("EPISTEMIC EVALUATION REPORT")
    print("=" * 70)

    if report.level0:
        print("\n--- Level 0: Relation Classification ---")
        print(f"Micro F1: {report.level0.micro_f1:.2%}")
        print(f"Macro F1: {report.level0.macro_f1:.2%}")
        print(f"Total pairs: {report.level0.total_pairs}, Correct: {report.level0.correct}")
        print("\nPer-class:")
        for rel, metrics in report.level0.per_class.items():
            print(f"  {rel}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    print("\n--- Level 1+: Clustering ---")
    print(f"B³ Precision: {report.clustering.b3_precision:.2%}")
    print(f"B³ Recall: {report.clustering.b3_recall:.2%}")
    print(f"B³ F1: {report.clustering.b3_f1:.2%}")
    print(f"V-measure: {report.clustering.v_measure:.2%}")
    print(f"Completeness: {report.clustering.completeness:.2%}")
    print(f"Fragmentation: {report.clustering.fragmentation:.2%}")
    print(f"Clusters: {report.clustering.num_clusters} (GT: {report.clustering.num_ground_truth})")

    print("\n--- Coherence (averaged) ---")
    if report.coherence:
        avg_temp = np.mean([c.temporal_violations for c in report.coherence.values()])
        avg_anchor = np.mean([c.anchor_coherence for c in report.coherence.values()])
        avg_disp = np.mean([c.semantic_dispersion for c in report.coherence.values()])
        print(f"Temporal violations: {avg_temp:.2%}")
        print(f"Anchor coherence: {avg_anchor:.2%}")
        print(f"Semantic dispersion: {avg_disp:.3f}")

    if report.stability:
        print("\n--- Stability ---")
        print(f"Order Jaccard: {report.stability.order_jaccard:.2%} ± {report.stability.order_std:.2%}")

    if report.ablations:
        print("\n--- Ablations ---")
        for ab in report.ablations:
            print(f"  {ab.config}: B³F1={ab.b3_f1:.2%}, Completeness={ab.completeness:.2%}, Clusters={ab.num_clusters}")

    if report.llm_roi:
        print("\n--- LLM ROI ---")
        print(f"LLM calls: {report.llm_roi.llm_calls} ({report.llm_roi.calls_per_100:.1f} per 100 claims)")
        print(f"F1 base: {report.llm_roi.f1_base:.2%}")
        print(f"F1 with LLM: {report.llm_roi.f1_with_llm:.2%}")
        print(f"F1 gain: {report.llm_roi.f1_gain:.2%}")
        print(f"F1 gain per call: {report.llm_roi.f1_gain_per_call:.4f}")


# =============================================================================
# TEST WITH SYNTHETIC DATA
# =============================================================================

def create_test_ground_truth() -> GroundTruth:
    """Create a small adversarial test set."""
    claims = [
        # Event 1: Hong Kong Fire
        LabeledClaim(
            Claim(id="hk1", text="Fire kills 13 in Hong Kong high-rise", source="BBC",
                  entities={"Hong Kong", "fire"}, anchor_entities=set()),
            event_label="hk_fire"
        ),
        LabeledClaim(
            Claim(id="hk2", text="Death toll reaches 13 in Hong Kong apartment blaze", source="Reuters",
                  entities={"Hong Kong", "fire"}, anchor_entities=set()),
            event_label="hk_fire",
            relation_to="hk1",
            relation_type="confirms"
        ),
        LabeledClaim(
            Claim(id="hk3", text="John Lee visits fire victims in hospital", source="SCMP",
                  entities={"Hong Kong", "John Lee"}, anchor_entities={"John Lee"}),
            event_label="hk_fire",
            relation_to="hk1",
            relation_type="refines"
        ),

        # Event 2: Jimmy Lai Trial (same location: Hong Kong - confounder!)
        LabeledClaim(
            Claim(id="lai1", text="Jimmy Lai trial continues in Hong Kong court", source="BBC",
                  entities={"Hong Kong", "Jimmy Lai"}, anchor_entities={"Jimmy Lai"}),
            event_label="lai_trial"
        ),
        LabeledClaim(
            Claim(id="lai2", text="Lai faces national security charges", source="Guardian",
                  entities={"Jimmy Lai", "national security"}, anchor_entities={"Jimmy Lai"}),
            event_label="lai_trial",
            relation_to="lai1",
            relation_type="refines"
        ),

        # Event 3: Charlie Kirk shooting (homonym confounder!)
        LabeledClaim(
            Claim(id="ck1", text="Charlie Kirk shot at political event", source="NYT",
                  entities={"Charlie Kirk", "shooting"}, anchor_entities={"Charlie Kirk"}),
            event_label="ck_shooting"
        ),
        LabeledClaim(
            Claim(id="ck2", text="Tyler Robinson charged in Charlie Kirk killing", source="AP",
                  entities={"Charlie Kirk", "Tyler Robinson"}, anchor_entities={"Charlie Kirk", "Tyler Robinson"}),
            event_label="ck_shooting",
            relation_to="ck1",
            relation_type="refines"
        ),

        # Confounder: Charlie Kirk (TPUSA) - different person!
        LabeledClaim(
            Claim(id="tpusa1", text="Charlie Kirk speaks at TPUSA conference", source="Fox",
                  entities={"Charlie Kirk", "TPUSA"}, anchor_entities={"Charlie Kirk"}),
            event_label="tpusa_event"
        ),
    ]

    return GroundTruth(
        claims=claims,
        events={
            "hk_fire": "Hong Kong High-Rise Fire",
            "lai_trial": "Jimmy Lai National Security Trial",
            "ck_shooting": "Charlie Kirk Shooting",
            "tpusa_event": "TPUSA Conference"
        },
        confounders=[
            "Hong Kong appears in both fire and Lai trial",
            "Charlie Kirk is a homonym (shooting victim vs TPUSA founder)"
        ]
    )


if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 70)
        print("EVALUATION HARNESS TEST")
        print("=" * 70)

        # Create ground truth
        gt = create_test_ground_truth()
        print(f"\nGround truth: {len(gt.claims)} claims, {len(gt.events)} events")
        print(f"Confounders: {gt.confounders}")

        # Run engine (no LLM for test)
        engine = EmergenceEngine(llm=None)
        for lc in gt.claims:
            await engine.add_claim(lc.claim)
        surfaces = engine.compute_surfaces()

        # Evaluate
        cluster_metrics = evaluate_clustering(surfaces, gt)
        coherence = evaluate_coherence(surfaces, engine.claims)

        # Print results
        print(f"\n--- Clustering Metrics ---")
        print(f"B³ F1: {cluster_metrics.b3_f1:.2%}")
        print(f"Completeness: {cluster_metrics.completeness:.2%}")
        print(f"Clusters: {cluster_metrics.num_clusters} (GT: {cluster_metrics.num_ground_truth})")

        print(f"\n--- Coherence ---")
        for sid, coh in coherence.items():
            print(f"  {sid}: temp_viol={coh.temporal_violations:.0%}, anchor={coh.anchor_coherence:.2f}")

    asyncio.run(test())
