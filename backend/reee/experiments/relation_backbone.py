"""
Relation Backbone Experiment
============================

Tests whether adding entity relations as aboutness signal improves event emergence.

Hypothesis: Surfaces with related (but not identical) anchors should connect
when the relation is corroborated, improving B³ recall without hurting precision.

Design:
1. Run baseline emergence (no relation signal)
2. Extract relations from claims (rule-based, high-precision)
3. Build relation belief states (simulate corroboration)
4. Run emergence WITH relation signal
5. Compare B³ metrics, false merge rate, cross-topic mixing

Run:
    docker exec herenews-app python -m reee.experiments.relation_backbone
    docker exec herenews-app python -m reee.experiments.relation_backbone --claims 500
"""

import asyncio
import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Literal

from reee import Claim, Parameters
from reee.types import Surface, Event
from reee.identity import IdentityLinker
from reee.aboutness.scorer import AboutnessScorer, compute_events_from_aboutness, cosine_similarity
from reee.aboutness.metrics import compute_b3_metrics, compute_purity_metrics
from reee.experiments.loader import create_context, log


# =============================================================================
# RELATION TYPES AND EXTRACTION
# =============================================================================

@dataclass
class RelationAssertion:
    """A single assertion of a relation from a claim."""
    subject: str
    relation: str
    object: str
    polarity: Literal['asserted', 'negated', 'uncertain']
    confidence: float
    source_claim_id: str


@dataclass
class RelationBeliefState:
    """Belief state over a relation proposition."""
    subject: str
    relation: str
    object: str

    p_asserted: float = 0.0
    p_negated: float = 0.0
    p_uncertain: float = 1.0  # Prior: uncertain

    asserting_claims: List[str] = field(default_factory=list)
    negating_claims: List[str] = field(default_factory=list)
    uncertain_claims: List[str] = field(default_factory=list)

    @property
    def is_safe_to_use(self) -> bool:
        """Should this relation be used as aboutness binder?"""
        return (
            self.p_asserted > 0.7 and
            self.p_negated < 0.15 and
            len(self.asserting_claims) >= 1  # Relaxed for experiment
        )


# High-precision relation patterns (Tier 1)
# Pattern matches the text BETWEEN two entities
# 'passive' means: first entity is object, second is subject (X was founded by Y => Y founded X)
# 'active' means: first entity is subject, second is object (X founded Y => X founded Y)
#
# CRITICAL: Check for "was...by" structure to detect passive voice
RELATION_PATTERNS = {
    'FOUNDED': [
        # Passive: "X was founded by Y" - between text contains "was...founded...by"
        (r'^\s*was\s+(?:co-)?founded\s+by\s*$', 'passive'),
        (r'^\s*,?\s*(?:co-)?founder\s+of\s*$', 'passive'),
        # Active: "X founded Y" - between text is just "founded" without "was" or "by"
        (r'^\s+(?:co-)?founded\s+(?:the\s+)?$', 'active'),
        (r'^\s+(?:co-)?founded\s*$', 'active'),
    ],
    'CEO_OF': [
        # Passive: "X, CEO of Y"
        (r'^,?\s*(?:the\s+)?(?:CEO|chief executive|head)\s+of\s*$', 'passive'),
        # Active: "X leads Y"
        (r'^\s+leads?\s+$', 'active'),
        (r'^\s+runs?\s+$', 'active'),
    ],
    'CREATED': [
        # Passive: "X, creator of Y"
        (r'^,?\s*creator\s+of\s*$', 'passive'),
        # Active: "X created Y"
        (r'^\s+created\s+$', 'active'),
        (r'^\s+built\s+$', 'active'),
        (r'^\s+developed\s+$', 'active'),
    ],
    'SUBSIDIARY_OF': [
        # Passive: "X, a subsidiary of Y"
        (r'^,?\s*(?:a\s+)?subsidiary\s+of\s*$', 'passive'),
        (r'^\s+is\s+owned\s+by\s*$', 'passive'),
        (r'^\s+is\s+part\s+of\s*$', 'passive'),
    ],
}

HEDGE_MARKERS = ['allegedly', 'reportedly', 'claimed', 'accused', 'suspected', 'may have']
NEGATION_MARKERS = ['not', 'never', 'denied', 'no longer', "didn't", "wasn't"]


def detect_polarity(text: str) -> Literal['asserted', 'negated', 'uncertain']:
    """Detect polarity from surrounding text."""
    text_lower = text.lower()
    for neg in NEGATION_MARKERS:
        if neg in text_lower:
            return 'negated'
    for hedge in HEDGE_MARKERS:
        if hedge in text_lower:
            return 'uncertain'
    return 'asserted'


# Direct text patterns for relations (bypasses incomplete NER)
DIRECT_RELATION_PATTERNS = [
    # FOUNDED patterns - passive voice (org founded by person)
    (re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Labs?|Inc|Corp|LLC|Ltd))?)\s+'
        r'was\s+(?:co-)?founded\s+by\s+'
        r'([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)',
        re.IGNORECASE
    ), 'FOUNDED', 'passive'),
    # FOUNDED patterns - active voice (person founded org)
    (re.compile(
        r'([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)\s+'
        r'(?:co-)?founded\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Labs?|Inc|Corp|LLC|Ltd))?)',
        re.IGNORECASE
    ), 'FOUNDED', 'active'),
    # CEO patterns
    (re.compile(
        r'([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+),?\s+'
        r'(?:the\s+)?(?:CEO|chief executive|head)\s+of\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        re.IGNORECASE
    ), 'CEO_OF', 'active'),
    # CREATED patterns
    (re.compile(
        r'([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)\s+'
        r'created\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        re.IGNORECASE
    ), 'CREATED', 'active'),
]


def extract_relations_from_text(
    claim_text: str,
    claim_id: str,
) -> List[RelationAssertion]:
    """
    Extract relations directly from text using high-precision patterns.

    Does NOT require entities to be in the NER set (bypasses incomplete NER).
    """
    relations = []

    for pattern, rel_type, direction in DIRECT_RELATION_PATTERNS:
        for match in pattern.finditer(claim_text):
            if direction == 'passive':
                # Passive: "X was founded by Y" → Y FOUNDED X
                obj, subject = match.group(1), match.group(2)
            else:
                # Active: "Y founded X" → Y FOUNDED X
                subject, obj = match.group(1), match.group(2)

            polarity = detect_polarity(claim_text)
            confidence = 0.9 if polarity == 'asserted' else 0.4

            relations.append(RelationAssertion(
                subject=subject.strip(),
                relation=rel_type,
                object=obj.strip(),
                polarity=polarity,
                confidence=confidence,
                source_claim_id=claim_id,
            ))

    return relations


def extract_relations_from_claim(
    claim_text: str,
    claim_id: str,
    entities: Set[str],
) -> List[RelationAssertion]:
    """
    Extract relation assertions from a claim.

    Uses pattern matching between known entity mentions.
    """
    relations = []
    text_lower = claim_text.lower()
    entity_list = list(entities)

    for i, ent_a in enumerate(entity_list):
        for ent_b in entity_list[i+1:]:
            # Find positions of entities in text
            pos_a = text_lower.find(ent_a.lower())
            pos_b = text_lower.find(ent_b.lower())

            if pos_a == -1 or pos_b == -1:
                continue

            # Get text between entities
            if pos_a < pos_b:
                between = claim_text[pos_a + len(ent_a):pos_b]
                first, second = ent_a, ent_b
            else:
                between = claim_text[pos_b + len(ent_b):pos_a]
                first, second = ent_b, ent_a

            # Check patterns
            for rel_type, patterns in RELATION_PATTERNS.items():
                for pattern, direction in patterns:
                    if re.search(pattern, between, re.IGNORECASE):
                        # Determine subject/object based on direction
                        if direction == 'passive':
                            subject, obj = second, first
                        else:
                            subject, obj = first, second

                        polarity = detect_polarity(between)
                        confidence = 0.8 if polarity == 'asserted' else 0.4

                        relations.append(RelationAssertion(
                            subject=subject,
                            relation=rel_type,
                            object=obj,
                            polarity=polarity,
                            confidence=confidence,
                            source_claim_id=claim_id,
                        ))
                        break  # One relation per pattern type

    return relations


def build_relation_beliefs(
    assertions: List[RelationAssertion],
) -> Dict[Tuple[str, str, str], RelationBeliefState]:
    """
    Build belief states from assertions.

    Uses simple counting with normalization.
    Multiple independent sources asserting the same relation
    increases confidence significantly.
    """
    # Group assertions by relation
    grouped: Dict[Tuple[str, str, str], List[RelationAssertion]] = defaultdict(list)
    for assertion in assertions:
        key = (assertion.subject, assertion.relation, assertion.object)
        grouped[key].append(assertion)

    beliefs = {}
    for key, assertion_list in grouped.items():
        belief = RelationBeliefState(
            subject=key[0],
            relation=key[1],
            object=key[2],
        )

        # Collect claims by polarity
        for a in assertion_list:
            if a.polarity == 'asserted':
                belief.asserting_claims.append(a.source_claim_id)
            elif a.polarity == 'negated':
                belief.negating_claims.append(a.source_claim_id)
            else:
                belief.uncertain_claims.append(a.source_claim_id)

        # Compute weighted counts
        n_assert = len(belief.asserting_claims)
        n_negate = len(belief.negating_claims)
        n_uncertain = len(belief.uncertain_claims)

        # Weight: asserted/negated count more than uncertain
        w_assert = n_assert * 1.0
        w_negate = n_negate * 1.0
        w_uncertain = n_uncertain * 0.2

        total = w_assert + w_negate + w_uncertain + 0.1  # Small prior for uncertainty

        belief.p_asserted = w_assert / total
        belief.p_negated = w_negate / total
        belief.p_uncertain = 1.0 - belief.p_asserted - belief.p_negated

        beliefs[key] = belief

    return beliefs


def build_entity_graph(
    beliefs: Dict[Tuple[str, str, str], RelationBeliefState],
) -> Dict[str, Set[str]]:
    """
    Build entity graph from safe-to-use relations.

    Returns: entity -> set of related entities (1-hop)
    """
    graph = defaultdict(set)

    for (subject, rel, obj), belief in beliefs.items():
        if belief.is_safe_to_use:
            graph[subject].add(obj)
            graph[obj].add(subject)  # Bidirectional

    return dict(graph)


# =============================================================================
# MODIFIED ABOUTNESS SCORER WITH RELATION SIGNAL
# =============================================================================

class RelationAwareAboutnessScorer(AboutnessScorer):
    """
    Aboutness scorer that uses entity relations as additional signal.
    """

    def __init__(
        self,
        surfaces: Dict[str, Surface],
        params: Parameters,
        entity_graph: Dict[str, Set[str]],
    ):
        super().__init__(surfaces, params)
        self.entity_graph = entity_graph

    def anchors_related(self, anchors1: Set[str], anchors2: Set[str]) -> bool:
        """Check if any anchor from set1 is related to any in set2."""
        for a1 in anchors1:
            if a1 in self.entity_graph:
                if anchors2 & self.entity_graph[a1]:
                    return True
        return False

    def score_pair(self, s1: Surface, s2: Surface) -> Tuple[float, Dict]:
        """
        Score with relation signal.

        Related anchors count as a signal (like shared anchors),
        but still require multi-signal for edge formation.
        """
        # Get base score and evidence
        base_score, evidence = super().score_pair(s1, s2)

        # Check for related anchors (not just shared)
        has_shared = bool(s1.anchor_entities & s2.anchor_entities)
        has_related = self.anchors_related(s1.anchor_entities, s2.anchor_entities)

        evidence['has_related_anchors'] = has_related
        evidence['relation_signal'] = has_related and not has_shared

        # If we have related anchors but no shared anchors,
        # and we failed due to no discriminative anchor,
        # reconsider with relation signal
        if has_related and not has_shared:
            gate = evidence.get('gate')

            if gate == 'no_discriminative_anchor':
                # Related anchors can substitute for discriminative anchor
                # BUT still require other signals (semantic, temporal)
                semantic = evidence.get('semantic_score', 0)
                temporal_ok = evidence.get('temporal_compatible', False)

                if semantic > 0.5 and temporal_ok:
                    # Recalculate score with relation bonus
                    anchor_score = 0.4  # Moderate anchor credit for relation
                    entity_score = evidence.get('entity_score', 0)
                    source_div = evidence.get('source_diversity', 0.5)

                    new_score = (
                        0.35 * anchor_score +
                        0.35 * semantic +
                        0.20 * entity_score +
                        0.10 * source_div
                    )

                    evidence['relation_rescue'] = True
                    evidence['gate'] = None
                    return new_score, evidence

            elif gate == 'signals_met < min':
                # Check if relation signal would add a signal
                signals = evidence.get('signals_met', 0)
                if has_related:
                    signals += 1
                    evidence['signals_met'] = signals

                    if signals >= self.params.aboutness_min_signals:
                        # Recalculate
                        semantic = evidence.get('semantic_score', 0)
                        anchor_score = 0.3  # Credit for relation
                        entity_score = evidence.get('entity_score', 0)
                        source_div = evidence.get('source_diversity', 0.5)

                        new_score = (
                            0.35 * anchor_score +
                            0.35 * semantic +
                            0.20 * entity_score +
                            0.10 * source_div
                        )

                        if new_score > self.params.aboutness_threshold:
                            evidence['relation_rescue'] = True
                            evidence['gate'] = None
                            return new_score, evidence

        return base_score, evidence


# =============================================================================
# EXPERIMENT
# =============================================================================

async def load_claims_with_legacy(ctx, limit: int = 200):
    """Load claims with legacy event labels."""
    from pgvector.asyncpg import register_vector

    HUB_LOCATIONS = {'Hong Kong', 'China', 'United States', 'UK', 'United Kingdom', 'US'}

    claims_data = await ctx.neo4j._execute_read('''
        MATCH (c:Claim)
        WHERE c.text IS NOT NULL
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        OPTIONAL MATCH (e:Event)-[:INTAKES]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(ent:Entity)
        WITH c, p, e, collect({name: ent.canonical_name, type: ent.entity_type}) as entities
        RETURN c.id as id, c.text as text, p.domain as source,
               entities, e.canonical_name as legacy_event,
               p.pub_time as pub_time
        LIMIT $limit
    ''', {'limit': limit})

    claims = []
    claim_to_legacy = {}

    async with ctx.db_pool.acquire() as conn:
        await register_vector(conn)

        for row in claims_data:
            if not row['text']:
                continue

            embedding = await conn.fetchval(
                'SELECT embedding FROM core.claim_embeddings WHERE claim_id = $1',
                row['id']
            )

            all_entities = set()
            anchor_entities = set()

            for ent in row['entities']:
                ent_name = ent.get('name')
                if ent_name:
                    all_entities.add(ent_name)
                    ent_type = ent.get('type')
                    if ent_type in ('PERSON', 'ORGANIZATION', 'ORG'):
                        anchor_entities.add(ent_name)
                    elif ent_type == 'LOCATION' and ent_name not in HUB_LOCATIONS:
                        anchor_entities.add(ent_name)

            emb = None
            if embedding is not None and len(embedding) > 0:
                emb = [float(x) for x in embedding]

            timestamp = None
            pub_time = row.get('pub_time')
            if pub_time:
                try:
                    if isinstance(pub_time, str):
                        timestamp = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                    else:
                        timestamp = pub_time
                except:
                    pass

            claim = Claim(
                id=row['id'],
                text=row['text'],
                source=row['source'] or 'unknown',
                embedding=emb,
                entities=all_entities,
                anchor_entities=anchor_entities,
                timestamp=timestamp,
            )
            claims.append(claim)

            if row['legacy_event']:
                claim_to_legacy[claim.id] = row['legacy_event']

    return claims, claim_to_legacy


async def run_experiment(num_claims: int = 200):
    """Run the relation backbone experiment."""
    log("=" * 70)
    log("Relation Backbone Experiment")
    log("=" * 70)
    log(f"Testing with {num_claims} claims")
    log("")

    ctx = await create_context()

    try:
        # 1. Load claims
        log("Loading claims...")
        claims, claim_to_legacy = await load_claims_with_legacy(ctx, limit=num_claims)
        log(f"  Loaded {len(claims)} claims, {len(claim_to_legacy)} with legacy labels")

        # 2. Extract relations (using direct text patterns, bypasses incomplete NER)
        log("")
        log("Extracting relations from claims...")
        all_assertions = []
        claims_with_relations = 0

        for claim in claims:
            # Use direct text extraction (doesn't require both entities in NER)
            assertions = extract_relations_from_text(claim.text, claim.id)
            if assertions:
                claims_with_relations += 1
            all_assertions.extend(assertions)

        log(f"  Claims with relations: {claims_with_relations}")
        log(f"  Total assertions: {len(all_assertions)}")

        # Show sample assertions
        if all_assertions:
            log("  Sample assertions:")
            for a in all_assertions[:5]:
                log(f"    {a.subject} --{a.relation}--> {a.object} ({a.polarity})")

        # 3. Build relation beliefs
        log("")
        log("Building relation belief states...")
        beliefs = build_relation_beliefs(all_assertions)
        safe_relations = sum(1 for b in beliefs.values() if b.is_safe_to_use)
        log(f"  Relation propositions: {len(beliefs)}")
        log(f"  Safe to use (corroborated): {safe_relations}")

        # Show safe relations
        if beliefs:
            log("  Safe relations:")
            for key, belief in beliefs.items():
                if belief.is_safe_to_use:
                    log(f"    {belief.subject} --{belief.relation}--> {belief.object}")
                    log(f"      P(asserted)={belief.p_asserted:.2f}, claims={len(belief.asserting_claims)}")

        # 4. Build entity graph
        entity_graph = build_entity_graph(beliefs)
        log(f"  Entity graph edges: {sum(len(v) for v in entity_graph.values()) // 2}")

        # 5. Run identity linking (same for both)
        log("")
        log("Running identity linking...")
        params = Parameters(
            identity_confidence_threshold=0.45,
            hub_max_df=5,
            aboutness_min_signals=2,
            aboutness_threshold=0.15,
        )

        linker = IdentityLinker(llm=None, params=params)
        for claim in claims:
            await linker.add_claim(claim, extract_qkey=False)

        surfaces = linker.compute_surfaces()
        log(f"  Surfaces: {len(surfaces)}")

        # 6. BASELINE: Aboutness without relations
        log("")
        log("=" * 70)
        log("BASELINE (no relation signal)")
        log("=" * 70)

        baseline_scorer = AboutnessScorer(surfaces, params)
        baseline_edges = baseline_scorer.compute_all_edges()
        baseline_events = compute_events_from_aboutness(surfaces, baseline_edges, params)

        baseline_b3 = compute_b3_metrics(baseline_events, surfaces, claim_to_legacy)
        baseline_purity = compute_purity_metrics(baseline_events, surfaces, claim_to_legacy)

        log(f"  Aboutness edges: {len(baseline_edges)}")
        log(f"  Events: {len(baseline_events)}")
        log(f"  B³ Precision: {baseline_b3.precision:.1%}")
        log(f"  B³ Recall: {baseline_b3.recall:.1%}")
        log(f"  B³ F1: {baseline_b3.f1:.1%}")
        log(f"  Purity: {baseline_purity.purity:.1%}")

        # 7. WITH RELATIONS: Aboutness with relation signal
        log("")
        log("=" * 70)
        log("WITH RELATION BACKBONE")
        log("=" * 70)

        # Reset surface about_links (they were modified by baseline)
        for s in surfaces.values():
            s.about_links = []

        relation_scorer = RelationAwareAboutnessScorer(surfaces, params, entity_graph)
        relation_edges = relation_scorer.compute_all_edges()
        relation_events = compute_events_from_aboutness(surfaces, relation_edges, params)

        relation_b3 = compute_b3_metrics(relation_events, surfaces, claim_to_legacy)
        relation_purity = compute_purity_metrics(relation_events, surfaces, claim_to_legacy)

        log(f"  Aboutness edges: {len(relation_edges)}")
        log(f"  Events: {len(relation_events)}")
        log(f"  B³ Precision: {relation_b3.precision:.1%}")
        log(f"  B³ Recall: {relation_b3.recall:.1%}")
        log(f"  B³ F1: {relation_b3.f1:.1%}")
        log(f"  Purity: {relation_purity.purity:.1%}")

        # Count relation rescues
        relation_rescues = sum(
            1 for _, _, _, ev in relation_edges
            if ev.get('relation_rescue')
        )
        log(f"  Edges via relation rescue: {relation_rescues}")

        # 8. COMPARISON
        log("")
        log("=" * 70)
        log("COMPARISON")
        log("=" * 70)

        delta_edges = len(relation_edges) - len(baseline_edges)
        delta_events = len(relation_events) - len(baseline_events)
        delta_precision = relation_b3.precision - baseline_b3.precision
        delta_recall = relation_b3.recall - baseline_b3.recall
        delta_f1 = relation_b3.f1 - baseline_b3.f1
        delta_purity = relation_purity.purity - baseline_purity.purity

        log(f"  Δ Edges: {delta_edges:+d}")
        log(f"  Δ Events: {delta_events:+d}")
        log(f"  Δ B³ Precision: {delta_precision:+.1%}")
        log(f"  Δ B³ Recall: {delta_recall:+.1%}")
        log(f"  Δ B³ F1: {delta_f1:+.1%}")
        log(f"  Δ Purity: {delta_purity:+.1%}")

        log("")
        if delta_recall > 0 and delta_precision >= -0.05:
            log("✅ Relation backbone IMPROVES event emergence")
            log("   Recall increased without significant precision loss")
        elif delta_recall > 0 and delta_precision < -0.05:
            log("⚠️  Relation backbone increases recall but hurts precision")
            log("   May need tighter guardrails")
        else:
            log("❌ Relation backbone did not improve metrics")
            log("   Check: relation extraction quality, graph density")

        # 9. Detailed: which events were affected?
        log("")
        log("=" * 70)
        log("DETAILED: Events affected by relations")
        log("=" * 70)

        # Find edges that only exist with relations
        baseline_edge_set = {(e[0], e[1]) for e in baseline_edges}
        new_edges = [
            (s1, s2, score, ev)
            for s1, s2, score, ev in relation_edges
            if (s1, s2) not in baseline_edge_set and (s2, s1) not in baseline_edge_set
        ]

        log(f"New edges from relations: {len(new_edges)}")
        for s1, s2, score, ev in new_edges[:5]:
            surf1 = surfaces[s1]
            surf2 = surfaces[s2]
            log(f"  {s1[:8]} {list(surf1.anchor_entities)[:2]}")
            log(f"    <--> {s2[:8]} {list(surf2.anchor_entities)[:2]}")
            log(f"    score={score:.2f}, relation_rescue={ev.get('relation_rescue', False)}")

    finally:
        await ctx.close()


def main():
    parser = argparse.ArgumentParser(description='Relation Backbone Experiment')
    parser.add_argument('--claims', type=int, default=200, help='Number of claims')
    args = parser.parse_args()

    asyncio.run(run_experiment(num_claims=args.claims))


if __name__ == "__main__":
    main()
