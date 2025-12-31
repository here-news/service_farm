"""
Theory-Validating Tests: REEE Invariant Laws
=============================================

These tests validate the theoretical claims in REEE.md as falsifiable hypotheses.
Organized into three categories:

A) HARD INVARIANTS - Must always hold, deterministic
B) METAMORPHIC STABILITY - Structure equivalence under permutations
C) RANGE LAWS - Monotonicity/bounds across parameter sweeps

Key design principles:
- Use canonical structure views (claim sets, not IDs)
- Build fresh engines per test (no internal state mutation)
- Assert shape properties, not exact values
- Include ground-truth annotations for validation
"""

import pytest
import random
from dataclasses import dataclass
from typing import Optional

from reee import Engine, Claim, Parameters, Relation


# =============================================================================
# CANONICAL STRUCTURE VIEWS
# =============================================================================

def surface_view(engine: Engine) -> set[frozenset[str]]:
    """Canonical surface representation: set of claim ID sets."""
    return {frozenset(s.claim_ids) for s in engine.surfaces.values()}


def event_view(engine: Engine) -> set[frozenset[str]]:
    """Canonical event representation: set of claim ID sets (union of surfaces)."""
    result = set()
    for ev in engine.events.values():
        claim_ids = set()
        for sid in ev.surface_ids:
            if sid in engine.surfaces:
                claim_ids.update(engine.surfaces[sid].claim_ids)
        result.add(frozenset(claim_ids))
    return result


def compute_purity(engine: Engine, claim_to_gt: dict[str, str]) -> float:
    """
    Compute event purity: fraction of events containing only one GT event.

    Returns value in [0, 1] where 1 = perfect (no mixed events).
    """
    if not engine.events:
        return 1.0

    pure_count = 0
    for ev in engine.events.values():
        gt_events = set()
        for sid in ev.surface_ids:
            if sid in engine.surfaces:
                for cid in engine.surfaces[sid].claim_ids:
                    gt_events.add(claim_to_gt.get(cid, 'unknown'))
        if len(gt_events) == 1:
            pure_count += 1

    return pure_count / len(engine.events)


def compute_mixing_rate(engine: Engine, claim_to_gt: dict[str, str]) -> float:
    """
    Compute mixing rate: fraction of events containing claims from multiple GT events.

    Returns value in [0, 1] where 0 = no mixing (good).
    """
    return 1.0 - compute_purity(engine, claim_to_gt)


# =============================================================================
# GROUND TRUTH DATA GENERATORS
# =============================================================================

@dataclass
class GTClaim:
    """Claim with ground truth annotations."""
    claim: Claim
    gt_proposition: str  # Same fact/question (identity)
    gt_event: str        # Same incident (aboutness)


def make_claim(
    id: str,
    event: str,
    fact: str,
    source: str,
    entities: Optional[set] = None,
    anchor: Optional[str] = None
) -> Claim:
    """Helper to create claims with consistent structure."""
    if entities is None:
        entities = {event, fact}
    if anchor is None:
        anchor = event
    return Claim(
        id=id,
        text=f"{fact} in {event}",
        source=source,
        entities=entities,
        anchor_entities={anchor},
        question_key=fact,
    )


def generate_controlled_dataset(
    n_events: int = 3,
    n_facts_per_event: int = 2,
    n_sources_per_fact: int = 2,
    hub_entity: Optional[str] = None,
    hub_rate: float = 0.0,
) -> tuple[list[GTClaim], list[tuple], dict[str, str]]:
    """
    Generate controlled dataset with known ground truth.

    Returns:
        (gt_claims, identity_edges, claim_to_gt_event)
    """
    gt_claims = []
    identity_edges = []
    claim_to_gt = {}

    claim_counter = 0
    for ev_idx in range(n_events):
        event_name = f"Event_{ev_idx}"

        for fact_idx in range(n_facts_per_event):
            fact_name = f"fact_{ev_idx}_{fact_idx}"
            proposition_id = f"{event_name}:{fact_name}"

            fact_claims = []
            for src_idx in range(n_sources_per_fact):
                claim_counter += 1
                claim_id = f"c{claim_counter:03d}"

                # Base entities
                entities = {event_name, fact_name}

                # Add hub entity based on rate
                if hub_entity and random.random() < hub_rate:
                    entities.add(hub_entity)

                claim = make_claim(
                    id=claim_id,
                    event=event_name,
                    fact=fact_name,
                    source=f"source_{src_idx}.com",
                    entities=entities,
                    anchor=event_name
                )

                gt_claims.append(GTClaim(
                    claim=claim,
                    gt_proposition=proposition_id,
                    gt_event=event_name,
                ))

                fact_claims.append(claim_id)
                claim_to_gt[claim_id] = event_name

            # Create identity edges within same fact
            for i in range(len(fact_claims) - 1):
                identity_edges.append((
                    fact_claims[i],
                    fact_claims[i + 1],
                    Relation.CONFIRMS,
                    0.9
                ))

    return gt_claims, identity_edges, claim_to_gt


# =============================================================================
# A) HARD INVARIANTS - Must Always Hold
# =============================================================================

class TestHardInvariants:
    """Invariants that must hold regardless of data or parameters."""

    @pytest.mark.asyncio
    async def test_identity_aboutness_separation(self):
        """
        CORE INVARIANT: Changing aboutness params must NEVER change surfaces.

        This is the fundamental layer separation principle:
        - Surfaces (L2) are determined solely by identity edges
        - Aboutness params affect only event formation (L3)
        """
        # Setup claims and identity edges
        claims = [
            make_claim("c1", "EventA", "fact1", "s1.com"),
            make_claim("c2", "EventA", "fact1", "s2.com"),
            make_claim("c3", "EventB", "fact2", "s3.com"),
        ]
        identity_edges = [("c1", "c2", Relation.CONFIRMS, 0.9)]

        async def get_surfaces(aboutness_threshold: float, aboutness_min_signals: int):
            """Build fresh engine and return surface view."""
            e = Engine(params=Parameters(
                aboutness_threshold=aboutness_threshold,
                aboutness_min_signals=aboutness_min_signals
            ))
            for c in claims:
                await e.add_claim(c)
            e.claim_edges.extend(identity_edges)
            e.compute_surfaces()
            return surface_view(e)

        # Vary aboutness params across extreme range
        surfaces_a = await get_surfaces(0.1, 1)
        surfaces_b = await get_surfaces(0.5, 3)
        surfaces_c = await get_surfaces(0.9, 10)

        # All must be IDENTICAL
        assert surfaces_a == surfaces_b == surfaces_c, \
            "Aboutness params changed surface membership - layer separation violated!"

    @pytest.mark.asyncio
    async def test_surface_internal_edges_are_identity_only(self):
        """Surface.internal_edges must only contain Relation (identity) types."""
        e = Engine(params=Parameters(aboutness_min_signals=1))

        c1 = make_claim("c1", "EventA", "f1", "s1.com", {"EventA", "shared"})
        c2 = make_claim("c2", "EventA", "f2", "s2.com", {"EventA", "shared"})

        await e.add_claim(c1)
        await e.add_claim(c2)
        e.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))

        e.compute_surfaces()
        e.compute_surface_aboutness()

        for s in e.surfaces.values():
            for edge in s.internal_edges:
                assert isinstance(edge[2], Relation), \
                    f"Non-identity edge in surface: {edge}"

    @pytest.mark.asyncio
    async def test_aboutness_links_are_not_relations(self):
        """Surface.about_links must be AboutnessLink, not Relation."""
        e = Engine(params=Parameters(aboutness_min_signals=1))

        c1 = make_claim("c1", "EventA", "f1", "s1.com", {"EventA", "shared"})
        c2 = make_claim("c2", "EventA", "f2", "s2.com", {"EventA", "shared"})

        await e.add_claim(c1)
        await e.add_claim(c2)

        e.compute_surfaces()
        e.compute_surface_aboutness()

        for s in e.surfaces.values():
            for link in s.about_links:
                assert hasattr(link, 'score'), "about_link missing 'score' attribute"
                assert not isinstance(link, Relation), \
                    "Relation type found in about_links - layer mixing!"

    @pytest.mark.asyncio
    async def test_unrelated_claim_cannot_change_existing_surfaces(self):
        """Adding unrelated claim must not change membership of existing surfaces."""
        # Build initial engine
        c1 = make_claim("c1", "EventA", "fact1", "s1.com")
        c2 = make_claim("c2", "EventA", "fact1", "s2.com")
        identity_edges = [("c1", "c2", Relation.CONFIRMS, 0.9)]

        async def build_engine(claims_list, edges):
            e = Engine(params=Parameters())
            for c in claims_list:
                await e.add_claim(c)
            e.claim_edges.extend(edges)
            e.compute_surfaces()
            return e

        # Engine without unrelated claim
        e1 = await build_engine([c1, c2], identity_edges)
        surfaces_before = surface_view(e1)

        # Engine with unrelated claim added
        unrelated = make_claim("unrelated", "TotallyDifferent", "otherfact", "other.com")
        e2 = await build_engine([c1, c2, unrelated], identity_edges)

        # Find original surface
        original_surface = None
        for s in e2.surfaces.values():
            if "c1" in s.claim_ids:
                original_surface = frozenset(s.claim_ids)
                break

        # Original claims must still be together
        assert original_surface is not None
        assert "c1" in original_surface
        assert "c2" in original_surface
        assert "unrelated" not in original_surface

        # Original surface structure preserved
        assert frozenset({"c1", "c2"}) in surface_view(e2)


# =============================================================================
# B) METAMORPHIC STABILITY - Structure Equivalence Under Permutations
# =============================================================================

class TestMetamorphicStability:
    """Tests that structure is invariant under claim ordering permutations."""

    @pytest.mark.asyncio
    async def test_order_invariance_surfaces(self):
        """Surface structure must be identical regardless of claim arrival order."""
        claims = [
            make_claim("c1", "EventA", "fact1", "src1.com"),
            make_claim("c2", "EventA", "fact1", "src2.com"),
            make_claim("c3", "EventA", "fact2", "src3.com"),
            make_claim("c4", "EventB", "fact3", "src4.com"),
            make_claim("c5", "EventB", "fact3", "src5.com"),
        ]
        identity_edges = [
            ("c1", "c2", Relation.CONFIRMS, 0.9),
            ("c4", "c5", Relation.CONFIRMS, 0.9),
        ]

        async def build_with_order(claim_order):
            e = Engine(params=Parameters())
            for c in claim_order:
                await e.add_claim(c)
            e.claim_edges.extend(identity_edges)
            e.compute_surfaces()
            return surface_view(e)

        # Original order
        surfaces_original = await build_with_order(claims)

        # Reversed order
        surfaces_reversed = await build_with_order(list(reversed(claims)))

        # Random shuffle (fixed seed for reproducibility)
        shuffled = claims.copy()
        random.seed(42)
        random.shuffle(shuffled)
        surfaces_shuffled = await build_with_order(shuffled)

        # All must be identical
        assert surfaces_original == surfaces_reversed == surfaces_shuffled

    @pytest.mark.asyncio
    async def test_order_invariance_events(self):
        """Event structure (by claims) must be identical regardless of order."""
        claims = [
            make_claim("c1", "EventA", "fact1", "src1.com", {"EventA", "shared_entity"}),
            make_claim("c2", "EventA", "fact2", "src2.com", {"EventA", "shared_entity"}),
            make_claim("c3", "EventB", "fact3", "src3.com", {"EventB", "other_entity"}),
        ]

        async def build_with_order(claim_order):
            e = Engine(params=Parameters(aboutness_min_signals=1))
            for c in claim_order:
                await e.add_claim(c)
            e.compute_surfaces()
            e.compute_surface_aboutness()
            e.compute_events()
            return event_view(e)

        events_original = await build_with_order(claims)
        events_reversed = await build_with_order(list(reversed(claims)))

        # Same claim groupings
        assert events_original == events_reversed

    @pytest.mark.asyncio
    async def test_multiple_shuffles_same_result(self):
        """Multiple random shuffles should all produce same structure."""
        gt_claims, identity_edges, _ = generate_controlled_dataset(
            n_events=2, n_facts_per_event=2, n_sources_per_fact=2
        )
        claims = [gc.claim for gc in gt_claims]

        async def build_with_order(claim_order):
            e = Engine(params=Parameters(aboutness_min_signals=1))
            for c in claim_order:
                await e.add_claim(c)
            e.claim_edges.extend(identity_edges)
            e.compute_surfaces()
            return surface_view(e)

        baseline = await build_with_order(claims)

        # Test 5 different shuffles
        for seed in [1, 17, 42, 99, 256]:
            shuffled = claims.copy()
            random.seed(seed)
            random.shuffle(shuffled)
            result = await build_with_order(shuffled)
            assert result == baseline, f"Shuffle with seed {seed} produced different result"


# =============================================================================
# C) RANGE LAWS - Monotonicity and Bounds Across Parameter Sweeps
# =============================================================================

class TestRangeLaws:
    """Tests that assert monotonicity and bounds across parameter ranges."""

    @pytest.mark.asyncio
    async def test_threshold_monotonicity_event_count(self):
        """
        Higher aboutness_threshold → fewer merges → event count non-decreasing.

        Theory: Stricter threshold means fewer aboutness edges pass filter,
        so events fragment more (or stay same), never consolidate more.
        """
        gt_claims, identity_edges, claim_to_gt = generate_controlled_dataset(
            n_events=3, n_facts_per_event=2, n_sources_per_fact=2
        )
        claims = [gc.claim for gc in gt_claims]

        async def count_events(threshold: float) -> int:
            e = Engine(params=Parameters(
                aboutness_threshold=threshold,
                aboutness_min_signals=1
            ))
            for c in claims:
                await e.add_claim(c)
            e.claim_edges.extend(identity_edges)
            e.compute_surfaces()
            e.compute_surface_aboutness()
            e.compute_events()
            return len(e.events)

        # Sweep thresholds (low to high)
        events_at_01 = await count_events(0.1)
        events_at_03 = await count_events(0.3)
        events_at_05 = await count_events(0.5)
        events_at_07 = await count_events(0.7)

        # Non-decreasing (more fragmentation with higher threshold)
        assert events_at_01 <= events_at_03 <= events_at_05 <= events_at_07, \
            f"Event count not monotonic: {events_at_01} → {events_at_03} → {events_at_05} → {events_at_07}"

    @pytest.mark.asyncio
    async def test_min_signals_monotonicity_mixing(self):
        """
        Higher min_signals → fewer cross-event links → mixing non-increasing.

        Theory: Requiring more signals to form aboutness edge means
        spurious single-signal bridges get filtered out.
        """
        # Create dataset with weak cross-event signal
        claims = [
            # Event A
            make_claim("a1", "EventA", "f1", "s1.com", {"EventA", "UniqueA", "WeakBridge"}),
            make_claim("a2", "EventA", "f2", "s2.com", {"EventA", "UniqueA"}),
            # Event B
            make_claim("b1", "EventB", "f3", "s3.com", {"EventB", "UniqueB", "WeakBridge"}),
            make_claim("b2", "EventB", "f4", "s4.com", {"EventB", "UniqueB"}),
        ]
        claim_to_gt = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}

        async def get_mixing(min_signals: int) -> float:
            e = Engine(params=Parameters(aboutness_min_signals=min_signals))
            for c in claims:
                await e.add_claim(c)
            e.compute_surfaces()
            e.compute_surface_aboutness()
            e.compute_events()
            return compute_mixing_rate(e, claim_to_gt)

        mixing_1 = await get_mixing(1)
        mixing_2 = await get_mixing(2)
        mixing_3 = await get_mixing(3)

        # Non-increasing mixing
        assert mixing_1 >= mixing_2 >= mixing_3, \
            f"Mixing not monotonic: {mixing_1:.2f} → {mixing_2:.2f} → {mixing_3:.2f}"

    @pytest.mark.asyncio
    async def test_hub_suppression_bounds_mixing(self):
        """
        Hub entity suppression should keep mixing bounded even under high hubness.

        Theory: With hub_max_df filtering, high-frequency entities don't create
        spurious cross-event links. Without it, they would bridge unrelated events.
        """
        random.seed(42)

        # Generate dataset with aggressive hub injection
        gt_claims, identity_edges, claim_to_gt = generate_controlled_dataset(
            n_events=3,
            n_facts_per_event=2,
            n_sources_per_fact=2,
            hub_entity="HongKong",  # Hub entity
            hub_rate=0.8,           # 80% of claims mention it
        )
        claims = [gc.claim for gc in gt_claims]

        async def get_mixing(hub_max_df: int) -> float:
            e = Engine(params=Parameters(
                hub_max_df=hub_max_df,
                aboutness_min_signals=1,
            ))
            for c in claims:
                await e.add_claim(c)
            e.claim_edges.extend(identity_edges)
            e.compute_surfaces()
            e.compute_surface_aboutness()
            e.compute_events()
            return compute_mixing_rate(e, claim_to_gt)

        # No suppression (high hub_max_df allows hub)
        mixing_no_suppress = await get_mixing(100)

        # Strong suppression (low hub_max_df filters hub)
        mixing_with_suppress = await get_mixing(2)

        # Suppression should reduce mixing (or at worst keep it same)
        assert mixing_with_suppress <= mixing_no_suppress, \
            f"Hub suppression increased mixing: {mixing_no_suppress:.2f} → {mixing_with_suppress:.2f}"

    @pytest.mark.asyncio
    async def test_hub_entity_filtered_above_threshold(self):
        """Entities appearing in > hub_max_df claims should not create aboutness edges."""
        hub_max_df = 2

        # 4 claims from different events, all mentioning "HubEntity"
        claims = [
            make_claim(f"c{i}", f"Event{i}", f"fact{i}", f"src{i}.com",
                      {f"Event{i}", "HubEntity"}, f"Event{i}")
            for i in range(4)
        ]

        e = Engine(params=Parameters(hub_max_df=hub_max_df, aboutness_min_signals=1))
        for c in claims:
            await e.add_claim(c)

        e.compute_surfaces()
        aboutness = e.compute_surface_aboutness()

        # HubEntity appears in 4 claims (> hub_max_df=2)
        # No aboutness edges should form (only shared signal is the hub)
        assert len(aboutness) == 0, \
            f"Hub entity created {len(aboutness)} edges despite exceeding threshold"

    @pytest.mark.asyncio
    async def test_rare_entity_creates_aboutness(self):
        """Entities below hub_max_df should create aboutness edges."""
        hub_max_df = 10

        # 2 claims sharing "RareEntity" (below threshold)
        claims = [
            make_claim("c1", "EventA", "f1", "s1.com", {"EventA", "RareEntity"}, "EventA"),
            make_claim("c2", "EventA", "f2", "s2.com", {"EventA", "RareEntity"}, "EventA"),
        ]

        e = Engine(params=Parameters(hub_max_df=hub_max_df, aboutness_min_signals=1))
        for c in claims:
            await e.add_claim(c)

        e.compute_surfaces()
        aboutness = e.compute_surface_aboutness()

        # RareEntity appears in 2 claims (≤ hub_max_df=10), should create edge
        assert len(aboutness) >= 1, "Rare entity failed to create aboutness edge"


# =============================================================================
# D) DUPLICATION INVARIANTS - Confirmation Behavior
# =============================================================================

class TestDuplicationInvariants:
    """Tests that confirming claims join surfaces, not create new ones."""

    @pytest.mark.asyncio
    async def test_confirmation_joins_surface(self):
        """Confirming claim should join existing surface, not create new one."""
        c1 = make_claim("c1", "EventA", "death_count", "src1.com")
        c2 = make_claim("c2", "EventA", "death_count", "src2.com")

        e = Engine(params=Parameters())
        await e.add_claim(c1)
        await e.add_claim(c2)
        e.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.95))

        e.compute_surfaces()

        # Should be one surface with both claims
        assert len(e.surfaces) == 1
        surface = list(e.surfaces.values())[0]
        assert len(surface.claim_ids) == 2
        assert {"c1", "c2"} == set(surface.claim_ids)

    @pytest.mark.asyncio
    async def test_chain_confirmation_single_surface(self):
        """Chain of confirmations should all merge into one surface."""
        claims = [make_claim(f"c{i}", "EventA", "fact1", f"src{i}.com") for i in range(5)]

        e = Engine(params=Parameters())
        for c in claims:
            await e.add_claim(c)

        # Create chain: c0 → c1 → c2 → c3 → c4
        for i in range(4):
            e.claim_edges.append((f"c{i}", f"c{i+1}", Relation.CONFIRMS, 0.9))

        e.compute_surfaces()

        # All should be in one surface
        assert len(e.surfaces) == 1
        surface = list(e.surfaces.values())[0]
        assert len(surface.claim_ids) == 5
        assert len(surface.sources) == 5  # All different sources

    @pytest.mark.asyncio
    async def test_star_confirmation_single_surface(self):
        """Star topology (one central claim confirmed by many) should merge."""
        central = make_claim("central", "EventA", "fact1", "central.com")
        satellites = [make_claim(f"sat{i}", "EventA", "fact1", f"sat{i}.com") for i in range(4)]

        e = Engine(params=Parameters())
        await e.add_claim(central)
        for s in satellites:
            await e.add_claim(s)

        # All satellites confirm central
        for i in range(4):
            e.claim_edges.append(("central", f"sat{i}", Relation.CONFIRMS, 0.9))

        e.compute_surfaces()

        # All should be in one surface
        assert len(e.surfaces) == 1
        surface = list(e.surfaces.values())[0]
        assert len(surface.claim_ids) == 5


# =============================================================================
# E) LAYER SEPARATION - Events From Aboutness Only
# =============================================================================

class TestLayerSeparation:
    """Tests that events form from aboutness, not identity edges."""

    @pytest.mark.asyncio
    async def test_different_facts_same_event_via_aboutness(self):
        """
        Two different facts about same event should:
        - Be separate surfaces (no identity edge)
        - Be grouped into one event (via aboutness)
        """
        # Two facts about EventA, no identity relationship
        c1 = make_claim("c1", "EventA", "death_toll", "s1.com", {"EventA", "shared_topic"})
        c2 = make_claim("c2", "EventA", "cause", "s2.com", {"EventA", "shared_topic"})

        e = Engine(params=Parameters(aboutness_min_signals=1))
        await e.add_claim(c1)
        await e.add_claim(c2)

        # No identity edges (different facts)
        e.compute_surfaces()
        assert len(e.surfaces) == 2, "Different facts should be separate surfaces"

        e.compute_surface_aboutness()
        e.compute_events()

        # Should merge into one event via aboutness
        assert len(e.events) == 1, "Same-event facts should form one event"
        event = list(e.events.values())[0]
        assert len(event.surface_ids) == 2

    @pytest.mark.asyncio
    async def test_unrelated_events_stay_separate(self):
        """Claims about different events should not merge even with low threshold."""
        c1 = make_claim("c1", "EventA", "f1", "s1.com", {"EventA", "UniqueA"})
        c2 = make_claim("c2", "EventB", "f2", "s2.com", {"EventB", "UniqueB"})

        e = Engine(params=Parameters(
            aboutness_threshold=0.01,  # Very low threshold
            aboutness_min_signals=1
        ))
        await e.add_claim(c1)
        await e.add_claim(c2)

        e.compute_surfaces()
        e.compute_surface_aboutness()
        e.compute_events()

        # Should be 2 separate events (no shared entities)
        assert len(e.events) == 2, "Unrelated events merged despite no shared signals"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
