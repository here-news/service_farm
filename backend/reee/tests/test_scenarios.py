"""
Scenario Tests: "Perfect" vs "Imperfect" Evidence
==================================================

These tests are intentionally synthetic and deterministic (no LLM required).
They validate end-to-end REEE behavior:

- Tier-1 (identity): q1/q2 buckets + same-fact relations
- L2 Surfaces: connected components of identity edges only
- Tier-2 (aboutness): surface-to-surface associations form events without
  contaminating surfaces
"""

import pytest

from reee import Engine, Claim, Parameters, Relation


@pytest.fixture
def engine_strict_aboutness():
    """
    Engine with default (strict) aboutness settings.

    We keep aboutness_min_signals=2 to avoid accidental cross-event merges in tests.
    """
    return Engine(params=Parameters(aboutness_min_signals=2, aboutness_threshold=0.15))


class TestTier1Identity:
    @pytest.mark.asyncio
    async def test_same_fact_updates_and_conflicts_form_one_surface(self, engine_strict_aboutness):
        """
        "Perfect" identity case: same question_key bucket + shared event anchor.

        Expected:
        - L0->L2 identity edges created (CONFIRMS/SUPERSEDES/CONFLICTS)
        - L2 surfaces: all claims about the same fact are in one surface
        """
        e = engine_strict_aboutness

        c1 = Claim(
            id="c1",
            text="2 people were killed in the shooting at Brown University Library.",
            source="source-a",
            entities={"Brown University Library", "shooting", "killed"},
            anchor_entities={"Brown University Library"},
        )
        c2 = Claim(
            id="c2",
            text="Death toll rises to 3 after the shooting at Brown University Library.",
            source="source-b",
            entities={"Brown University Library", "shooting", "death toll"},
            anchor_entities={"Brown University Library"},
        )
        c3 = Claim(
            id="c3",
            text="2 people were killed in the Brown University Library shooting.",
            source="source-c",
            entities={"Brown University Library", "shooting", "killed"},
            anchor_entities={"Brown University Library"},
        )
        c4 = Claim(
            id="c4",
            text="4 people were killed in the shooting at Brown University Library.",
            source="source-d",
            entities={"Brown University Library", "shooting", "killed"},
            anchor_entities={"Brown University Library"},
        )

        for claim in (c1, c2, c3, c4):
            await e.add_claim(claim, extract_question_key=True)

        surfaces = e.compute_surfaces()

        # All claims should belong to exactly one surface in total.
        assert sum(len(s.claim_ids) for s in surfaces) == 4

        # We expect exactly one multi-claim surface for this same-fact cluster.
        multi = [s for s in surfaces if len(s.claim_ids) > 1]
        assert len(multi) == 1
        surface = multi[0]
        assert surface.claim_ids == {"c1", "c2", "c3", "c4"}

        # Internal edges should include at least one supersedes and one conflict.
        rels = {rel for _, _, rel in surface.internal_edges}
        assert Relation.SUPERSEDES in rels
        assert Relation.CONFLICTS in rels

    @pytest.mark.asyncio
    async def test_q1q2_bucket_does_not_conflate_distinct_events(self, engine_strict_aboutness):
        """
        "Imperfect" / confounder case:
        Two different events can share the same question_key (death_count).

        Expected:
        - q1/q2 bucket is NOT sufficient to link; anchor/entity check blocks it.
        - No identity edge is created between the two claims.
        - Surfaces remain separate.
        """
        e = engine_strict_aboutness

        # Event A: Brown shooting.
        a = Claim(
            id="a",
            text="2 people were killed at Brown University Library.",
            source="source-a",
            entities={"Brown University Library", "killed"},
            anchor_entities={"Brown University Library"},
        )
        # Event B: Wang Fuk Court fire.
        b = Claim(
            id="b",
            text="2 people were killed in the fire at Wang Fuk Court.",
            source="source-b",
            entities={"Wang Fuk Court", "fire", "killed"},
            anchor_entities={"Wang Fuk Court"},
        )

        await e.add_claim(a, extract_question_key=True)
        await e.add_claim(b, extract_question_key=True)

        # Same question_key likely extracted for both (death_count),
        # but they must not relate due to anchor mismatch.
        assert a.question_key == "death_count"
        assert b.question_key == "death_count"

        # No identity edges should exist between these two claims.
        assert not any(
            {c1, c2} == {"a", "b"} and rel != Relation.UNRELATED
            for (c1, c2, rel, _conf) in e.claim_edges
        )

        surfaces = e.compute_surfaces()
        # Two claims, two singleton surfaces.
        assert len(surfaces) == 2
        assert all(len(s.claim_ids) == 1 for s in surfaces)


class TestTier2Aboutness:
    @pytest.mark.asyncio
    async def test_aboutness_groups_aspects_into_one_event_without_merging_surfaces(self, engine_strict_aboutness):
        """
        "Perfect" aboutness case:
        Two different aspects of the same event should remain separate surfaces
        (different facts) but be grouped into one event.

        We avoid embeddings/LLM by providing two independent aboutness signals:
        - shared anchor: "Jimmy Lai"
        - shared non-anchor entity: "National Security Law"
        """
        e = engine_strict_aboutness

        s1 = Claim(
            id="j1",
            text="Jimmy Lai has been detained for almost five years.",
            source="source-a",
            entities={"Jimmy Lai", "National Security Law"},
            anchor_entities={"Jimmy Lai"},
            question_key="detention_duration",
            extracted_value="~5y",
        )
        s2 = Claim(
            id="j2",
            text="Amnesty International condemned Jimmy Lai’s conviction.",
            source="source-b",
            entities={"Jimmy Lai", "Amnesty International", "National Security Law"},
            anchor_entities={"Jimmy Lai"},
            question_key="advocacy_reaction",
            extracted_value="condemned",
        )

        await e.add_claim(s1, extract_question_key=False)
        await e.add_claim(s2, extract_question_key=False)

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 2
        assert all(len(s.claim_ids) == 1 for s in surfaces)

        aboutness = e.compute_surface_aboutness()
        assert len(aboutness) >= 1

        events = e.compute_events()
        # Both singleton surfaces should be grouped into one event.
        assert len(events) == 1
        assert next(iter(events)).surface_ids == set(e.surfaces.keys())

    @pytest.mark.asyncio
    async def test_aboutness_does_not_link_unrelated_events(self, engine_strict_aboutness):
        """
        Two events with no shared anchors or entities should remain separate.
        """
        e = engine_strict_aboutness

        # Event A: Hong Kong fire
        a = Claim(
            id="hk1",
            text="Fire broke out at Wang Fuk Court in Hong Kong.",
            source="source-a",
            entities={"Wang Fuk Court", "Hong Kong", "fire"},
            anchor_entities={"Wang Fuk Court"},
        )
        # Event B: US election
        b = Claim(
            id="us1",
            text="Biden addresses climate policy in Washington.",
            source="source-b",
            entities={"Biden", "Washington", "climate policy"},
            anchor_entities={"Biden"},
        )

        await e.add_claim(a)
        await e.add_claim(b)

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 2

        aboutness = e.compute_surface_aboutness()
        # No aboutness edges between unrelated events
        assert len(aboutness) == 0

        events = e.compute_events()
        # Two separate events
        assert len(events) == 2


class TestTier3MetaClaims:
    """Meta-claim detection scenarios."""

    @pytest.mark.asyncio
    async def test_single_source_tension_detected(self):
        """Single-source surfaces should trigger meta-claims."""
        e = Engine(params=Parameters())

        c1 = Claim(
            id="c1",
            text="Breaking: Major announcement expected.",
            source="only-source.com",
            entities={"announcement"},
            anchor_entities=set(),
        )
        await e.add_claim(c1)

        e.compute_surfaces()
        meta_claims = e.detect_tensions()

        single_source = [mc for mc in meta_claims if mc.type == 'single_source_only']
        assert len(single_source) == 1
        assert single_source[0].target_type == 'surface'

    @pytest.mark.asyncio
    async def test_conflict_tension_detected(self):
        """Conflicting claims should trigger unresolved_conflict meta-claims."""
        e = Engine(params=Parameters())

        c1 = Claim(
            id="c1",
            text="3 people were killed in the incident.",
            source="source-a",
            entities={"incident", "killed"},
            anchor_entities={"incident location"},
            question_key="death_count",
        )
        c2 = Claim(
            id="c2",
            text="5 people were killed in the incident.",
            source="source-b",
            entities={"incident", "killed"},
            anchor_entities={"incident location"},
            question_key="death_count",
        )

        await e.add_claim(c1)
        await e.add_claim(c2)

        # Manually add conflict edge (simulating LLM classification)
        e.claim_edges.append(("c1", "c2", Relation.CONFLICTS, 0.9))

        e.compute_surfaces()
        meta_claims = e.detect_tensions()

        conflicts = [mc for mc in meta_claims if mc.type == 'unresolved_conflict']
        assert len(conflicts) >= 1
        assert conflicts[0].evidence.get('confidence') == 0.9


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_engine(self):
        """Engine with no claims should handle gracefully."""
        e = Engine(params=Parameters())

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 0

        aboutness = e.compute_surface_aboutness()
        assert len(aboutness) == 0

        events = e.compute_events()
        assert len(events) == 0

        meta_claims = e.detect_tensions()
        assert len(meta_claims) == 0

    @pytest.mark.asyncio
    async def test_single_claim(self):
        """Single claim should create one surface and one event."""
        e = Engine(params=Parameters())

        c1 = Claim(
            id="solo",
            text="A single claim about something.",
            source="source.com",
            entities={"something"},
            anchor_entities={"something"},
        )
        await e.add_claim(c1)

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1
        assert len(list(surfaces)[0].claim_ids) == 1

        e.compute_surface_aboutness()
        events = e.compute_events()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_duplicate_claim_id_overwrites(self):
        """Adding a claim with duplicate ID overwrites (last-write-wins)."""
        e = Engine(params=Parameters())

        c1 = Claim(id="dup", text="First claim", source="a.com", entities=set(), anchor_entities=set())
        c2 = Claim(id="dup", text="Second claim", source="b.com", entities=set(), anchor_entities=set())

        await e.add_claim(c1)
        await e.add_claim(c2)

        # Should only have one claim (latest version)
        assert len(e.claims) == 1
        assert e.claims["dup"].text == "Second claim"

    @pytest.mark.asyncio
    async def test_claim_without_entities(self):
        """Claims without entities should still form surfaces."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="No entities here.", source="a.com", entities=set(), anchor_entities=set())
        c2 = Claim(id="c2", text="Also no entities.", source="b.com", entities=set(), anchor_entities=set())

        await e.add_claim(c1)
        await e.add_claim(c2)

        surfaces = e.compute_surfaces()
        # Each claim in its own surface (no identity edges without entities)
        assert len(surfaces) == 2

    @pytest.mark.asyncio
    async def test_hub_entity_does_not_create_spurious_links(self):
        """Hub entities (high document frequency) should not link unrelated events."""
        e = Engine(params=Parameters(hub_max_df=2))

        # Three claims all mentioning "Hong Kong" (hub entity)
        claims = [
            Claim(
                id=f"c{i}",
                text=f"Event {i} in Hong Kong",
                source=f"src{i}.com",
                entities={"Hong Kong", f"unique_entity_{i}"},
                anchor_entities={f"unique_entity_{i}"},
            )
            for i in range(3)
        ]

        for c in claims:
            await e.add_claim(c)

        surfaces = e.compute_surfaces()
        # Each claim should be in its own surface (no identity from shared hub)
        assert len(surfaces) == 3

        aboutness = e.compute_surface_aboutness()
        # Hub entity "Hong Kong" appears in all 3, exceeding hub_max_df=2
        # Should not create aboutness edges
        assert len(aboutness) == 0


class TestRelationSemantics:
    """Tests for proper relation handling."""

    @pytest.mark.asyncio
    async def test_confirms_creates_identity_edge(self):
        """CONFIRMS relation should merge claims into same surface."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="Claim one", source="a.com", entities={"X"}, anchor_entities={"X"})
        c2 = Claim(id="c2", text="Claim two", source="b.com", entities={"X"}, anchor_entities={"X"})

        await e.add_claim(c1)
        await e.add_claim(c2)

        e.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.95))

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1
        assert surfaces[0].claim_ids == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_refines_creates_identity_edge(self):
        """REFINES relation should merge claims into same surface."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="Initial report", source="a.com", entities={"Y"}, anchor_entities={"Y"})
        c2 = Claim(id="c2", text="Updated report with details", source="b.com", entities={"Y"}, anchor_entities={"Y"})

        await e.add_claim(c1)
        await e.add_claim(c2)

        e.claim_edges.append(("c1", "c2", Relation.REFINES, 0.85))

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1

    @pytest.mark.asyncio
    async def test_supersedes_creates_identity_edge(self):
        """SUPERSEDES relation should merge claims into same surface."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="Old info", source="a.com", entities={"Z"}, anchor_entities={"Z"})
        c2 = Claim(id="c2", text="New corrected info", source="b.com", entities={"Z"}, anchor_entities={"Z"})

        await e.add_claim(c1)
        await e.add_claim(c2)

        e.claim_edges.append(("c1", "c2", Relation.SUPERSEDES, 0.90))

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1

    @pytest.mark.asyncio
    async def test_conflicts_creates_identity_edge_but_flags_tension(self):
        """CONFLICTS relation should merge into same surface AND create meta-claim."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="Says 2 dead", source="a.com", entities={"W"}, anchor_entities={"W"})
        c2 = Claim(id="c2", text="Says 5 dead", source="b.com", entities={"W"}, anchor_entities={"W"})

        await e.add_claim(c1)
        await e.add_claim(c2)

        e.claim_edges.append(("c1", "c2", Relation.CONFLICTS, 0.88))

        surfaces = e.compute_surfaces()
        # Conflicts still form same surface (same fact, different values)
        assert len(surfaces) == 1

        meta_claims = e.detect_tensions()
        conflicts = [mc for mc in meta_claims if mc.type == 'unresolved_conflict']
        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    async def test_unrelated_does_not_create_edge(self):
        """UNRELATED relation should NOT merge claims."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="Topic A", source="a.com", entities={"A"}, anchor_entities={"A"})
        c2 = Claim(id="c2", text="Topic B", source="b.com", entities={"B"}, anchor_entities={"B"})

        await e.add_claim(c1)
        await e.add_claim(c2)

        # Explicitly mark as unrelated
        e.claim_edges.append(("c1", "c2", Relation.UNRELATED, 0.99))

        surfaces = e.compute_surfaces()
        # Should remain separate
        assert len(surfaces) == 2


class TestMultiSourceCorroboration:
    """Tests for source diversity and corroboration."""

    @pytest.mark.asyncio
    async def test_multi_source_surface_no_single_source_tension(self):
        """Surface with multiple sources should NOT trigger single_source tension."""
        e = Engine(params=Parameters())

        c1 = Claim(id="c1", text="Fact X", source="bbc.com", entities={"X"}, anchor_entities={"X"})
        c2 = Claim(id="c2", text="Fact X confirmed", source="reuters.com", entities={"X"}, anchor_entities={"X"})
        c3 = Claim(id="c3", text="Fact X verified", source="ap.com", entities={"X"}, anchor_entities={"X"})

        await e.add_claim(c1)
        await e.add_claim(c2)
        await e.add_claim(c3)

        # Link them as confirming
        e.claim_edges.append(("c1", "c2", Relation.CONFIRMS, 0.9))
        e.claim_edges.append(("c2", "c3", Relation.CONFIRMS, 0.9))

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1
        assert surfaces[0].sources == {"bbc.com", "reuters.com", "ap.com"}

        meta_claims = e.detect_tensions()
        single_source = [mc for mc in meta_claims if mc.type == 'single_source_only']
        # Should NOT have single-source tension
        assert len(single_source) == 0

    @pytest.mark.asyncio
    async def test_surface_tracks_all_sources(self):
        """Surface should aggregate sources from all constituent claims."""
        e = Engine(params=Parameters())

        sources = ["src1.com", "src2.com", "src3.com", "src4.com"]
        for i, src in enumerate(sources):
            c = Claim(id=f"c{i}", text=f"Claim {i}", source=src, entities={"shared"}, anchor_entities={"shared"})
            await e.add_claim(c)

        # Link all as confirming
        for i in range(len(sources) - 1):
            e.claim_edges.append((f"c{i}", f"c{i+1}", Relation.CONFIRMS, 0.9))

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1
        assert surfaces[0].sources == set(sources)


class TestChainedRelations:
    """Tests for transitive closure of identity relations."""

    @pytest.mark.asyncio
    async def test_transitive_identity_closure(self):
        """Identity edges should form transitive closure (A→B, B→C implies A,B,C same surface)."""
        e = Engine(params=Parameters())

        for i in range(5):
            c = Claim(id=f"c{i}", text=f"Claim {i}", source=f"src{i}.com", entities={"chain"}, anchor_entities={"chain"})
            await e.add_claim(c)

        # Chain: c0→c1→c2→c3→c4
        for i in range(4):
            e.claim_edges.append((f"c{i}", f"c{i+1}", Relation.CONFIRMS, 0.9))

        surfaces = e.compute_surfaces()
        # All 5 claims should be in one surface
        assert len(surfaces) == 1
        assert len(surfaces[0].claim_ids) == 5

    @pytest.mark.asyncio
    async def test_branching_identity_graph(self):
        """Branching identity graph should still form one surface."""
        e = Engine(params=Parameters())

        # Star topology: center connected to 4 peripherals
        center = Claim(id="center", text="Center", source="a.com", entities={"star"}, anchor_entities={"star"})
        await e.add_claim(center)

        for i in range(4):
            c = Claim(id=f"p{i}", text=f"Peripheral {i}", source=f"src{i}.com", entities={"star"}, anchor_entities={"star"})
            await e.add_claim(c)
            e.claim_edges.append(("center", f"p{i}", Relation.CONFIRMS, 0.9))

        surfaces = e.compute_surfaces()
        assert len(surfaces) == 1
        assert len(surfaces[0].claim_ids) == 5
