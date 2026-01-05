"""
Test: Scoped Surface Invariant
==============================

This test validates the CRITICAL invariant that L2 surfaces must be
scoped by referent, not just by predicate (question_key).

THE INVARIANT:
- L2 surface key = (scope_signature, question_key)
- NOT just question_key alone

THE BUG THIS CATCHES:
- If L2 surface key = question_key only (global predicate bucket)
- Unrelated incidents share surfaces via generic question_keys
- Causes cross-event contamination

EXPECTED BEHAVIOR:
This test should FAIL until the surface scoping refactor is complete.
Once fixed, the test validates that the invariant holds.

Usage:
    pytest backend/reee/tests/test_scoped_surface.py -v
"""

import pytest
from pathlib import Path
from collections import defaultdict

from reee.tests.test_golden_trace import GoldenTrace, TraceKernel


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestScopedSurfaceInvariant:
    """
    Test that surfaces are scoped by referent, not just question_key.

    This test is expected to FAIL with the current implementation
    (which uses global question_key -> surface mapping).

    Once the refactor to (scope, question_key) is complete,
    this test should PASS.
    """

    @pytest.fixture
    def trace(self):
        """Load the scoped surface golden trace."""
        trace_path = FIXTURES_DIR / "golden_trace_scoped_surface.yaml"
        return GoldenTrace.load(trace_path)

    @pytest.fixture
    def kernel(self, trace):
        """Create kernel and process all claims."""
        kernel = TraceKernel(trace)
        for tc in trace.claims:
            kernel.process_claim(tc)
        return kernel

    def test_separate_scopes_produce_separate_surfaces(self, kernel, trace):
        """
        CRITICAL INVARIANT: Different scopes must produce different surfaces,
        even when question_key is identical.

        Hong Kong claims (HK_C1, HK_C2) and California claims (CA_C1, CA_C2)
        all have question_key='policy_announcement', but they must be in
        SEPARATE surfaces because they have different referents.
        """
        # Get surfaces for each claim group
        hk_claims = {"HK_C1", "HK_C2"}
        ca_claims = {"CA_C1", "CA_C2"}

        hk_surfaces = {
            kernel.claim_to_surface.get(cid)
            for cid in hk_claims
            if cid in kernel.claim_to_surface
        }
        ca_surfaces = {
            kernel.claim_to_surface.get(cid)
            for cid in ca_claims
            if cid in kernel.claim_to_surface
        }

        # Remove None if any claims weren't processed
        hk_surfaces.discard(None)
        ca_surfaces.discard(None)

        # THE INVARIANT: No overlap between HK and CA surfaces
        shared_surfaces = hk_surfaces & ca_surfaces

        assert len(shared_surfaces) == 0, (
            f"INVARIANT VIOLATED: Hong Kong and California claims share surfaces!\n"
            f"  HK surfaces: {hk_surfaces}\n"
            f"  CA surfaces: {ca_surfaces}\n"
            f"  Shared: {shared_surfaces}\n"
            f"\n"
            f"This means L2 is using global question_key buckets instead of\n"
            f"scoped surfaces. The surface key must be (scope, question_key),\n"
            f"not question_key alone."
        )

    def test_question_key_alone_insufficient(self, kernel, trace):
        """
        Same question_key should produce MULTIPLE surfaces if scopes differ.

        With question_key='policy_announcement' appearing in both HK and CA
        incidents, we expect at least 2 surfaces with that question_key.
        """
        # Count surfaces per question_key
        qk_to_surfaces = defaultdict(set)
        for surface_id, qk in kernel.surface_question_key.items():
            qk_to_surfaces[qk].add(surface_id)

        policy_surfaces = qk_to_surfaces.get("policy_announcement", set())

        assert len(policy_surfaces) >= 2, (
            f"INVARIANT VIOLATED: question_key='policy_announcement' has only "
            f"{len(policy_surfaces)} surface(s)!\n"
            f"  Expected: >= 2 (one per scope: HK and CA)\n"
            f"  Actual surfaces: {policy_surfaces}\n"
            f"\n"
            f"This confirms L2 is treating question_key as global identity,\n"
            f"which causes cross-event contamination."
        )

    def test_no_cross_incident_surface_sharing(self, kernel, trace):
        """
        Incidents with disjoint anchor motifs must not share surfaces.

        The HK incident (John Lee + Hong Kong) and CA incident (Newsom + California)
        have completely different anchor entities. They must not share any surface.
        """
        # Get incidents
        hk_incident = None
        ca_incident = None

        for inc_id, incident in kernel.incidents.items():
            anchors = incident.anchor_entities
            if "John Lee" in anchors or "Hong Kong" in anchors:
                hk_incident = incident
            elif "Gavin Newsom" in anchors or "California" in anchors:
                ca_incident = incident

        # If both incidents exist, check for shared surfaces
        if hk_incident and ca_incident:
            shared = hk_incident.surface_ids & ca_incident.surface_ids

            assert len(shared) == 0, (
                f"INVARIANT VIOLATED: Disjoint incidents share surfaces!\n"
                f"  HK incident surfaces: {hk_incident.surface_ids}\n"
                f"  CA incident surfaces: {ca_incident.surface_ids}\n"
                f"  Shared: {shared}\n"
                f"\n"
                f"This is the cross-event contamination bug."
            )

    def test_hub_entity_does_not_bridge_incidents(self, kernel, trace):
        """
        Generic hub entities (like 'United States') should not cause
        unrelated incidents to share surfaces.

        Both HK_C1 and CA_C1 mention 'United States', but this should not
        cause them to merge into the same surface.
        """
        # Get surfaces for claims that mention GENERIC entity
        hk_c1_surface = kernel.claim_to_surface.get("HK_C1")
        ca_c1_surface = kernel.claim_to_surface.get("CA_C1")

        if hk_c1_surface and ca_c1_surface:
            assert hk_c1_surface != ca_c1_surface, (
                f"INVARIANT VIOLATED: Hub entity 'United States' bridged unrelated claims!\n"
                f"  HK_C1 surface: {hk_c1_surface}\n"
                f"  CA_C1 surface: {ca_c1_surface}\n"
                f"\n"
                f"Generic entities should be suppressed in scope computation."
            )


class TestQuestionKeyUnscopedMetaClaim:
    """
    Test that the system emits a diagnostic meta-claim when
    question_key is used without proper scoping.

    This is the self-awareness mechanism that makes the failure auditable.
    """

    @pytest.fixture
    def trace(self):
        trace_path = FIXTURES_DIR / "golden_trace_scoped_surface.yaml"
        return GoldenTrace.load(trace_path)

    @pytest.fixture
    def kernel(self, trace):
        kernel = TraceKernel(trace)
        for tc in trace.claims:
            kernel.process_claim(tc)
        return kernel

    @pytest.mark.skip(reason="Meta-claim detector not yet implemented")
    def test_emits_question_key_unscoped_warning(self, kernel, trace):
        """
        If scoping fails, system should emit question_key_unscoped meta-claim.

        This test is skipped until the meta-claim detector is implemented.
        """
        # Check for question_key_unscoped meta-claim
        unscoped_claims = [
            mc for mc in kernel.meta_claims
            if mc.type == "question_key_unscoped"
        ]

        # If surfaces are improperly merged (test_separate_scopes fails),
        # we expect this meta-claim to be emitted
        if len(kernel.question_key_to_surface) == 1:
            # All claims went to one surface - scoping failed
            assert len(unscoped_claims) > 0, (
                "Scoping failed (global question_key bucket) but no "
                "question_key_unscoped meta-claim was emitted.\n"
                "The system should be self-aware of this failure mode."
            )


# =============================================================================
# Run as standalone script for debugging
# =============================================================================

if __name__ == "__main__":
    import sys

    trace_path = FIXTURES_DIR / "golden_trace_scoped_surface.yaml"
    trace = GoldenTrace.load(trace_path)

    print(f"Loaded trace: {trace.name}")
    print(f"Description: {trace.description}")
    print(f"Claims: {len(trace.claims)}")
    print()

    kernel = TraceKernel(trace)

    print("Processing claims...")
    for tc in trace.claims:
        exp = kernel.process_claim(tc)
        surface_id = kernel.claim_to_surface.get(tc.id)
        print(f"  {tc.id}: question_key={tc.question_key} -> surface={surface_id}")

    print()
    print("=== RESULT ===")
    print(f"Surfaces created: {len(kernel.surfaces)}")
    print(f"Question keys: {set(kernel.surface_question_key.values())}")

    # Check invariant
    hk_surface = kernel.claim_to_surface.get("HK_C1")
    ca_surface = kernel.claim_to_surface.get("CA_C1")

    if hk_surface == ca_surface:
        print()
        print("!!! INVARIANT VIOLATED !!!")
        print(f"HK and CA claims share the same surface: {hk_surface}")
        print("This is the bug that needs to be fixed.")
        sys.exit(1)
    else:
        print()
        print("INVARIANT HOLDS")
        print(f"HK surface: {hk_surface}")
        print(f"CA surface: {ca_surface}")
        sys.exit(0)
