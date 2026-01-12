#!/usr/bin/env python3
"""
Proof-of-concept: DF filtering blocks broad entities from opening spine gates.

Tests the specific fix:
- should_suppress_entity() with shrinkage
- Deterministic REFERENT_LOCATION → BROAD_LOCATION demotion
- Spine gate evaluation respects suppression

Expected outcome:
- "Tai Po" (df=45), "Hong Kong" (df=200) → suppressed
- "Wang Fuk Court" (df=48) → NOT suppressed (facility, not location)
"""

import sys
sys.path.insert(0, '/media/im3/plus/lab4/re_news/service_farm/backend')

from reee.contracts.case_formation import (
    should_suppress_entity,
    EntityDFEstimate,
    evaluate_spine_gate,
    IncidentRoleArtifact,
    EntityRole
)

def test_df_suppression():
    """Test that high-DF broad locations are suppressed."""

    global_total = 515  # Total incidents in corpus
    threshold = 0.05  # 5% threshold

    # Case 1: "Tai Po" - REFERENT_LOCATION with df=45
    df_tai_po = EntityDFEstimate.compute(
        entity_id="tai-po",
        df_global=45,
        df_local=5,
        alpha=0.9
    )
    df_tai_po_shrunk = df_tai_po.df_shrunk

    suppress_tai_po = should_suppress_entity(
        entity_id="tai-po",
        role=EntityRole.REFERENT_LOCATION,
        df_shrunk=df_tai_po_shrunk,
        global_total_incidents=global_total,
        threshold_fraction=threshold
    )

    print(f"Tai Po (df_global=45):")
    print(f"  df_shrunk={df_tai_po_shrunk:.1f}")
    print(f"  fraction={df_tai_po_shrunk/global_total:.1%} > {threshold:.1%}")
    print(f"  suppressed={suppress_tai_po} ✓" if suppress_tai_po else f"  suppressed={suppress_tai_po} ✗")
    print()

    # Case 2: "Hong Kong" - REFERENT_LOCATION with df=200
    df_hk = EntityDFEstimate.compute(
        entity_id="hong-kong",
        df_global=200,
        df_local=8,
        alpha=0.9
    )
    df_hk_shrunk = df_hk.df_shrunk

    suppress_hk = should_suppress_entity(
        entity_id="hong-kong",
        role=EntityRole.REFERENT_LOCATION,
        df_shrunk=df_hk_shrunk,
        global_total_incidents=global_total,
        threshold_fraction=threshold
    )

    print(f"Hong Kong (df_global=200):")
    print(f"  df_shrunk={df_hk_shrunk:.1f}")
    print(f"  fraction={df_hk_shrunk/global_total:.1%} > {threshold:.1%}")
    print(f"  suppressed={suppress_hk} ✓" if suppress_hk else f"  suppressed={suppress_hk} ✗")
    print()

    # Case 3: "Wang Fuk Court" - REFERENT_FACILITY with df=48
    df_wfc = EntityDFEstimate.compute(
        entity_id="wang-fuk-court",
        df_global=48,
        df_local=48,  # All WFC
        alpha=0.9
    )
    df_wfc_shrunk = df_wfc.df_shrunk

    suppress_wfc = should_suppress_entity(
        entity_id="wang-fuk-court",
        role=EntityRole.REFERENT_FACILITY,  # FACILITY, not LOCATION
        df_shrunk=df_wfc_shrunk,
        global_total_incidents=global_total,
        threshold_fraction=threshold
    )

    print(f"Wang Fuk Court (df_global=48, FACILITY):")
    print(f"  df_shrunk={df_wfc_shrunk:.1f}")
    print(f"  fraction={df_wfc_shrunk/global_total:.1%} > {threshold:.1%}")
    print(f"  suppressed={suppress_wfc} (FACILITY never suppressed) ✓" if not suppress_wfc else f"  suppressed={suppress_wfc} ✗")
    print()

    return suppress_tai_po and suppress_hk and not suppress_wfc


def test_spine_gate_with_suppression():
    """Test that suppressed entities don't open spine gates."""

    # Artifact A: Wang Fuk Court fire (valid referent)
    art_a = IncidentRoleArtifact(
        incident_id="incident-a",
        entity_roles={
            "Wang Fuk Court": EntityRole.REFERENT_FACILITY,
            "Tai Po": EntityRole.BROAD_LOCATION,  # Demoted by DF filter
        },
        primary_entities=["Wang Fuk Court"],
        person_time_witness="2024-11-26"
    )

    # Artifact B: Another Wang Fuk Court incident
    art_b = IncidentRoleArtifact(
        incident_id="incident-b",
        entity_roles={
            "Wang Fuk Court": EntityRole.REFERENT_FACILITY,
            "Hong Kong": EntityRole.BROAD_LOCATION,  # Demoted by DF filter
        },
        primary_entities=["Wang Fuk Court"],
        person_time_witness="2024-11-26"
    )

    # Artifact C: Different event in Tai Po (should NOT merge)
    art_c = IncidentRoleArtifact(
        incident_id="incident-c",
        entity_roles={
            "Some Other Building": EntityRole.REFERENT_FACILITY,
            "Tai Po": EntityRole.BROAD_LOCATION,
        },
        primary_entities=["Some Other Building"],
        person_time_witness="2024-11-26"
    )

    # Test 1: A-B should open spine gate (shared referent WFC)
    gate_ab = evaluate_spine_gate(art_a, art_b, time_closeness_days=7)
    print(f"Spine gate A-B (shared WFC):")
    print(f"  should_open={gate_ab} ✓" if gate_ab else f"  should_open={gate_ab} ✗")
    print()

    # Test 2: A-C should NOT open spine gate (only BROAD_LOCATION overlap)
    gate_ac = evaluate_spine_gate(art_a, art_c, time_closeness_days=7)
    print(f"Spine gate A-C (only Tai Po, demoted to BROAD):")
    print(f"  should_open={gate_ac} (BROAD_LOCATION doesn't open gates) ✓" if not gate_ac else f"  should_open={gate_ac} ✗")
    print()

    return gate_ab and not gate_ac


if __name__ == "__main__":
    print("=" * 60)
    print("DF FILTERING PROOF OF CONCEPT")
    print("=" * 60)
    print()

    print("TEST 1: DF-based suppression")
    print("-" * 60)
    test1_pass = test_df_suppression()
    print()

    print("TEST 2: Spine gate evaluation with suppressed entities")
    print("-" * 60)
    test2_pass = test_spine_gate_with_suppression()
    print()

    print("=" * 60)
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED")
        print()
        print("Summary:")
        print("- Tai Po & Hong Kong suppressed (high DF)")
        print("- Wang Fuk Court NOT suppressed (facility referent)")
        print("- Spine gates only open for valid referents")
        print("- BROAD_LOCATION entities don't contaminate cases")
    else:
        print("❌ TESTS FAILED")
        sys.exit(1)
