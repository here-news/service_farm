#!/usr/bin/env python3
"""
Scope Contamination and Entity Homonym Tests
=============================================

Critical experiments to ensure REEE doesn't mix up:
1. Different events with similar entities (two fires in same city)
2. Entity homonyms (two different people with same name)
3. Similar events with overlapping scope

These are the hardest cases for any news aggregation system.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.inquiry.engine import InquiryEngine
from reee.inquiry.types import (
    InquirySchema, RigorLevel, InquiryStatus,
    ContributionType
)
from reee.typed_belief import ObservationKind


def log(msg: str):
    print(msg, flush=True)


def section(title: str):
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}\n")


def subsection(title: str):
    log(f"\n  --- {title} ---\n")


# =============================================================================
# TEST 1: Two Different Fires in Hong Kong
# =============================================================================

async def test_two_fires_same_city():
    """
    Scenario: Two different building fires in Hong Kong
    - Fire A: Wang Fuk Court, November 2025
    - Fire B: Lai Chi Kok, December 2025

    Challenge: Both are "Hong Kong fire" - must not mix death tolls
    """
    section("TEST 1: Two Fires in Same City (Scope Contamination)")

    engine = InquiryEngine()

    schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="medium",
        count_max=500,
        count_monotone=True,
        rigor=RigorLevel.B
    )

    # Create two distinct inquiries
    fire_a = engine.create_inquiry(
        title="Deaths in Wang Fuk Court fire (Tai Po, Nov 2025)?",
        schema=schema,
        created_by="investigator",
        scope_entities=["Wang Fuk Court", "Tai Po"],
        scope_keywords=["november", "2025", "wang fuk"],
        initial_stake=100.0,
    )

    fire_b = engine.create_inquiry(
        title="Deaths in Lai Chi Kok warehouse fire (Dec 2025)?",
        schema=schema,
        created_by="investigator",
        scope_entities=["Lai Chi Kok", "warehouse"],
        scope_keywords=["december", "2025", "warehouse"],
        initial_stake=100.0,
    )

    log(f"  Fire A: {fire_a.title}")
    log(f"    Scope: {fire_a.scope_entities}")
    log(f"  Fire B: {fire_b.title}")
    log(f"    Scope: {fire_b.scope_entities}")

    subsection("Adding evidence to Fire A")

    await engine.add_contribution(
        inquiry_id=fire_a.id,
        user_id="reporter1",
        contribution_type=ContributionType.EVIDENCE,
        text="Wang Fuk Court fire: 160 confirmed dead",
        source_url="https://news.example.com/wang-fuk",
        extracted_value=160,
        observation_kind=ObservationKind.POINT
    )

    await engine.add_contribution(
        inquiry_id=fire_a.id,
        user_id="reporter2",
        contribution_type=ContributionType.EVIDENCE,
        text="Tai Po fire death toll at 160",
        source_url="https://local.news/tai-po-fire",
        extracted_value=160,
        observation_kind=ObservationKind.POINT
    )

    subsection("Adding evidence to Fire B")

    await engine.add_contribution(
        inquiry_id=fire_b.id,
        user_id="reporter3",
        contribution_type=ContributionType.EVIDENCE,
        text="Lai Chi Kok warehouse fire: 12 workers dead",
        source_url="https://news.example.com/warehouse",
        extracted_value=12,
        observation_kind=ObservationKind.POINT
    )

    subsection("Adding ambiguous 'Hong Kong fire' evidence")

    # This evidence mentions "Hong Kong fire" without specifics
    # Current system: User submits to specific inquiry
    # Future: System should detect scope mismatch

    await engine.add_contribution(
        inquiry_id=fire_a.id,  # User submits to Fire A
        user_id="reporter4",
        contribution_type=ContributionType.EVIDENCE,
        text="Hong Kong fire death toll continues to rise",
        source_url="https://vague.news/hk-fire",
        extracted_value=165,
        observation_kind=ObservationKind.LOWER_BOUND
    )

    # Someone tries to submit warehouse fire data to wrong inquiry
    await engine.add_contribution(
        inquiry_id=fire_a.id,
        user_id="confused_user",
        contribution_type=ContributionType.EVIDENCE,
        text="Warehouse fire in Hong Kong kills 12",
        source_url="https://other.news/warehouse",
        extracted_value=12,
        observation_kind=ObservationKind.POINT
    )

    # Scope correction catches the error
    await engine.add_contribution(
        inquiry_id=fire_a.id,
        user_id="moderator",
        contribution_type=ContributionType.SCOPE_CORRECTION,
        text="The '12 dead' claim is about Lai Chi Kok warehouse fire, not Wang Fuk Court",
        source_url=None
    )

    subsection("Results")

    trace_a = engine.get_trace(fire_a.id)
    trace_b = engine.get_trace(fire_b.id)

    log(f"  Fire A (Wang Fuk Court):")
    log(f"    MAP: {trace_a['belief_state']['map']}")
    log(f"    P(MAP): {trace_a['belief_state']['map_probability']*100:.1f}%")
    log(f"    Observations: {trace_a['belief_state']['observation_count']}")
    log(f"    Scope corrections: {len([c for c in trace_a['contributions'] if c['type'] == 'scope_correction'])}")

    log(f"\n  Fire B (Lai Chi Kok):")
    log(f"    MAP: {trace_b['belief_state']['map']}")
    log(f"    P(MAP): {trace_b['belief_state']['map_probability']*100:.1f}%")
    log(f"    Observations: {trace_b['belief_state']['observation_count']}")

    # Validation
    errors = []

    if trace_a['belief_state']['map'] < 160:
        errors.append(f"Fire A MAP should be >= 160, got {trace_a['belief_state']['map']}")

    if trace_b['belief_state']['map'] != 12:
        errors.append(f"Fire B MAP should be 12, got {trace_b['belief_state']['map']}")

    log(f"\n  Validation: {'PASSED' if not errors else 'FAILED'}")
    for e in errors:
        log(f"    ❌ {e}")

    log("""
  KEY INSIGHT:
    - Scope entities prevent automatic mixing
    - Scope corrections log cross-contamination attempts
    - Future: Automatic scope matching at submission time
    """)

    return fire_a, fire_b, errors


# =============================================================================
# TEST 2: Entity Homonyms (Two People, Same Name)
# =============================================================================

async def test_entity_homonyms():
    """
    Scenario: Two different people named "Charlie Kirk"
    - Charlie Kirk A: Founder of TPUSA, political commentator
    - Charlie Kirk B: Victim in an unrelated incident

    Challenge: Must not merge claims about different people
    """
    section("TEST 2: Entity Homonyms (Two People Named Charlie Kirk)")

    engine = InquiryEngine()

    # Inquiry about political figure
    inquiry_political = engine.create_inquiry(
        title="Did Charlie Kirk (TPUSA) make the controversial statement?",
        schema=InquirySchema(schema_type="boolean", rigor=RigorLevel.B),
        created_by="fact_checker",
        scope_entities=["Charlie Kirk", "TPUSA", "Turning Point"],
        scope_keywords=["political", "conservative", "statement"],
        initial_stake=50.0,
    )

    # Inquiry about accident victim
    inquiry_victim = engine.create_inquiry(
        title="Was Charlie Kirk (age 34) the victim in the accident?",
        schema=InquirySchema(schema_type="boolean", rigor=RigorLevel.B),
        created_by="investigator",
        scope_entities=["Charlie Kirk", "Highway 101", "accident"],
        scope_keywords=["accident", "victim", "crash"],
        initial_stake=30.0,
    )

    log(f"  Inquiry 1 (Political): {inquiry_political.title}")
    log(f"    Scope: {inquiry_political.scope_entities}")
    log(f"  Inquiry 2 (Victim): {inquiry_victim.title}")
    log(f"    Scope: {inquiry_victim.scope_entities}")

    subsection("Adding evidence to political inquiry")

    await engine.add_contribution(
        inquiry_id=inquiry_political.id,
        user_id="reporter1",
        contribution_type=ContributionType.EVIDENCE,
        text="TPUSA founder Charlie Kirk posted statement on X",
        source_url="https://x.com/charliekirk11/status/123",
        extracted_value="true",
        observation_kind="point"
    )

    await engine.add_contribution(
        inquiry_id=inquiry_political.id,
        user_id="reporter2",
        contribution_type=ContributionType.EVIDENCE,
        text="Turning Point USA confirms Kirk's statement",
        source_url="https://tpusa.com/news",
        extracted_value="true",
        observation_kind="point"
    )

    subsection("Adding evidence to victim inquiry")

    await engine.add_contribution(
        inquiry_id=inquiry_victim.id,
        user_id="reporter3",
        contribution_type=ContributionType.EVIDENCE,
        text="Victim identified as Charlie Kirk, 34, of San Jose",
        source_url="https://local.news/accident",
        extracted_value="true",
        observation_kind="point"
    )

    subsection("Testing disambiguation")

    # Someone submits without context - should trigger disambiguation
    await engine.add_contribution(
        inquiry_id=inquiry_political.id,
        user_id="confused_user",
        contribution_type=ContributionType.EVIDENCE,
        text="Charlie Kirk was killed in the accident",
        source_url="https://news.example.com/accident",
        extracted_value="false",  # Wait, this seems wrong...
        observation_kind="point"
    )

    # Disambiguation contribution
    await engine.add_contribution(
        inquiry_id=inquiry_political.id,
        user_id="moderator",
        contribution_type=ContributionType.DISAMBIGUATION,
        text="The 'killed in accident' claim refers to a different Charlie Kirk (age 34, San Jose), not the TPUSA founder",
        source_url=None
    )

    subsection("Results")

    trace_political = engine.get_trace(inquiry_political.id)
    trace_victim = engine.get_trace(inquiry_victim.id)

    log(f"  Political inquiry:")
    log(f"    MAP: {trace_political['belief_state']['map']}")
    log(f"    P(MAP): {trace_political['belief_state']['map_probability']*100:.1f}%")
    log(f"    Disambiguations: {len([c for c in trace_political['contributions'] if c['type'] == 'disambiguation'])}")

    log(f"\n  Victim inquiry:")
    log(f"    MAP: {trace_victim['belief_state']['map']}")
    log(f"    P(MAP): {trace_victim['belief_state']['map_probability']*100:.1f}%")

    errors = []

    # Political inquiry should still show "true" (statement was made)
    if trace_political['belief_state']['map'] != "true":
        errors.append(f"Political inquiry should be 'true', got {trace_political['belief_state']['map']}")

    log(f"\n  Validation: {'PASSED' if not errors else 'FAILED'}")
    for e in errors:
        log(f"    ❌ {e}")

    log("""
  KEY INSIGHT:
    - Entity homonyms require additional scope context
    - TPUSA, Turning Point = political Charlie Kirk
    - Highway 101, accident = victim Charlie Kirk
    - Disambiguation contribution corrects misattribution
    """)

    return inquiry_political, inquiry_victim, errors


# =============================================================================
# TEST 3: Overlapping but Distinct Events
# =============================================================================

async def test_overlapping_events():
    """
    Scenario: Multiple related but distinct events
    - Mass shooting at location X
    - Subsequent memorial event at same location
    - Anniversary coverage

    Challenge: Different inquiries about same location, different events
    """
    section("TEST 3: Overlapping Events at Same Location")

    engine = InquiryEngine()

    # Original incident
    incident_inquiry = engine.create_inquiry(
        title="Casualties in the Brown University shooting (Dec 2025)?",
        schema=InquirySchema(
            schema_type="monotone_count",
            count_scale="small",
            count_max=50,
            count_monotone=True,
            rigor=RigorLevel.A
        ),
        created_by="investigator",
        scope_entities=["Brown University", "shooting"],
        scope_keywords=["shooting", "december", "2025", "victim"],
        initial_stake=200.0,
    )

    # Memorial event
    memorial_inquiry = engine.create_inquiry(
        title="Attendance at Brown University memorial (Jan 2026)?",
        schema=InquirySchema(
            schema_type="monotone_count",
            count_scale="medium",
            count_max=10000,
            count_monotone=False,  # Attendance can be revised down
            rigor=RigorLevel.C  # Exploratory
        ),
        created_by="reporter",
        scope_entities=["Brown University", "memorial"],
        scope_keywords=["memorial", "january", "2026", "attendance"],
        initial_stake=20.0,
    )

    log(f"  Incident inquiry: {incident_inquiry.title}")
    log(f"  Memorial inquiry: {memorial_inquiry.title}")

    subsection("Adding evidence to incident")

    await engine.add_contribution(
        inquiry_id=incident_inquiry.id,
        user_id="police",
        contribution_type=ContributionType.EVIDENCE,
        text="Brown University shooting: 4 dead, 12 injured",
        source_url="https://police.gov/statement",
        extracted_value=4,
        observation_kind=ObservationKind.POINT
    )

    subsection("Adding evidence to memorial")

    await engine.add_contribution(
        inquiry_id=memorial_inquiry.id,
        user_id="reporter",
        contribution_type=ContributionType.EVIDENCE,
        text="Thousands gather at Brown University for memorial",
        source_url="https://news.example.com/memorial",
        extracted_value=3000,
        observation_kind=ObservationKind.APPROXIMATE
    )

    # Confusing headline
    await engine.add_contribution(
        inquiry_id=memorial_inquiry.id,
        user_id="reporter2",
        contribution_type=ContributionType.EVIDENCE,
        text="Brown University: 4,000 gather to remember victims",
        source_url="https://local.news/memorial",
        extracted_value=4000,
        observation_kind=ObservationKind.POINT
    )

    subsection("Results")

    trace_incident = engine.get_trace(incident_inquiry.id)
    trace_memorial = engine.get_trace(memorial_inquiry.id)

    log(f"  Incident (shooting deaths):")
    log(f"    MAP: {trace_incident['belief_state']['map']}")
    log(f"    P(MAP): {trace_incident['belief_state']['map_probability']*100:.1f}%")

    log(f"\n  Memorial (attendance):")
    log(f"    MAP: {trace_memorial['belief_state']['map']}")
    log(f"    P(MAP): {trace_memorial['belief_state']['map_probability']*100:.1f}%")

    errors = []

    if trace_incident['belief_state']['map'] != 4:
        errors.append(f"Incident should be 4, got {trace_incident['belief_state']['map']}")

    # Memorial should be in thousands, not single digits
    if trace_memorial['belief_state']['map'] < 1000:
        errors.append(f"Memorial should be >1000, got {trace_memorial['belief_state']['map']}")

    log(f"\n  Validation: {'PASSED' if not errors else 'FAILED'}")
    for e in errors:
        log(f"    ❌ {e}")

    log("""
  KEY INSIGHT:
    - Same location can have multiple distinct events
    - Keywords (shooting vs memorial) disambiguate
    - Time scope (December vs January) helps
    - Different scales prevent cross-contamination
    """)

    return incident_inquiry, memorial_inquiry, errors


# =============================================================================
# MAIN
# =============================================================================

async def main():
    section("SCOPE AND HOMONYM EXPERIMENTS")

    log("  Testing critical edge cases for news aggregation:")
    log("    1. Two events in same city (scope contamination)")
    log("    2. Entity homonyms (same name, different people)")
    log("    3. Overlapping events at same location")

    all_errors = []

    # Run tests
    _, _, errors1 = await test_two_fires_same_city()
    all_errors.extend(errors1)

    _, _, errors2 = await test_entity_homonyms()
    all_errors.extend(errors2)

    _, _, errors3 = await test_overlapping_events()
    all_errors.extend(errors3)

    # Summary
    section("EXPERIMENT SUMMARY")

    if not all_errors:
        log("  ✓ All scope and homonym tests PASSED")
    else:
        log(f"  ✗ {len(all_errors)} errors found:")
        for e in all_errors:
            log(f"    - {e}")

    log("""
  CURRENT CAPABILITIES:
    ✓ Scope entities recorded per inquiry
    ✓ Scope correction contributions tracked
    ✓ Disambiguation contributions tracked
    ✓ Separate inquiries don't share evidence

  NEEDED FOR PRODUCTION:
    ○ Automatic scope matching at submission
    ○ Entity linking to canonical IDs
    ○ Cross-inquiry contamination warnings
    ○ Surface-level identity separation
    """)


if __name__ == "__main__":
    asyncio.run(main())
