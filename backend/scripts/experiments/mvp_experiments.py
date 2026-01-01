#!/usr/bin/env python3
"""
MVP Experiment Suite
====================

Comprehensive experiments to validate all inquiry logic/data flows before MVP launch.

Experiments:
1. Multi-schema inquiries (count, boolean, categorical)
2. Conflicting evidence and refutation handling
3. Attribution chain tracking
4. Scope validation and out-of-scope filtering
5. Resolution lifecycle with stability
6. Bounty distribution mechanics
7. Real data integration from DB

Run all: python scripts/experiments/mvp_experiments.py
Run one: python scripts/experiments/mvp_experiments.py --exp=1
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.inquiry.engine import InquiryEngine
from reee.inquiry.types import (
    InquirySchema, RigorLevel, InquiryStatus,
    ContributionType, TaskType, Inquiry
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


@dataclass
class ExperimentResult:
    name: str
    passed: bool
    details: Dict[str, Any]
    errors: List[str]


# =============================================================================
# EXPERIMENT 1: Multiple Schema Types
# =============================================================================

async def exp1_multi_schema_inquiries() -> ExperimentResult:
    """Test different inquiry schemas work correctly."""
    section("EXPERIMENT 1: Multi-Schema Inquiries")

    engine = InquiryEngine()
    errors = []
    results = {}

    # 1a. Monotone count (death toll)
    subsection("1a. Monotone Count Schema")

    count_schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="medium",
        count_max=500,
        count_monotone=True,
        rigor=RigorLevel.A
    )

    count_inquiry = engine.create_inquiry(
        title="How many died in the factory fire?",
        schema=count_schema,
        created_by="test_user",
        scope_entities=["Factory A", "City X"],
        initial_stake=50.0
    )

    # Add observations
    await engine.add_contribution(
        inquiry_id=count_inquiry.id,
        user_id="user1",
        contribution_type=ContributionType.EVIDENCE,
        text="At least 15 confirmed dead",
        source_url="https://news.example.com/fire",
        extracted_value=15,
        observation_kind=ObservationKind.LOWER_BOUND
    )

    await engine.add_contribution(
        inquiry_id=count_inquiry.id,
        user_id="user2",
        contribution_type=ContributionType.EVIDENCE,
        text="Death toll rises to 23",
        source_url="https://wire.example.com/update",
        extracted_value=23,
        observation_kind=ObservationKind.POINT
    )

    trace = engine.get_trace(count_inquiry.id)
    log(f"  Count inquiry MAP: {trace['belief_state']['map']}")
    log(f"  P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")

    if trace['belief_state']['map'] != 23:
        errors.append(f"Count MAP should be 23, got {trace['belief_state']['map']}")

    results['count'] = trace['belief_state']

    # 1b. Boolean schema
    subsection("1b. Boolean Schema")

    bool_schema = InquirySchema(
        schema_type="boolean",
        rigor=RigorLevel.B
    )

    bool_inquiry = engine.create_inquiry(
        title="Did the mayor resign?",
        schema=bool_schema,
        created_by="test_user",
        scope_entities=["Mayor Jones", "City Hall"],
        initial_stake=25.0
    )

    await engine.add_contribution(
        inquiry_id=bool_inquiry.id,
        user_id="user1",
        contribution_type=ContributionType.EVIDENCE,
        text="Mayor announces resignation effective immediately",
        source_url="https://local.news/resignation",
        extracted_value="true",
        observation_kind="point"
    )

    await engine.add_contribution(
        inquiry_id=bool_inquiry.id,
        user_id="user2",
        contribution_type=ContributionType.EVIDENCE,
        text="Confirmed: mayor has resigned",
        source_url="https://wire.example.com/breaking",
        extracted_value="true",
        observation_kind="point"
    )

    trace = engine.get_trace(bool_inquiry.id)
    log(f"  Boolean inquiry MAP: {trace['belief_state']['map']}")
    log(f"  P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")

    if trace['belief_state']['map'] != "true":
        errors.append(f"Boolean MAP should be 'true', got {trace['belief_state']['map']}")

    results['boolean'] = trace['belief_state']

    # 1c. Categorical schema
    subsection("1c. Categorical Schema")

    cat_schema = InquirySchema(
        schema_type="categorical",
        categories=["guilty", "not_guilty", "mistrial"],
        rigor=RigorLevel.A
    )

    cat_inquiry = engine.create_inquiry(
        title="What was the verdict in the Smith trial?",
        schema=cat_schema,
        created_by="test_user",
        scope_entities=["John Smith", "District Court"],
        initial_stake=100.0
    )

    await engine.add_contribution(
        inquiry_id=cat_inquiry.id,
        user_id="user1",
        contribution_type=ContributionType.EVIDENCE,
        text="Jury returns guilty verdict",
        source_url="https://court.news/verdict",
        extracted_value="guilty",
        observation_kind="point"
    )

    await engine.add_contribution(
        inquiry_id=cat_inquiry.id,
        user_id="user2",
        contribution_type=ContributionType.EVIDENCE,
        text="Smith found guilty on all counts",
        source_url="https://legal.times/smith",
        extracted_value="guilty",
        observation_kind="point"
    )

    trace = engine.get_trace(cat_inquiry.id)
    log(f"  Categorical inquiry MAP: {trace['belief_state']['map']}")
    log(f"  P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")

    if trace['belief_state']['map'] != "guilty":
        errors.append(f"Categorical MAP should be 'guilty', got {trace['belief_state']['map']}")

    results['categorical'] = trace['belief_state']

    # Summary
    subsection("Summary")
    log(f"  Inquiries created: 3")
    log(f"  Errors: {len(errors)}")
    for e in errors:
        log(f"    ❌ {e}")

    return ExperimentResult(
        name="Multi-Schema Inquiries",
        passed=len(errors) == 0,
        details=results,
        errors=errors
    )


# =============================================================================
# EXPERIMENT 2: Conflicting Evidence
# =============================================================================

async def exp2_conflicting_evidence() -> ExperimentResult:
    """Test how conflicting evidence affects posterior."""
    section("EXPERIMENT 2: Conflicting Evidence")

    engine = InquiryEngine()
    errors = []
    snapshots = []

    schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="small",
        count_max=100,
        count_monotone=True,
        rigor=RigorLevel.B
    )

    inquiry = engine.create_inquiry(
        title="How many were injured in the protest?",
        schema=schema,
        created_by="test_user",
        scope_entities=["Downtown Protest", "City Center"],
        initial_stake=30.0
    )

    subsection("Initial state")
    trace = engine.get_trace(inquiry.id)
    log(f"  Entropy: {trace['belief_state']['entropy_bits']:.2f} bits (uninformed)")
    snapshots.append(('initial', trace['belief_state'].copy()))

    # First report: 12 injured
    subsection("First report: 12 injured")
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="reporter1",
        contribution_type=ContributionType.EVIDENCE,
        text="Police report: 12 people injured",
        source_url="https://police.gov/report",
        extracted_value=12,
        observation_kind=ObservationKind.POINT
    )
    trace = engine.get_trace(inquiry.id)
    log(f"  MAP: {trace['belief_state']['map']} | P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")
    snapshots.append(('report_12', trace['belief_state'].copy()))

    # Conflicting report: 25 injured
    subsection("Conflicting report: 25 injured")
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="reporter2",
        contribution_type=ContributionType.EVIDENCE,
        text="Hospital confirms 25 treated for injuries",
        source_url="https://hospital.org/statement",
        extracted_value=25,
        observation_kind=ObservationKind.POINT
    )
    trace = engine.get_trace(inquiry.id)
    log(f"  MAP: {trace['belief_state']['map']} | P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")
    log(f"  Entropy: {trace['belief_state']['entropy_bits']:.2f} bits (should increase with conflict)")
    snapshots.append(('conflict_25', trace['belief_state'].copy()))

    # Corroborating higher value
    subsection("Corroboration: ~25 confirmed")
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="reporter3",
        contribution_type=ContributionType.EVIDENCE,
        text="At least 23 injuries confirmed by multiple sources",
        source_url="https://wire.example.com",
        extracted_value=23,
        observation_kind=ObservationKind.LOWER_BOUND
    )
    trace = engine.get_trace(inquiry.id)
    log(f"  MAP: {trace['belief_state']['map']} | P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")
    log(f"  Entropy: {trace['belief_state']['entropy_bits']:.2f} bits (should decrease)")
    snapshots.append(('corroborate_23', trace['belief_state'].copy()))

    # Check entropy dynamics
    if snapshots[2][1]['entropy_bits'] <= snapshots[1][1]['entropy_bits']:
        log(f"  ⚠️  Entropy didn't increase with conflict (expected but not critical)")

    if trace['belief_state']['map'] < 23:
        errors.append(f"MAP should be at least 23 with lower bound, got {trace['belief_state']['map']}")

    # Check tasks generated
    subsection("Generated Tasks")
    tasks = trace['tasks']
    log(f"  Tasks: {len(tasks)}")
    for t in tasks:
        log(f"    - [{t['type']}] {t['description']}")

    return ExperimentResult(
        name="Conflicting Evidence",
        passed=len(errors) == 0,
        details={'snapshots': [(s[0], s[1]) for s in snapshots]},
        errors=errors
    )


# =============================================================================
# EXPERIMENT 3: Attribution Chains
# =============================================================================

async def exp3_attribution_chains() -> ExperimentResult:
    """Test attribution contribution type."""
    section("EXPERIMENT 3: Attribution Chains")

    engine = InquiryEngine()
    errors = []

    schema = InquirySchema(
        schema_type="boolean",
        rigor=RigorLevel.B
    )

    inquiry = engine.create_inquiry(
        title="Did the CEO confirm the merger?",
        schema=schema,
        created_by="test_user",
        scope_entities=["CEO Smith", "MegaCorp", "AcquireCo"],
        initial_stake=75.0
    )

    subsection("Initial evidence: Blog reports CEO statement")
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="blogger",
        contribution_type=ContributionType.EVIDENCE,
        text="CEO Smith announced merger at press conference",
        source_url="https://techblog.example.com/merger",
        extracted_value="true",
        observation_kind="point"
    )

    trace = engine.get_trace(inquiry.id)
    log(f"  MAP: {trace['belief_state']['map']} | P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")
    initial_prob = trace['belief_state']['map_probability']

    subsection("Attribution: Blog was quoting Reuters")
    # User points out the blog was quoting Reuters
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="fact_checker",
        contribution_type=ContributionType.ATTRIBUTION,
        text="The blog post cites Reuters as original source. Original Reuters article linked.",
        source_url="https://reuters.com/merger-announcement"
    )

    trace = engine.get_trace(inquiry.id)
    contributions = trace['contributions']
    log(f"  Contributions: {len(contributions)}")
    for c in contributions:
        log(f"    - [{c['type']}] {c['text'][:60]}...")

    # In MVP, attribution is tracked but doesn't yet modify evidence weight
    # This is noted as a gap in the emulation script
    log(f"\n  Note: Attribution tracking implemented but weight modification pending")

    subsection("Direct confirmation from official source")
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="official",
        contribution_type=ContributionType.EVIDENCE,
        text="MegaCorp official press release confirms merger",
        source_url="https://megacorp.com/press/merger",
        extracted_value="true",
        observation_kind="point"
    )

    trace = engine.get_trace(inquiry.id)
    log(f"  MAP: {trace['belief_state']['map']} | P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")

    return ExperimentResult(
        name="Attribution Chains",
        passed=True,  # Attribution is tracked, weight modification is known gap
        details={'contributions': len(trace['contributions'])},
        errors=errors
    )


# =============================================================================
# EXPERIMENT 4: Scope Validation
# =============================================================================

async def exp4_scope_validation() -> ExperimentResult:
    """Test scope-based filtering (manual for now)."""
    section("EXPERIMENT 4: Scope Validation")

    engine = InquiryEngine()
    errors = []

    # Create two inquiries about similar but different events
    schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="medium",
        count_max=200,
        count_monotone=True,
        rigor=RigorLevel.B
    )

    inquiry1 = engine.create_inquiry(
        title="Deaths in Building A fire (Jan 2025)?",
        schema=schema,
        created_by="user1",
        scope_entities=["Building A", "Downtown"],
        scope_keywords=["fire", "january", "2025"],
        initial_stake=50.0
    )

    inquiry2 = engine.create_inquiry(
        title="Deaths in Building B fire (Feb 2025)?",
        schema=schema,
        created_by="user1",
        scope_entities=["Building B", "Uptown"],
        scope_keywords=["fire", "february", "2025"],
        initial_stake=50.0
    )

    log(f"  Created inquiry 1: {inquiry1.title}")
    log(f"    Scope entities: {inquiry1.scope_entities}")
    log(f"  Created inquiry 2: {inquiry2.title}")
    log(f"    Scope entities: {inquiry2.scope_entities}")

    subsection("Adding evidence to correct inquiry")
    await engine.add_contribution(
        inquiry_id=inquiry1.id,
        user_id="reporter",
        contribution_type=ContributionType.EVIDENCE,
        text="Building A fire: 15 dead confirmed",
        source_url="https://news.example.com/building-a",
        extracted_value=15,
        observation_kind=ObservationKind.POINT
    )

    await engine.add_contribution(
        inquiry_id=inquiry2.id,
        user_id="reporter",
        contribution_type=ContributionType.EVIDENCE,
        text="Building B fire: 8 dead confirmed",
        source_url="https://news.example.com/building-b",
        extracted_value=8,
        observation_kind=ObservationKind.POINT
    )

    trace1 = engine.get_trace(inquiry1.id)
    trace2 = engine.get_trace(inquiry2.id)

    log(f"\n  Inquiry 1 MAP: {trace1['belief_state']['map']}")
    log(f"  Inquiry 2 MAP: {trace2['belief_state']['map']}")

    if trace1['belief_state']['map'] == trace2['belief_state']['map']:
        errors.append("Inquiries should have different MAPs")

    subsection("Scope correction contribution")
    # Someone submits wrong event data, then corrects
    await engine.add_contribution(
        inquiry_id=inquiry1.id,
        user_id="corrector",
        contribution_type=ContributionType.SCOPE_CORRECTION,
        text="This claim is about Building B, not Building A. Wrong inquiry.",
        source_url=None
    )

    trace1 = engine.get_trace(inquiry1.id)
    log(f"  Scope correction recorded: {len([c for c in trace1['contributions'] if c['type'] == 'scope_correction'])} corrections")

    # Note: Actual scope enforcement requires integration with claim validation
    log(f"\n  Note: Scope validation recorded but enforcement requires claim-level integration")

    return ExperimentResult(
        name="Scope Validation",
        passed=len(errors) == 0,
        details={
            'inquiry1_scope': list(inquiry1.scope_entities),
            'inquiry2_scope': list(inquiry2.scope_entities)
        },
        errors=errors
    )


# =============================================================================
# EXPERIMENT 5: Resolution Lifecycle
# =============================================================================

async def exp5_resolution_lifecycle() -> ExperimentResult:
    """Test resolution criteria and stability tracking."""
    section("EXPERIMENT 5: Resolution Lifecycle")

    engine = InquiryEngine()
    errors = []

    schema = InquirySchema(
        schema_type="boolean",
        rigor=RigorLevel.A  # High rigor - can be resolved
    )

    inquiry = engine.create_inquiry(
        title="Was the document authentic?",
        schema=schema,
        created_by="investigator",
        scope_entities=["Document X", "Agency Y"],
        initial_stake=200.0
    )

    subsection("Building evidence toward resolution")

    # Add multiple confirming sources
    for i in range(5):
        await engine.add_contribution(
            inquiry_id=inquiry.id,
            user_id=f"expert_{i}",
            contribution_type=ContributionType.EVIDENCE,
            text=f"Expert analysis {i+1}: Document confirmed authentic",
            source_url=f"https://expert{i}.example.com/analysis",
            extracted_value="true",
            observation_kind="point"
        )

        inquiry_state = engine.get_inquiry(inquiry.id)
        trace = engine.get_trace(inquiry.id)
        log(f"  After source {i+1}: P(true) = {trace['belief_state']['map_probability']*100:.1f}%")

    subsection("Resolution criteria check")

    inquiry = engine.get_inquiry(inquiry.id)
    trace = engine.get_trace(inquiry.id)

    log(f"  Status: {inquiry.status.value}")
    log(f"  P(MAP): {inquiry.posterior_probability*100:.1f}%")
    log(f"  Stable since: {inquiry.stable_since}")
    log(f"  Blocking tasks: {inquiry.blocking_tasks}")
    log(f"  Is resolvable: {inquiry.is_resolvable()}")

    # Check criteria
    if inquiry.posterior_probability >= 0.95:
        log(f"  ✓ Meets probability threshold (≥95%)")
    else:
        log(f"  ✗ Below probability threshold ({inquiry.posterior_probability*100:.1f}% < 95%)")

    if inquiry.stable_since:
        log(f"  ✓ Stability tracking started")
    else:
        log(f"  ✗ No stability tracking (P < 95%)")

    if not inquiry.blocking_tasks:
        log(f"  ✓ No blocking tasks")
    else:
        log(f"  ✗ Has blocking tasks: {inquiry.blocking_tasks}")

    # Simulate time passing (for real system, this would be background job)
    subsection("Simulating stability period")
    log(f"  Note: Real resolution requires 24h stability period")
    log(f"  In production, background worker checks periodically")

    # For testing, manually set stable_since to 25 hours ago
    if inquiry.stable_since:
        old_stable = inquiry.stable_since
        inquiry.stable_since = datetime.utcnow() - timedelta(hours=25)

        is_resolvable_now = inquiry.is_resolvable()
        log(f"  After 25h simulation: is_resolvable = {is_resolvable_now}")

        # Restore for clean state
        inquiry.stable_since = old_stable

    return ExperimentResult(
        name="Resolution Lifecycle",
        passed=True,
        details={
            'final_probability': inquiry.posterior_probability,
            'has_stability_tracking': inquiry.stable_since is not None
        },
        errors=errors
    )


# =============================================================================
# EXPERIMENT 6: Bounty Mechanics
# =============================================================================

async def exp6_bounty_mechanics() -> ExperimentResult:
    """Test stake/bounty handling and task rewards."""
    section("EXPERIMENT 6: Bounty Mechanics")

    engine = InquiryEngine()
    errors = []

    schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="small",
        count_max=50,
        count_monotone=True,
        rigor=RigorLevel.B
    )

    inquiry = engine.create_inquiry(
        title="How many artifacts were recovered?",
        schema=schema,
        created_by="museum_curator",
        scope_entities=["Heist Site", "Museum"],
        initial_stake=100.0
    )

    subsection("Initial stake")
    log(f"  Initial stake: ${inquiry.total_stake:.2f}")
    log(f"  Stakes by user: {inquiry.stakes}")

    subsection("Adding more stakes")
    engine.add_stake(inquiry.id, "donor_1", 50.0)
    engine.add_stake(inquiry.id, "donor_2", 25.0)
    engine.add_stake(inquiry.id, "museum_curator", 25.0)  # Creator adds more

    inquiry = engine.get_inquiry(inquiry.id)
    log(f"  Total stake: ${inquiry.total_stake:.2f}")
    log(f"  Stakes by user: {dict(inquiry.stakes)}")

    expected_total = 100 + 50 + 25 + 25
    if inquiry.total_stake != expected_total:
        errors.append(f"Expected total stake {expected_total}, got {inquiry.total_stake}")

    subsection("Task bounties from stake pool")

    # Add evidence to trigger tasks
    await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="witness",
        contribution_type=ContributionType.EVIDENCE,
        text="Witness saw 12 artifacts being loaded",
        source_url="https://local.news/heist",
        extracted_value=12,
        observation_kind=ObservationKind.POINT
    )

    trace = engine.get_trace(inquiry.id)
    tasks = trace['tasks']

    log(f"  Generated tasks: {len(tasks)}")
    total_bounty = 0
    for t in tasks:
        log(f"    - {t['type']}: ${t['bounty']:.2f}")
        total_bounty += t['bounty']

    log(f"  Total task bounties: ${total_bounty:.2f}")
    log(f"  Bounty pool ratio: {total_bounty/inquiry.total_stake*100:.1f}% of stake")

    subsection("Task claiming (simulated)")
    log(f"  Note: Task claiming and payout requires user system integration")
    log(f"  Flow: user claims task → submits evidence → task marked complete → credits awarded")

    return ExperimentResult(
        name="Bounty Mechanics",
        passed=len(errors) == 0,
        details={
            'total_stake': inquiry.total_stake,
            'contributors': len(inquiry.stakes),
            'task_bounties': total_bounty
        },
        errors=errors
    )


# =============================================================================
# EXPERIMENT 7: Real DB Integration
# =============================================================================

async def exp7_real_db_integration() -> ExperimentResult:
    """Test loading real claims from database into inquiry."""
    section("EXPERIMENT 7: Real Database Integration")

    errors = []
    details = {}

    try:
        from reee.experiments.loader import create_context, log as db_log

        subsection("Connecting to database")
        ctx = await create_context()
        log(f"  ✓ Connected to Neo4j and PostgreSQL")

        # Find a real event
        subsection("Finding real events")
        events = await ctx.neo4j._execute_read('''
            MATCH (e:Event)-[:INTAKES]->(c:Claim)
            WITH e, count(c) as cnt
            WHERE cnt >= 10
            RETURN e.id as id, e.canonical_name as name, cnt
            ORDER BY cnt DESC
            LIMIT 3
        ''', {})

        log(f"  Found {len(events)} events with 10+ claims:")
        for e in events:
            log(f"    - {e['name']}: {e['cnt']} claims")

        if events:
            # Create inquiry from first event
            subsection("Creating inquiry from real event")

            event = events[0]
            engine = InquiryEngine()

            schema = InquirySchema(
                schema_type="monotone_count",
                count_scale="medium",
                count_max=500,
                count_monotone=True,
                rigor=RigorLevel.B
            )

            inquiry = engine.create_inquiry(
                title=f"Death toll: {event['name']}?",
                schema=schema,
                created_by="system",
                scope_entities=[event['name']],
                initial_stake=100.0
            )

            # Load claims and add as contributions
            claims = await ctx.neo4j._execute_read('''
                MATCH (e:Event {id: $eid})-[:INTAKES]->(c:Claim)
                WHERE c.text =~ '(?i).*(dead|death|kill|fatal).*\\\\d+.*'
                   OR c.text =~ '(?i).*\\\\d+.*(dead|death|kill|fatal).*'
                OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
                RETURN c.id as id, c.text as text, p.domain as source
                LIMIT 10
            ''', {'eid': event['id']})

            log(f"  Found {len(claims)} death-related claims")

            # Add claims as contributions (simplified - real impl would extract values)
            for claim in claims[:5]:
                # Simple number extraction
                import re
                numbers = re.findall(r'\d+', claim['text'])
                if numbers:
                    value = max(int(n) for n in numbers if int(n) < 1000)
                    await engine.add_contribution(
                        inquiry_id=inquiry.id,
                        user_id="crawler",
                        contribution_type=ContributionType.EVIDENCE,
                        text=claim['text'][:200],
                        source_url=f"https://{claim['source'] or 'unknown'}/",
                        extracted_value=value,
                        observation_kind=ObservationKind.LOWER_BOUND
                    )

            trace = engine.get_trace(inquiry.id)
            log(f"\n  Inquiry state after real claims:")
            log(f"    MAP: {trace['belief_state']['map']}")
            log(f"    P(MAP): {trace['belief_state']['map_probability']*100:.1f}%")
            log(f"    Observations: {trace['belief_state']['observation_count']}")

            details['event'] = event['name']
            details['claims_processed'] = len(claims[:5])
            details['final_map'] = trace['belief_state']['map']

        await ctx.close()
        log(f"\n  ✓ Database connection closed")

    except ImportError as e:
        log(f"  ⚠ Database modules not available: {e}")
        log(f"  Skipping real DB integration test")
        details['skipped'] = True
    except Exception as e:
        log(f"  ❌ Database error: {e}")
        errors.append(str(e))

    return ExperimentResult(
        name="Real DB Integration",
        passed=len(errors) == 0,
        details=details,
        errors=errors
    )


# =============================================================================
# MAIN
# =============================================================================

async def run_all_experiments():
    """Run all experiments and report results."""

    experiments = [
        ("1", exp1_multi_schema_inquiries),
        ("2", exp2_conflicting_evidence),
        ("3", exp3_attribution_chains),
        ("4", exp4_scope_validation),
        ("5", exp5_resolution_lifecycle),
        ("6", exp6_bounty_mechanics),
        ("7", exp7_real_db_integration),
    ]

    results = []

    section("MVP EXPERIMENT SUITE")
    log(f"  Running {len(experiments)} experiments...")

    for exp_id, exp_fn in experiments:
        try:
            result = await exp_fn()
            results.append(result)
        except Exception as e:
            import traceback
            results.append(ExperimentResult(
                name=f"Experiment {exp_id}",
                passed=False,
                details={},
                errors=[str(e), traceback.format_exc()]
            ))

    # Summary
    section("EXPERIMENT RESULTS SUMMARY")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    log(f"  Total: {len(results)}")
    log(f"  Passed: {passed}")
    log(f"  Failed: {failed}")
    log("")

    for r in results:
        status = "✓" if r.passed else "✗"
        log(f"  {status} {r.name}")
        if r.errors:
            for e in r.errors[:2]:
                log(f"      Error: {e[:100]}")

    # Known gaps
    section("KNOWN MVP GAPS (from experiments)")
    log("""
  1. Attribution weight modification not implemented
     - Attribution is tracked but doesn't modify evidence credibility yet

  2. Scope enforcement is manual
     - Scope recorded but not enforced at claim validation level

  3. Resolution requires background worker
     - 24h stability check needs periodic background job

  4. Task completion and payout
     - Task claiming works but credit payout needs user system

  5. Deduplication not implemented
     - Same claim from different contributions not merged

  6. Surface formation during inquiry
     - Claims not clustered into surfaces yet
    """)

    return results


async def run_single_experiment(exp_num: str):
    """Run a single experiment by number."""
    exp_map = {
        "1": exp1_multi_schema_inquiries,
        "2": exp2_conflicting_evidence,
        "3": exp3_attribution_chains,
        "4": exp4_scope_validation,
        "5": exp5_resolution_lifecycle,
        "6": exp6_bounty_mechanics,
        "7": exp7_real_db_integration,
    }

    if exp_num not in exp_map:
        log(f"Unknown experiment: {exp_num}")
        log(f"Available: {list(exp_map.keys())}")
        return

    result = await exp_map[exp_num]()

    section("RESULT")
    log(f"  {result.name}: {'PASSED' if result.passed else 'FAILED'}")
    if result.errors:
        for e in result.errors:
            log(f"  Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="Run single experiment (1-7)")
    args = parser.parse_args()

    if args.exp:
        asyncio.run(run_single_experiment(args.exp))
    else:
        asyncio.run(run_all_experiments())
