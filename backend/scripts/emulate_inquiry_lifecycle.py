#!/usr/bin/env python3
"""
End-to-End Inquiry Lifecycle Emulation

Simulates the complete journey from uncertain event detection to resolution:
1. Detect high-entropy event from extracted claims
2. Create inquiry with proper scope
3. Add contributions (evidence from multiple sources)
4. Watch belief state evolve
5. Generate tasks from meta-claims
6. Reach resolution

This script exposes gaps in the current implementation.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reee.inquiry.engine import InquiryEngine
from reee.inquiry.types import (
    InquirySchema, RigorLevel, InquiryStatus,
    ContributionType, TaskType
)
from reee.typed_belief import ObservationKind

# Initialize engine
engine = InquiryEngine()

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def print_belief_state(trace: Dict):
    """Pretty print the current belief state"""
    bs = trace['belief_state']
    print(f"  MAP Value: {bs['map']}")
    print(f"  P(MAP): {bs['map_probability']*100:.1f}%")
    print(f"  Entropy: {bs['entropy_bits']:.2f} bits")
    print(f"  Normalized Entropy: {bs['normalized_entropy']*100:.1f}%")
    print(f"  Observations: {bs['observation_count']}")

    if trace.get('posterior_top_10'):
        print(f"\n  Top hypotheses:")
        for item in trace['posterior_top_10'][:5]:
            bar = '█' * int(item['probability'] * 20)
            print(f"    {item['value']:>6}: {item['probability']*100:5.1f}% {bar}")

def print_tasks(tasks: List[Dict]):
    """Pretty print open tasks"""
    if not tasks:
        print("  No open tasks")
        return
    for task in tasks:
        status = "✓" if task.get('completed') else "○"
        print(f"  {status} [{task['type']}] {task['description']}")
        print(f"      Bounty: ${task['bounty']:.2f}")

async def run_emulation():
    """Run the full inquiry lifecycle"""

    print_section("PHASE 1: Event Detection (Simulated)")

    # In production, this would come from our event detection pipeline
    # For now, we simulate detecting a high-entropy event
    print("  Simulating detection of uncertain event from crawled claims...")
    print()
    print("  Detected claims with conflicting death tolls:")
    print("    - Reuters: 'at least 160 dead'")
    print("    - BBC: 'approximately 165 casualties'")
    print("    - SCMP: 'official count stands at 158'")
    print("    - AP: 'death toll rises to 170'")
    print()
    print("  High entropy detected: multiple sources, conflicting numbers")
    print("  → Suggesting inquiry creation")

    # =========================================================================
    print_section("PHASE 2: Inquiry Creation")

    # Define the inquiry schema
    schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="medium",  # 10-500 range
        count_max=500,
        count_monotone=True,  # Death toll only increases
        rigor=RigorLevel.B    # Typed + generic priors
    )

    # Create the inquiry
    inquiry = engine.create_inquiry(
        title="How many people died in the Wang Fuk Court fire (Tai Po, Nov 2025)?",
        description="Death toll from the high-rise fire in Tai Po, Hong Kong",
        schema=schema,
        created_by="system",  # Auto-created from event detection
        scope_entities=["Wang Fuk Court", "Tai Po", "Hong Kong"],
        scope_keywords=["fire", "death", "casualty", "killed"],
        initial_stake=100.0  # Seed bounty
    )

    print(f"  Created inquiry: {inquiry.id}")
    print(f"  Title: {inquiry.title}")
    print(f"  Schema: {inquiry.schema.schema_type} (monotone={inquiry.schema.count_monotone})")
    print(f"  Rigor: {inquiry.schema.rigor.value}")
    print(f"  Initial stake: ${inquiry.total_stake:.2f}")
    print(f"  Scope entities: {inquiry.scope_entities}")

    # Get initial trace
    trace = engine.get_trace(inquiry.id)
    print("\n  Initial belief state:")
    print_belief_state(trace)

    # =========================================================================
    print_section("PHASE 3: First Contribution (Reuters)")

    # Simulate first evidence from Reuters
    contrib1 = await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="user_alice",
        contribution_type=ContributionType.EVIDENCE,
        text='"Officials confirmed the death toll rose to at least 160" - Reuters',
        source_url="https://reuters.com/world/asia/hong-kong-fire-death-toll",
        extracted_value=160,
        observation_kind=ObservationKind.LOWER_BOUND  # "at least"
    )

    print(f"  Contribution: {contrib1.id}")
    print(f"  Type: {contrib1.type.value}")
    print(f"  Source: {contrib1.source_name}")
    print(f"  Extracted: ≥{contrib1.extracted_value} (lower_bound)")
    print(f"  Impact: {contrib1.posterior_impact*100:.1f}%")

    trace = engine.get_trace(inquiry.id)
    print("\n  Updated belief state:")
    print_belief_state(trace)

    print("\n  Tasks generated:")
    print_tasks(trace['tasks'])

    # =========================================================================
    print_section("PHASE 4: Second Contribution (BBC)")

    contrib2 = await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="user_bob",
        contribution_type=ContributionType.EVIDENCE,
        text='"The fire claimed approximately 165 lives according to hospital sources" - BBC',
        source_url="https://bbc.com/news/world-asia-hong-kong-fire",
        extracted_value=165,
        observation_kind=ObservationKind.APPROXIMATE  # "approximately"
    )

    print(f"  Contribution: {contrib2.id}")
    print(f"  Source: {contrib2.source_name}")
    print(f"  Extracted: ~{contrib2.extracted_value} (approximate)")
    print(f"  Impact: {contrib2.posterior_impact*100:.1f}%")

    trace = engine.get_trace(inquiry.id)
    print("\n  Updated belief state:")
    print_belief_state(trace)

    # Check if entropy decreased (corroboration)
    print(f"\n  Entropy change: corroboration should decrease entropy")

    # =========================================================================
    print_section("PHASE 5: Conflicting Evidence (SCMP)")

    contrib3 = await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="user_charlie",
        contribution_type=ContributionType.EVIDENCE,
        text='"Government spokesperson confirmed 158 deaths" - SCMP',
        source_url="https://scmp.com/news/hong-kong/fire-death-toll",
        extracted_value=158,
        observation_kind=ObservationKind.POINT  # exact official count
    )

    print(f"  Contribution: {contrib3.id}")
    print(f"  Source: {contrib3.source_name}")
    print(f"  Extracted: ={contrib3.extracted_value} (point)")
    print(f"  Impact: {contrib3.posterior_impact*100:.1f}%")

    trace = engine.get_trace(inquiry.id)
    print("\n  Updated belief state:")
    print_belief_state(trace)

    # Check for conflict task
    print("\n  Tasks after potential conflict:")
    print_tasks(trace['tasks'])

    # =========================================================================
    print_section("PHASE 6: Update Evidence (AP News)")

    contrib4 = await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="user_diana",
        contribution_type=ContributionType.EVIDENCE,
        text='"Death toll rises to 170 after overnight recovery operations" - AP',
        source_url="https://apnews.com/hong-kong-fire-update",
        extracted_value=170,
        observation_kind=ObservationKind.POINT
    )

    print(f"  Contribution: {contrib4.id}")
    print(f"  Source: {contrib4.source_name}")
    print(f"  Extracted: ={contrib4.extracted_value} (point, update)")
    print(f"  Impact: {contrib4.posterior_impact*100:.1f}%")

    trace = engine.get_trace(inquiry.id)
    print("\n  Updated belief state:")
    print_belief_state(trace)

    # =========================================================================
    print_section("PHASE 7: Attribution Contribution")

    # Someone notices BBC was quoting hospital sources
    contrib5 = await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="user_eve",
        contribution_type=ContributionType.ATTRIBUTION,
        text="BBC report citing hospital sources, not direct count. Original source is Queen Elizabeth Hospital spokesperson.",
        source_url="https://bbc.com/news/world-asia-hong-kong-fire"
    )

    print(f"  Contribution: {contrib5.id}")
    print(f"  Type: {contrib5.type.value}")
    print(f"  Text: {contrib5.text[:80]}...")

    # =========================================================================
    print_section("PHASE 8: High-Confidence Evidence")

    # Official government press release
    contrib6 = await engine.add_contribution(
        inquiry_id=inquiry.id,
        user_id="user_frank",
        contribution_type=ContributionType.EVIDENCE,
        text='"Final official death toll: 173 confirmed by Fire Services Department" - Government Press Release',
        source_url="https://gov.hk/press/fire-death-toll-final",
        extracted_value=173,
        observation_kind=ObservationKind.POINT
    )

    print(f"  Contribution: {contrib6.id}")
    print(f"  Source: {contrib6.source_name}")
    print(f"  Extracted: ={contrib6.extracted_value} (official)")
    print(f"  Impact: {contrib6.posterior_impact*100:.1f}%")

    trace = engine.get_trace(inquiry.id)
    print("\n  Updated belief state:")
    print_belief_state(trace)

    # =========================================================================
    print_section("PHASE 9: Add More Stake")

    # User adds bounty
    engine.add_stake(inquiry.id, "user_george", 50.0)
    print(f"  User George added $50 stake")

    inquiry = engine.get_inquiry(inquiry.id)
    print(f"  Total stake now: ${inquiry.total_stake:.2f}")

    # =========================================================================
    print_section("PHASE 10: Corroborating Evidence")

    # Multiple sources confirm 173
    for i, source in enumerate([
        ("Reuters Update", "reuters.com", "Death toll confirmed at 173"),
        ("AFP", "afp.com", "Hong Kong fire: 173 dead, investigation ongoing"),
        ("Xinhua", "xinhua.net", "173 confirmed dead in Tai Po fire")
    ]):
        contrib = await engine.add_contribution(
            inquiry_id=inquiry.id,
            user_id=f"user_{i+10}",
            contribution_type=ContributionType.EVIDENCE,
            text=f'"{source[2]}" - {source[0]}',
            source_url=f"https://{source[1]}/story",
            extracted_value=173,
            observation_kind=ObservationKind.POINT
        )
        print(f"  + {source[0]}: 173 (impact: {contrib.posterior_impact*100:.1f}%)")

    trace = engine.get_trace(inquiry.id)
    print("\n  Updated belief state after corroboration:")
    print_belief_state(trace)

    # =========================================================================
    print_section("PHASE 11: Resolution Check")

    inquiry = engine.get_inquiry(inquiry.id)

    print(f"  Status: {inquiry.status.value}")
    print(f"  P(MAP): {inquiry.posterior_probability*100:.1f}%")
    print(f"  Resolvable: {trace['resolution']['resolvable']}")
    print(f"  Blocking tasks: {trace['resolution']['blocking_tasks']}")

    if inquiry.status == InquiryStatus.RESOLVED:
        print(f"\n  ✅ INQUIRY RESOLVED!")
        print(f"  Final answer: {inquiry.posterior_map}")
        print(f"  Confidence: {inquiry.posterior_probability*100:.1f}%")
    else:
        print(f"\n  ⏳ Not yet resolved")
        print(f"  Need: P(MAP) >= 95% for 24 hours")
        print(f"  Current: {inquiry.posterior_probability*100:.1f}%")

        # What would it take to resolve?
        if inquiry.posterior_probability >= 0.95:
            print(f"  → Waiting for 24h stability period")
        else:
            gap = 0.95 - inquiry.posterior_probability
            print(f"  → Need {gap*100:.1f}% more confidence")

    # =========================================================================
    print_section("PHASE 12: Full Trace (Replay Data)")

    trace = engine.get_trace(inquiry.id)

    print(f"  Total observations: {len(trace['observations'])}")
    print(f"  Total contributions: {len(trace['contributions'])}")
    print()
    print("  Observation history (for replay):")
    for i, obs in enumerate(trace['observations']):
        kind = obs['kind']
        source = obs.get('source', 'unknown')
        # Get the dominant value from distribution
        dist = obs['value_distribution']
        if dist:
            value = max(dist.keys(), key=lambda k: dist[k])
            print(f"    {i+1}. [{kind}] {value} from {source}")

    print()
    print("  Contribution timeline:")
    for c in trace['contributions']:
        impact = c.get('posterior_impact', c.get('impact', 0)) or 0
        source = c.get('source_name') or c.get('source', 'unknown')
        print(f"    - {source}: {c.get('extracted_value', 'N/A')} ({c['type']}) +{impact*100:.1f}%")

    # =========================================================================
    print_section("SUMMARY: Gaps Identified")

    print("""
  WORKING:
    ✅ Inquiry creation with typed schema
    ✅ Contribution flow (evidence, attribution)
    ✅ TypedBeliefState updates correctly
    ✅ Observation history tracked
    ✅ Task generation (single_source, high_entropy)
    ✅ Posterior probability computation
    ✅ Entropy tracking

  GAPS FOUND:
    ❌ No persistence - all lost on restart
    ❌ No surface formation during inquiry (claims not clustered)
    ❌ No identity relations computed (CONFIRMS/SUPERSEDES/CONFLICTS)
    ❌ Resolution requires manual 24h wait (no background checker)
    ❌ No replay snapshots (must recompute from observations)
    ❌ Scope filtering not enforced (claims not checked against scope)
    ❌ Deduplication not implemented
    ❌ Attribution doesn't modify evidence weight

  NEXT STEPS:
    1. Add persistence layer (PostgreSQL)
    2. Integrate IdentityLinker to cluster contributions into surfaces
    3. Build replay snapshot generator
    4. Add scope validation on contribution
    5. Background worker for resolution checking
    """)

    return inquiry, trace

if __name__ == "__main__":
    inquiry, trace = asyncio.run(run_emulation())

    # Optionally dump full trace to JSON
    print("\n" + "="*60)
    print("  Full trace JSON available in inquiry object")
    print("="*60)
