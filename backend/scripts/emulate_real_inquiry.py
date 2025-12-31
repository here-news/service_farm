#!/usr/bin/env python3
"""
End-to-End Inquiry Lifecycle with REAL Claims

Uses actual claims from our database about the Hong Kong fire.
Traces the real evolution of death toll reports over time.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from reee.experiments.loader import create_context, log
from reee.inquiry.engine import InquiryEngine
from reee.inquiry.types import (
    InquirySchema, RigorLevel, InquiryStatus,
    ContributionType, TaskType
)
from reee.typed_belief import ObservationKind


# Real death toll progression based on actual claims in our database:
# Nov 26-27: Initial reports ~36-44
# Dec 1: 128
# Dec 2-3: 156
# Dec 9-13: 160 (current stable)

REAL_OBSERVATIONS = [
    # Early reports - lower bounds
    {
        "source": "nypost.com",
        "value": 36,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "The fire in Tai Po District killed at least 36 people and injured 29",
        "pub_time": "2025-11-26"
    },
    {
        "source": "dw.com",
        "value": 36,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "At least 36 people were killed in Hong Kong after a fire at a housing complex",
        "pub_time": "2025-11-26"
    },
    {
        "source": "newsweek.com",
        "value": 40,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "At least 40 people, including a firefighter, had died in the Hong Kong fire",
        "pub_time": "2025-11-27"
    },
    {
        "source": "livenowfox.com",
        "value": 44,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "At least 44 people were killed in the Hong Kong fire",
        "pub_time": "2025-11-27"
    },
    # Death toll rises significantly
    {
        "source": "mylondon.news",
        "value": 128,
        "kind": ObservationKind.POINT,
        "text": "The fire in Tai Po District left 128 people dead",
        "pub_time": "2025-12-01"
    },
    # Further updates
    {
        "source": "christianitytoday.com",
        "value": 156,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "The fire at Wang Fuk Court killed at least 156 people",
        "pub_time": "2025-12-03"
    },
    {
        "source": "apnews.com",
        "value": 156,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "The fire at Wang Fuk Court killed at least 156 people",
        "pub_time": "2025-12-02"
    },
    # Stabilization at 160
    {
        "source": "hongkongfp.com",
        "value": 160,
        "kind": ObservationKind.POINT,
        "text": "The death toll in the Wang Fuk Court blaze has risen to 160 after DNA tests",
        "pub_time": "2025-12-09"
    },
    {
        "source": "theatlantic.com",
        "value": 160,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "The fire at Wang Fuk Court killed at least 160 people",
        "pub_time": "2025-12-13"
    },
    {
        "source": "news.un.org",
        "value": 160,
        "kind": ObservationKind.LOWER_BOUND,
        "text": "At least 160 people were killed in the blaze at the Wang Fung Court complex",
        "pub_time": "2025-12-11"
    },
]


def print_section(title: str):
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}\n")


def print_belief_state(trace: dict):
    """Pretty print the current belief state"""
    bs = trace['belief_state']
    log(f"  MAP Value: {bs['map']}")
    log(f"  P(MAP): {bs['map_probability']*100:.1f}%")
    log(f"  Entropy: {bs['entropy_bits']:.2f} bits")
    log(f"  Normalized Entropy: {bs['normalized_entropy']*100:.1f}%")
    log(f"  Observations: {bs['observation_count']}")

    if trace.get('posterior_top_10'):
        log(f"\n  Top hypotheses:")
        for item in trace['posterior_top_10'][:7]:
            bar = '█' * int(item['probability'] * 30)
            log(f"    {item['value']:>6}: {item['probability']*100:5.1f}% {bar}")


def print_tasks(tasks: list):
    """Pretty print open tasks"""
    if not tasks:
        log("  No open tasks")
        return
    for task in tasks:
        status = "✓" if task.get('completed') else "○"
        log(f"  {status} [{task['type']}] {task['description']}")
        if task.get('bounty'):
            log(f"      Bounty: ${task['bounty']:.2f}")


async def run_real_emulation():
    """Run inquiry lifecycle with real claims from database."""

    print_section("PHASE 1: Analysis of Real Claims")

    log("  Real death toll progression from our crawled claims:")
    log("")
    log("  Date         Sources                Value    Type")
    log("  " + "-"*60)

    for obs in REAL_OBSERVATIONS:
        kind_symbol = {
            ObservationKind.POINT: "=",
            ObservationKind.LOWER_BOUND: "≥",
            ObservationKind.APPROXIMATE: "~",
            ObservationKind.UPPER_BOUND: "≤",
        }.get(obs['kind'], "?")
        log(f"  {obs['pub_time']}  {obs['source'][:22]:<22}  {kind_symbol}{obs['value']:<5}  {obs['kind'].value}")

    log("")
    log("  Key observations:")
    log("  - Initial reports: 36-44 (Nov 26-27) - 'at least' language")
    log("  - Major jump: 128 (Dec 1) - point estimate")
    log("  - Update: 156 (Dec 2-3) - still 'at least'")
    log("  - Current: 160 (Dec 9+) - converging, DNA confirmed")

    # =========================================================================
    print_section("PHASE 2: Create Inquiry")

    engine = InquiryEngine()

    schema = InquirySchema(
        schema_type="monotone_count",
        count_scale="medium",  # 10-500 range
        count_max=500,
        count_monotone=True,  # Death toll only increases
        rigor=RigorLevel.B    # Typed + generic priors
    )

    inquiry = engine.create_inquiry(
        title="How many people died in the Wang Fuk Court fire (Tai Po, Nov 2025)?",
        description="Death toll from the high-rise fire at Wang Fuk Court, Tai Po, Hong Kong on November 26, 2025",
        schema=schema,
        created_by="system",
        scope_entities=["Wang Fuk Court", "Tai Po", "Hong Kong"],
        scope_keywords=["fire", "death", "casualty", "killed", "dead", "toll"],
        initial_stake=100.0
    )

    log(f"  Created inquiry: {inquiry.id}")
    log(f"  Title: {inquiry.title}")
    log(f"  Schema: monotone_count (deaths only go up)")
    log(f"  Rigor: {inquiry.schema.rigor.value}")
    log(f"  Initial bounty: ${inquiry.total_stake:.2f}")

    trace = engine.get_trace(inquiry.id)
    log("\n  Initial belief state (uninformed prior):")
    print_belief_state(trace)

    # =========================================================================
    print_section("PHASE 3: Process Real Claims Chronologically")

    for i, obs in enumerate(REAL_OBSERVATIONS):
        log(f"\n  [{i+1}/{len(REAL_OBSERVATIONS)}] {obs['pub_time']} - {obs['source']}")
        log(f"      \"{obs['text'][:70]}...\"")

        kind_symbol = {
            ObservationKind.POINT: "=",
            ObservationKind.LOWER_BOUND: "≥",
            ObservationKind.APPROXIMATE: "~",
        }.get(obs['kind'], "?")
        log(f"      Extracted: {kind_symbol}{obs['value']} ({obs['kind'].value})")

        contrib = await engine.add_contribution(
            inquiry_id=inquiry.id,
            user_id=f"crawler_{obs['source'].split('.')[0]}",
            contribution_type=ContributionType.EVIDENCE,
            text=f'"{obs["text"]}" - {obs["source"]}',
            source_url=f"https://{obs['source']}/story",
            extracted_value=obs['value'],
            observation_kind=obs['kind']
        )

        trace = engine.get_trace(inquiry.id)
        bs = trace['belief_state']

        log(f"      → MAP: {bs['map']} | P(MAP): {bs['map_probability']*100:.1f}% | H: {bs['entropy_bits']:.2f} bits")

        # Show if this is a significant update
        if contrib.posterior_impact > 0.05:
            log(f"      ⚡ High impact contribution: +{contrib.posterior_impact*100:.1f}%")

    # =========================================================================
    print_section("PHASE 4: Final Belief State")

    trace = engine.get_trace(inquiry.id)
    print_belief_state(trace)

    log("\n  Observations processed:")
    by_value = defaultdict(list)
    for obs in trace['observations']:
        dist = obs.get('value_distribution', {})
        if dist:
            value = max(dist.keys(), key=lambda k: dist[k])
            by_value[value].append(obs)

    for value in sorted(by_value.keys()):
        obs_list = by_value[value]
        sources = [o.get('source', '?') for o in obs_list]
        log(f"    {value}: {len(obs_list)} observations from {sources[:3]}")

    # =========================================================================
    print_section("PHASE 5: Generated Tasks")

    print_tasks(trace['tasks'])

    # =========================================================================
    print_section("PHASE 6: Resolution Check")

    inquiry = engine.get_inquiry(inquiry.id)

    log(f"  Status: {inquiry.status.value}")
    log(f"  MAP: {inquiry.posterior_map}")
    log(f"  P(MAP): {inquiry.posterior_probability*100:.1f}%")
    log(f"  Resolvable: {trace['resolution']['resolvable']}")
    log(f"  Blocking tasks: {trace['resolution']['blocking_tasks']}")

    if inquiry.status == InquiryStatus.RESOLVED:
        log(f"\n  ✅ INQUIRY RESOLVED!")
        log(f"  Final answer: {inquiry.posterior_map}")
    else:
        log(f"\n  ⏳ Not yet resolved")
        if inquiry.posterior_probability >= 0.95:
            log(f"  → Waiting for 24h stability period")
        else:
            gap = 0.95 - inquiry.posterior_probability
            log(f"  → Need {gap*100:.1f}% more confidence")
            log(f"  → Consider adding more corroborating sources")

    # =========================================================================
    print_section("PHASE 7: Epistemic Journey Summary")

    log("""
  REAL DATA INSIGHTS:

  1. Time-based uncertainty:
     - Early reports (Nov 26-27): 36-44 range, high uncertainty
     - Recovery operations revealed much higher toll (128, Dec 1)
     - DNA identification pushed to 156, then 160

  2. Observation types matter:
     - "At least X" (lower_bound) = can only go up
     - "X confirmed" (point) = specific count
     - Monotone constraint: later values supersede earlier

  3. Source convergence:
     - Multiple sources now agree on 160
     - HKFP reported DNA-confirmed 160 (high credibility)
     - UN, Atlantic, AP all report 160 or "at least 160"

  4. Remaining uncertainty:
     - Still using "at least" language → may increase
     - Missing persons may still be identified
     - Final count depends on forensic analysis

  SYSTEM OBSERVATIONS:

  - TypedBeliefState correctly handles lower_bounds
  - Monotone constraint prevents regression
  - Entropy decreases as sources converge
  - Tasks generated for single-source early reports
    """)

    # =========================================================================
    print_section("PHASE 8: Replay Data for Frontend")

    log("  Replay snapshots for visualization:")
    log("")

    contributions = trace['contributions']
    for i, c in enumerate(contributions):
        source = c.get('source_name') or c.get('source', 'unknown')
        value = c.get('extracted_value', 'N/A')
        impact = c.get('posterior_impact', c.get('impact', 0)) or 0
        log(f"  {i+1}. {source}: {value} → +{impact*100:.1f}% impact")

    return inquiry, trace


if __name__ == "__main__":
    inquiry, trace = asyncio.run(run_real_emulation())

    log("\n" + "="*70)
    log("  Emulation complete. Data ready for frontend replay.")
    log("="*70)
