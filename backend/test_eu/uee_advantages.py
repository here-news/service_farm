"""
UEE ADVANTAGES ANALYSIS

1. Metrics for public display (mass, heat, threshold)
2. "Global newsroom" quality assessment
3. Unique UEE capabilities (cross-event claims, entity bridges, etc.)
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from datetime import datetime, timedelta
from math import log, exp, log2
import json
import numpy as np
from load_graph import load_snapshot


# =============================================================================
# 1. METRICS FOR PUBLIC DISPLAY
# =============================================================================

def compute_display_metrics(events, claims, pages):
    """
    Compute metrics that determine what events show to public.

    Candidates:
    - mass: credibility-weighted evidence
    - heat: recency-weighted claim rate
    - diversity: distinct source count
    - coherence: internal agreement
    - entropy: uncertainty level
    """
    print("=" * 80)
    print("1. METRICS FOR PUBLIC DISPLAY")
    print("=" * 80)

    SOURCE_CRED = {
        'bbc.com': 0.90, 'reuters.com': 0.88, 'apnews.com': 0.88,
        'theguardian.com': 0.85, 'dw.com': 0.85, 'cnn.com': 0.80,
        'scmp.com': 0.82, 'aljazeera.com': 0.80, 'newsweek.com': 0.70,
        'nypost.com': 0.60, 'foxnews.com': 0.65, 'default': 0.5
    }

    def get_cred(url):
        if not url:
            return 0.5
        for domain, cred in SOURCE_CRED.items():
            if domain in url.lower():
                return cred
        return 0.5

    # Simulate events from our data
    event_metrics = []

    # Group claims by topic (using entity overlap as proxy)
    # For demo, use keyword-based grouping
    topics = {
        'hk_fire': lambda c: 'fire' in c.text.lower() and ('hong kong' in c.text.lower() or 'tai po' in c.text.lower()),
        'jimmy_lai': lambda c: 'lai' in c.text.lower() or 'apple daily' in c.text.lower(),
        'bondi': lambda c: 'bondi' in c.text.lower() or 'westfield' in c.text.lower(),
        'brown': lambda c: 'brown' in c.text.lower() and 'university' in c.text.lower(),
        'venezuela': lambda c: 'venezuela' in c.text.lower() and ('tanker' in c.text.lower() or 'oil' in c.text.lower()),
        'do_kwon': lambda c: 'kwon' in c.text.lower() or 'terra' in c.text.lower(),
        'charlie_kirk': lambda c: 'kirk' in c.text.lower() and 'charlie' in c.text.lower(),
    }

    now = datetime.utcnow()

    for topic_name, filter_fn in topics.items():
        topic_claims = [c for c in claims.values() if filter_fn(c)]
        if not topic_claims:
            continue

        # Mass: sum of source credibilities
        mass = 0
        sources = set()
        for c in topic_claims:
            page = pages.get(c.page_id)
            url = page.url if page else ""
            cred = get_cred(url)
            mass += cred
            if url:
                domain = url.split('/')[2].replace('www.', '') if '/' in url else url
                sources.add(domain)

        # Heat: recency-weighted (simulated - assume claims spread over 48h)
        # heat = Σ exp(-λt) where t is hours since claim
        # For demo, use claim count as proxy (real system would use timestamps)
        claim_count = len(topic_claims)
        heat = claim_count  # Simplified - real: sum of recency weights

        # Diversity: distinct sources (more sources = more verified)
        diversity = len(sources)

        # Coherence: would need edges - use 0.85 as placeholder
        coherence = 0.85

        # Entropy: would need contradictions - use 0.2 as placeholder
        entropy = 0.2

        event_metrics.append({
            'topic': topic_name,
            'claims': claim_count,
            'mass': mass,
            'heat': heat,
            'sources': diversity,
            'coherence': coherence,
            'entropy': entropy
        })

    # Sort by mass
    event_metrics.sort(key=lambda x: -x['mass'])

    print("""
    PROPOSED METRICS:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  MASS (Evidence Weight)                                             │
    │                                                                     │
    │  mass = Σ source_credibility_i                                      │
    │                                                                     │
    │  Higher mass = more credible evidence accumulated                   │
    │  BBC claim worth more than random blog                              │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  HEAT (Timeliness)                                                  │
    │                                                                     │
    │  heat = Σ exp(-λ × hours_since_claim)                               │
    │                                                                     │
    │  λ = 0.1 → half-life ≈ 7 hours                                      │
    │  Recent claims contribute more than old ones                        │
    │  "Breaking" events have high heat                                   │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  SOURCE DIVERSITY                                                   │
    │                                                                     │
    │  diversity = |distinct source domains|                              │
    │                                                                     │
    │  More sources = more cross-verified                                 │
    │  Single-source stories are suspect                                  │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  DISPLAY SCORE (Combined)                                           │
    │                                                                     │
    │  display = mass × heat_factor × log(1 + sources)                    │
    │                                                                     │
    │  Where heat_factor = min(1.5, 0.5 + heat/max_heat)                  │
    │  Boosts breaking news, penalizes single-source                      │
    └─────────────────────────────────────────────────────────────────────┘

    CURRENT EVENTS RANKED:
    """)

    max_heat = max(e['heat'] for e in event_metrics)

    for e in event_metrics:
        heat_factor = min(1.5, 0.5 + e['heat'] / max_heat)
        display = e['mass'] * heat_factor * log(1 + e['sources'])
        e['display'] = display

        print(f"    {e['topic']:15} | mass={e['mass']:5.1f} | heat={e['heat']:3} | sources={e['sources']:2} | DISPLAY={display:6.1f}")

    print("""
    THRESHOLD RECOMMENDATIONS:

    ┌────────────────┬─────────────────────────────────────────────────────┐
    │  TIER          │  CRITERIA                                           │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │  HEADLINE      │  display > 100 AND sources >= 5                     │
    │                │  Top stories, homepage-worthy                       │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │  SIGNIFICANT   │  display > 50 AND sources >= 3                      │
    │                │  Important stories, category pages                  │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │  EMERGING      │  display > 20 AND heat > median                     │
    │                │  Breaking/developing, watchlist                     │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │  INTERNAL      │  display < 20                                       │
    │                │  Not yet newsworthy, keep monitoring                │
    └────────────────┴─────────────────────────────────────────────────────┘
    """)

    return event_metrics


# =============================================================================
# 2. GLOBAL NEWSROOM QUALITY
# =============================================================================

def assess_newsroom_quality(event_metrics):
    """
    Does our system capture editorial judgment?
    Compare algorithmic ranking to expected newsroom priorities.
    """
    print("\n" + "=" * 80)
    print("2. GLOBAL NEWSROOM QUALITY ASSESSMENT")
    print("=" * 80)

    # Expected newsroom priorities (major stories first)
    newsroom_priority = [
        ('hk_fire', 'BREAKING - mass casualty'),
        ('jimmy_lai', 'ONGOING - press freedom, diplomatic'),
        ('bondi', 'BREAKING - mass casualty'),
        ('brown', 'BREAKING - campus violence'),
        ('venezuela', 'INTERNATIONAL - sanctions, geopolitics'),
        ('charlie_kirk', 'NATIONAL - political violence'),
        ('do_kwon', 'BUSINESS - crypto/finance'),
    ]

    # Our algorithmic ranking
    algo_order = [e['topic'] for e in sorted(event_metrics, key=lambda x: -x['display'])]

    # Newsroom expected order
    newsroom_order = [t[0] for t in newsroom_priority]

    print("""
    COMPARISON: Algorithm vs Editorial Judgment

    Newsroom Priority          | Algorithm Ranking
    (Expected)                 | (display score)
    ─────────────────────────────────────────────────
    """)

    matches = 0
    for i, (expected, algo) in enumerate(zip(newsroom_order, algo_order)):
        match = "✓" if expected == algo else "✗"
        if expected == algo:
            matches += 1
        print(f"    {i+1}. {expected:15} | {algo:15} {match}")

    accuracy = matches / len(newsroom_order)

    print(f"""
    ALIGNMENT SCORE: {matches}/{len(newsroom_order)} = {100*accuracy:.0f}%

    ANALYSIS:
    """)

    if accuracy >= 0.7:
        print("""
    ✓ HIGH ALIGNMENT with editorial judgment

    The algorithm captures "newsworthiness" because:
    - Mass correlates with story importance (more coverage = more newsworthy)
    - Source diversity correlates with verification (editors check multiple sources)
    - Heat correlates with breaking news priority

    This is NOT circular reasoning - we're measuring the SAME signals editors use:
    - How much credible coverage exists?
    - How many independent sources confirm?
    - How recent is the development?
    """)
    else:
        print("""
    ⚠ MODERATE ALIGNMENT - may need tuning

    Possible issues:
    - Heat weight may be too high/low
    - Source diversity threshold may need adjustment
    - Some story types need different weights (business vs breaking)
    """)

    print("""
    WHAT "GLOBAL NEWSROOM" MEANS:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Traditional newsroom: One editor's judgment (local, biased)        │
    │                                                                     │
    │  Global newsroom: Aggregate signal from ALL sources worldwide       │
    │                                                                     │
    │  When BBC, Reuters, AP, Guardian, SCMP, Al Jazeera ALL cover        │
    │  the same story with similar facts → HIGH CONFIDENCE                │
    │                                                                     │
    │  This is Jaynesian: P(true|many independent sources) >> P(true|one) │
    └─────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# 3. UEE UNIQUE ADVANTAGES
# =============================================================================

def analyze_uee_advantages(claims, entities, pages):
    """
    What can UEE do that current system can't?
    """
    print("\n" + "=" * 80)
    print("3. UEE UNIQUE ADVANTAGES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # ADVANTAGE 1: Cross-Event Claims
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADVANTAGE 1: CROSS-EVENT CLAIMS                                    │
    └─────────────────────────────────────────────────────────────────────┘

    One claim can contribute evidence to MULTIPLE events.

    Current system: claim → ONE event (page routing)
    UEE: claim → MULTIPLE events (via shared entities/semantics)

    EXAMPLE:
    """)

    # Find claims that mention both HK government AND fire-related topics
    cross_claims = []
    for c in claims.values():
        text = c.text.lower()

        # Mentions HK government figure
        has_gov = any(x in text for x in ['john lee', 'carrie lam', 'hong kong government', 'authorities'])

        # Mentions fire
        has_fire = 'fire' in text and ('tai po' in text or 'wang fuk' in text)

        # Mentions Jimmy Lai / press freedom
        has_lai = 'lai' in text or 'press freedom' in text or 'apple daily' in text

        if has_gov and has_fire:
            cross_claims.append(('fire+gov', c.text[:80]))
        elif has_gov and has_lai:
            cross_claims.append(('lai+gov', c.text[:80]))

    print("    Claims that bridge events:")
    for event_pair, text in cross_claims[:5]:
        print(f"    [{event_pair}] {text}...")

    print("""
    WHY THIS MATTERS:

    A claim about "Hong Kong government's emergency response" contributes to:
    - HK Fire event (response quality)
    - Hong Kong governance narrative (competence)
    - Potentially: contrast with Jimmy Lai prosecution (priorities)

    The SAME claim is EVIDENCE for multiple propositions.
    Current system forces a choice; UEE can propagate to all.
    """)

    # -------------------------------------------------------------------------
    # ADVANTAGE 2: Entity Bridges (Knowledge Graph)
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADVANTAGE 2: ENTITY BRIDGES (Knowledge Graph)                      │
    └─────────────────────────────────────────────────────────────────────┘

    Entities that appear in multiple events create NARRATIVE THREADS.
    """)

    # Count entity frequency across topics
    entity_topics = defaultdict(set)

    for c in claims.values():
        text = c.text.lower()
        topic = None
        if 'fire' in text and ('tai po' in text or 'hong kong' in text):
            topic = 'fire'
        elif 'lai' in text:
            topic = 'lai'
        elif 'bondi' in text:
            topic = 'bondi'
        elif 'kwon' in text:
            topic = 'kwon'

        if topic:
            for eid in c.entity_ids:
                entity_topics[eid].add(topic)

    # Find bridge entities
    bridges = [(eid, topics) for eid, topics in entity_topics.items() if len(topics) > 1]

    print("    Bridge entities (appear in 2+ events):")
    for eid, topics in bridges[:10]:
        ent = entities.get(eid)
        name = ent.canonical_name if ent else eid
        print(f"    {name:30} → {', '.join(topics)}")

    print("""
    WHY THIS MATTERS:

    Bridge entities reveal META-NARRATIVES:
    - "Hong Kong" bridges Fire + Jimmy Lai → "What's happening in HK?"
    - "John Lee" bridges multiple events → "How is leadership performing?"

    UEE can surface these cross-event narratives automatically.
    Current system treats events as isolated.
    """)

    # -------------------------------------------------------------------------
    # ADVANTAGE 3: Source Credibility Propagation
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADVANTAGE 3: SOURCE CREDIBILITY PROPAGATION                        │
    └─────────────────────────────────────────────────────────────────────┘

    If a source is wrong on Event A, update its credibility for Event B.

    EXAMPLE:
    - Source X claims "death toll 500" for HK fire
    - Later confirmed: death toll 160
    - Source X was WRONG by 3x
    - Bayesian update: P(X credible) decreases
    - X's claims on OTHER events now weighted less

    This is JAYNESIAN: evidence about source quality propagates.
    Current system: static credibility priors, no learning.
    """)

    # -------------------------------------------------------------------------
    # ADVANTAGE 4: Contradiction as Signal
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADVANTAGE 4: CONTRADICTION AS SIGNAL                               │
    └─────────────────────────────────────────────────────────────────────┘

    Contradictions ACROSS events can reveal systemic patterns.

    EXAMPLE:
    - Event A: Government claims "swift response"
    - Event B: Government claims "followed all protocols"
    - Event C: Victims claim "waited hours for help"

    Cross-event contradiction pattern → systemic credibility issue

    UEE can detect: "Government official statements have high entropy"
    Current system: each event isolated, pattern invisible.
    """)

    # -------------------------------------------------------------------------
    # ADVANTAGE 5: Temporal Evolution Tracking
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADVANTAGE 5: TEMPORAL EVOLUTION TRACKING                           │
    └─────────────────────────────────────────────────────────────────────┘

    Track how events DEVELOP over time with Jaynesian updates.

    HK FIRE EVOLUTION:

    T0: "Fire reported at Wang Fuk Court"
        mass=2, entropy=high (what happened?)

    T1: "36 killed" (multiple sources)
        mass=15, entropy=medium (confirmed fire, death toll uncertain)

    T2: "Death toll rises to 128"
        mass=40, entropy=low for count (converging)

    T3: "160 confirmed dead"
        mass=80, entropy=very low (consensus reached)

    Each stage has DIFFERENT confidence levels.
    UEE tracks this; current system overwrites.
    """)

    # -------------------------------------------------------------------------
    # ADVANTAGE 6: Claim-Level Granularity
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ADVANTAGE 6: CLAIM-LEVEL GRANULARITY                               │
    └─────────────────────────────────────────────────────────────────────┘

    Process claims individually, not pages.

    CURRENT PROBLEM (50% orphans):
    - Page has 10 claims about different aspects
    - Page routes to Event A based on dominant topic
    - 3 claims actually belong to Event B → ORPHANED

    UEE SOLUTION:
    - Each claim routes independently
    - Same page can contribute to multiple events
    - No orphans by design

    EXAMPLE:
    Article "Hong Kong crisis deepens" might have:
    - 5 claims about fire → Fire event
    - 3 claims about Jimmy Lai → Lai event
    - 2 claims about economy → Economy event

    Current: all 10 go to one event
    UEE: proper distribution
    """)

    # -------------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  SUMMARY: UEE vs CURRENT SYSTEM                                     │
    └─────────────────────────────────────────────────────────────────────┘

    Feature                  │ Current System │ UEE
    ─────────────────────────┼────────────────┼─────────────────────────
    Routing unit             │ Page           │ Claim
    Cross-event claims       │ No             │ Yes
    Entity knowledge graph   │ Per-event      │ Global
    Source learning          │ Static priors  │ Bayesian update
    Contradiction detection  │ Within event   │ Across events
    Temporal tracking        │ Overwrites     │ Evolution chain
    Orphan claims            │ 50%            │ 0%
    Event merging            │ Manual         │ Automatic (same rules)
    Jaynesian quantities     │ None           │ Mass, entropy, coherence

    THE KEY INSIGHT:

    UEE treats claims as EVIDENCE in an EPISTEMIC GRAPH.
    Current system treats pages as DOCUMENTS to be FILED.

    Same signals, fundamentally different paradigm.
    """)


def main():
    snapshot = load_snapshot()

    event_metrics = compute_display_metrics(
        {},  # events would come from UEE
        snapshot.claims,
        snapshot.pages
    )

    assess_newsroom_quality(event_metrics)

    analyze_uee_advantages(
        snapshot.claims,
        snapshot.entities,
        snapshot.pages
    )


if __name__ == "__main__":
    main()
