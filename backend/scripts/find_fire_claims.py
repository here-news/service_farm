#!/usr/bin/env python3
"""
Find real claims about the Hong Kong fire from our database.
Analyze actual uncertainties and conflicting reports.
"""

import asyncio
import sys
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from reee.experiments.loader import create_context, log


async def find_fire_event(ctx):
    """Find the Hong Kong fire event in Neo4j."""
    events = await ctx.neo4j._execute_read('''
        MATCH (e:Event)
        WHERE e.canonical_name =~ '(?i).*hong kong.*fire.*'
           OR e.canonical_name =~ '(?i).*tai po.*'
           OR e.canonical_name =~ '(?i).*wang fuk.*'
        RETURN e.id as id, e.canonical_name as name, e.summary as summary
        LIMIT 10
    ''', {})
    return events


async def find_fire_claims(ctx):
    """Find all claims mentioning the Hong Kong fire with death toll numbers."""

    # First, let's find claims with death-related keywords
    claims = await ctx.neo4j._execute_read('''
        MATCH (c:Claim)
        WHERE (c.text =~ '(?i).*hong kong.*fire.*' OR c.text =~ '(?i).*tai po.*' OR c.text =~ '(?i).*wang fuk.*')
        AND (c.text =~ '(?i).*(dead|death|die|kill|fatal|casualt|victim).*')
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        RETURN c.id as id, c.text as text, p.domain as source, p.pub_time as pub_time
        ORDER BY p.pub_time DESC
        LIMIT 50
    ''', {})

    return claims


def extract_numbers(text):
    """Extract numbers that might be death tolls from text."""
    patterns = [
        r'(\d+)\s*(?:people\s+)?(?:dead|died|killed|deaths?|fatalities|casualties|victims)',
        r'(?:death toll|toll|killed|dead).*?(\d+)',
        r'(?:at least|more than|over|about|approximately|around|nearly)\s*(\d+)',
        r'(\d+)\s*(?:were|have been|had been)\s*(?:killed|dead)',
        r'rises?\s*to\s*(\d+)',
    ]

    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for m in matches:
            try:
                n = int(m)
                if 1 <= n <= 1000:  # Reasonable death toll range
                    numbers.append(n)
            except:
                pass

    return list(set(numbers))


def classify_observation_kind(text):
    """Determine observation kind from text patterns."""
    text_lower = text.lower()

    if 'at least' in text_lower or 'more than' in text_lower or 'over' in text_lower:
        return 'lower_bound'
    elif 'approximately' in text_lower or 'about' in text_lower or 'around' in text_lower or 'nearly' in text_lower:
        return 'approximate'
    elif 'up to' in text_lower or 'no more than' in text_lower:
        return 'upper_bound'
    elif 'between' in text_lower and 'and' in text_lower:
        return 'interval'
    else:
        return 'point'


async def main():
    log("="*70)
    log("  Finding Real Claims About Hong Kong Fire")
    log("="*70)

    ctx = await create_context()

    try:
        # Find events
        log("\n1. Looking for Hong Kong fire events...")
        events = await find_fire_event(ctx)

        if events:
            log(f"\n   Found {len(events)} related events:")
            for e in events:
                log(f"   - {e['id']}: {e['name']}")
                if e.get('summary'):
                    log(f"     Summary: {e['summary'][:100]}...")
        else:
            log("   No events found with 'hong kong fire' in name")

        # Find claims
        log("\n2. Finding claims with death toll mentions...")
        claims = await find_fire_claims(ctx)

        log(f"\n   Found {len(claims)} claims mentioning fire + casualties")

        # Analyze numbers
        log("\n3. Extracting death toll numbers from claims...")

        by_source = defaultdict(list)
        all_extractions = []

        for c in claims:
            numbers = extract_numbers(c['text'])
            if numbers:
                obs_kind = classify_observation_kind(c['text'])
                for n in numbers:
                    extraction = {
                        'source': c['source'],
                        'value': n,
                        'kind': obs_kind,
                        'text': c['text'][:150],
                        'pub_time': c['pub_time']
                    }
                    all_extractions.append(extraction)
                    by_source[c['source']].append(extraction)

        log(f"\n   Extracted {len(all_extractions)} death toll values:")

        # Group by value
        by_value = defaultdict(list)
        for e in all_extractions:
            by_value[e['value']].append(e)

        log("\n   Death toll values reported:")
        for value in sorted(by_value.keys()):
            extractions = by_value[value]
            sources = set(e['source'] for e in extractions)
            log(f"   {value}: {len(extractions)} mentions from {len(sources)} sources")
            for e in extractions[:3]:
                kind_icon = {'point': '=', 'lower_bound': '≥', 'approximate': '~', 'upper_bound': '≤', 'interval': '↔'}.get(e['kind'], '?')
                log(f"      {kind_icon} {e['source']}: \"{e['text'][:80]}...\"")

        # Show by source
        log("\n   By source:")
        for source, extractions in sorted(by_source.items(), key=lambda x: -len(x[1])):
            values = sorted(set(e['value'] for e in extractions))
            log(f"   {source}: {values}")

        # Identify uncertainties
        log("\n" + "="*70)
        log("  UNCERTAINTY ANALYSIS")
        log("="*70)

        unique_values = sorted(by_value.keys())
        if len(unique_values) > 1:
            log(f"\n   Range of reported values: {min(unique_values)} - {max(unique_values)}")
            log(f"   Spread: {max(unique_values) - min(unique_values)}")

            # Most common
            most_common = max(by_value.items(), key=lambda x: len(x[1]))
            log(f"   Most reported value: {most_common[0]} ({len(most_common[1])} mentions)")

            # Conflicts
            log("\n   Potential conflicts:")
            values_list = list(by_value.keys())
            for i, v1 in enumerate(values_list):
                for v2 in values_list[i+1:]:
                    if abs(v1 - v2) > 5:  # Significant difference
                        s1 = set(e['source'] for e in by_value[v1])
                        s2 = set(e['source'] for e in by_value[v2])
                        log(f"   - {v1} vs {v2}: {s1} vs {s2}")

        # Sample claims for inquiry
        log("\n" + "="*70)
        log("  SAMPLE CLAIMS FOR INQUIRY")
        log("="*70)

        for e in all_extractions[:10]:
            log(f"\n   Source: {e['source']}")
            log(f"   Value: {e['value']} ({e['kind']})")
            log(f"   Text: {e['text']}")
            log(f"   Time: {e['pub_time']}")

    finally:
        await ctx.close()


if __name__ == '__main__':
    asyncio.run(main())
