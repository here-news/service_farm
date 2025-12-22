"""
EPISTEMIC DEPTH TEST

Can we extract multi-voice, temporal, attributed narrative from current data?

What we have:
- ~1271 claims from ~120 pages
- Mostly mainstream sources (BBC, Reuters, SCMP, AFP, etc.)
- No user contributions (yet)

What we want to demonstrate:
- Source attribution (who said what)
- Temporal ordering (when they said it)
- Corroboration patterns (who confirms whom)
- Contradiction detection (who disagrees)
- Update sequences (death toll evolution)
- Provenance chains (who quoted whom)
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from datetime import datetime
import re
from load_graph import load_snapshot


def extract_epistemic_structure(claims, pages, entities):
    """Extract the epistemic narrative structure from HK Fire claims"""

    print("=" * 80)
    print("EPISTEMIC DEPTH TEST: Hong Kong Fire Event")
    print("=" * 80)

    # Filter to HK Fire claims
    fire_claims = []
    for c in claims.values():
        text = c.text.lower()
        if ('fire' in text or 'blaze' in text) and \
           ('hong kong' in text or 'tai po' in text or 'wang fuk' in text):
            fire_claims.append(c)

    print(f"\nFound {len(fire_claims)} fire-related claims")

    # =========================================================================
    # 1. VOICE EXTRACTION (Who are the sources?)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. VOICES (Sources in this event)")
    print("=" * 80)

    voices = defaultdict(lambda: {'claims': [], 'type': 'unknown'})

    SOURCE_TYPES = {
        'bbc.com': ('BBC', 'international'),
        'reuters.com': ('Reuters', 'wire'),
        'apnews.com': ('AP', 'wire'),
        'afp.com': ('AFP', 'wire'),
        'scmp.com': ('SCMP', 'local_news'),
        'theguardian.com': ('Guardian', 'international'),
        'dw.com': ('DW', 'international'),
        'aljazeera.com': ('Al Jazeera', 'international'),
        'cnn.com': ('CNN', 'international'),
        'newsweek.com': ('Newsweek', 'magazine'),
        'nypost.com': ('NY Post', 'tabloid'),
        'gov.hk': ('HK Government', 'official'),
        'info.gov.hk': ('HK Government', 'official'),
    }

    for c in fire_claims:
        page = pages.get(c.page_id)
        if not page:
            continue

        url = page.url.lower()
        source_name = 'Unknown'
        source_type = 'unknown'

        for domain, (name, stype) in SOURCE_TYPES.items():
            if domain in url:
                source_name = name
                source_type = stype
                break

        if source_name == 'Unknown':
            # Extract domain
            try:
                domain = url.split('/')[2].replace('www.', '')
                source_name = domain.split('.')[0].upper()
            except:
                pass

        voices[source_name]['claims'].append(c)
        voices[source_name]['type'] = source_type
        voices[source_name]['url'] = page.url

    print(f"\nDistinct voices: {len(voices)}")
    print("\nVoice breakdown:")
    for name, data in sorted(voices.items(), key=lambda x: -len(x[1]['claims'])):
        print(f"  {name:20} [{data['type']:12}] - {len(data['claims']):3} claims")

    # =========================================================================
    # 2. TEMPORAL STRUCTURE (When did they say it?)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. TEMPORAL STRUCTURE")
    print("=" * 80)

    claims_with_time = []
    for c in fire_claims:
        page = pages.get(c.page_id)
        pub_time = None

        # Try to get publication time from page
        if page and hasattr(page, 'pub_time') and page.pub_time:
            pub_time = page.pub_time
        elif c.event_time:
            pub_time = c.event_time

        if pub_time:
            claims_with_time.append((pub_time, c, page))

    # Sort by time
    claims_with_time.sort(key=lambda x: x[0] if x[0] else datetime.min)

    print(f"\nClaims with timestamps: {len(claims_with_time)}/{len(fire_claims)}")

    print("\nTemporal sequence (sample):")
    for i, (time, claim, page) in enumerate(claims_with_time[:15]):
        source = 'Unknown'
        if page:
            for domain, (name, _) in SOURCE_TYPES.items():
                if domain in page.url.lower():
                    source = name
                    break

        time_str = time.strftime('%Y-%m-%d %H:%M') if hasattr(time, 'strftime') else str(time)[:16]
        print(f"  [{time_str}] {source:12} | {claim.text[:60]}...")

    # =========================================================================
    # 3. DEATH TOLL EVOLUTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. VALUE EVOLUTION (Death Toll)")
    print("=" * 80)

    death_toll_pattern = r'(\d+)\s*(?:people\s+)?(?:killed|dead|died|death|fatalities|perished)'
    death_toll_pattern2 = r'(?:death\s+toll|toll)\s*(?:of|:)?\s*(\d+)'
    death_toll_pattern3 = r'(?:killed|claimed)\s+(?:at\s+least\s+)?(\d+)'
    death_toll_pattern4 = r'(?:death\s+toll|toll).*?(?:rose|risen|climbed|reached|hit)\s+(?:to\s+)?(\d+)'

    death_reports = []
    for time, claim, page in claims_with_time:
        text = claim.text.lower()

        count = None
        for pattern in [death_toll_pattern4, death_toll_pattern2, death_toll_pattern, death_toll_pattern3]:
            match = re.search(pattern, text)
            if match:
                count = int(match.group(1))
                break

        if count and count > 5 and count < 500:  # Sanity check
            source = 'Unknown'
            if page:
                for domain, (name, _) in SOURCE_TYPES.items():
                    if domain in page.url.lower():
                        source = name
                        break
            death_reports.append({
                'count': count,
                'time': time,
                'source': source,
                'text': claim.text[:100]
            })

    # Dedupe by count+source
    seen = set()
    unique_reports = []
    for r in death_reports:
        key = (r['count'], r['source'])
        if key not in seen:
            seen.add(key)
            unique_reports.append(r)

    # Sort by count
    unique_reports.sort(key=lambda x: x['count'])

    print(f"\nDeath toll reports found: {len(unique_reports)}")
    print("\nEvolution by source:")

    # Group by source
    by_source = defaultdict(list)
    for r in unique_reports:
        by_source[r['source']].append(r['count'])

    for source, counts in sorted(by_source.items()):
        print(f"  {source:15} reported: {' → '.join(map(str, sorted(set(counts))))}")

    print("\nChronological evolution:")
    for r in sorted(unique_reports, key=lambda x: x['time'] if x['time'] else datetime.min):
        time_str = r['time'].strftime('%m-%d %H:%M') if hasattr(r['time'], 'strftime') else '??'
        print(f"  [{time_str}] {r['source']:12} | {r['count']:3} dead | {r['text'][:50]}...")

    # =========================================================================
    # 4. CORROBORATION PATTERNS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. CORROBORATION PATTERNS")
    print("=" * 80)

    # Find claims that say similar things from different sources
    # Group by key facts
    fact_groups = defaultdict(list)

    for c in fire_claims:
        page = pages.get(c.page_id)
        source = 'Unknown'
        if page:
            for domain, (name, _) in SOURCE_TYPES.items():
                if domain in page.url.lower():
                    source = name
                    break

        text = c.text.lower()

        # Categorize by fact type
        if 'wang fuk' in text or 'wanf fuk' in text:
            fact_groups['location_wang_fuk'].append((source, c.text[:80]))
        if 'tai po' in text:
            fact_groups['location_tai_po'].append((source, c.text[:80]))
        if '160' in text and ('dead' in text or 'killed' in text or 'death' in text):
            fact_groups['death_toll_160'].append((source, c.text[:80]))
        if 'john lee' in text:
            fact_groups['john_lee_response'].append((source, c.text[:80]))
        if 'xi jinping' in text:
            fact_groups['xi_jinping_response'].append((source, c.text[:80]))

    print("\nCorroborated facts (multiple sources agree):")
    for fact, reports in sorted(fact_groups.items(), key=lambda x: -len(x[1])):
        sources = list(set(r[0] for r in reports))
        if len(sources) >= 2:
            print(f"\n  [{fact}] - {len(sources)} sources agree:")
            for source in sources[:5]:
                sample = next(r[1] for r in reports if r[0] == source)
                print(f"    • {source}: \"{sample[:60]}...\"")

    # =========================================================================
    # 5. CONTRADICTION DETECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. CONTRADICTIONS / DIFFERENCES")
    print("=" * 80)

    # Death toll contradictions
    death_counts = set(r['count'] for r in unique_reports)
    if len(death_counts) > 1:
        print(f"\n  Death toll values reported: {sorted(death_counts)}")
        print("  This represents UPDATE SEQUENCE, not contradiction")
        print("  (death toll naturally rises as more victims found)")

    # Look for actual contradictions
    print("\n  Checking for genuine contradictions...")

    # Time of fire start
    time_reports = []
    for c in fire_claims:
        text = c.text.lower()
        # Look for specific times
        time_match = re.search(r'(\d{1,2}[:.]\d{2}\s*(?:am|pm|a\.m\.|p\.m\.))', text)
        if time_match and ('start' in text or 'broke' in text or 'began' in text):
            page = pages.get(c.page_id)
            source = 'Unknown'
            if page:
                for domain, (name, _) in SOURCE_TYPES.items():
                    if domain in page.url.lower():
                        source = name
                        break
            time_reports.append((source, time_match.group(1), c.text[:80]))

    if time_reports:
        times = set(r[1] for r in time_reports)
        if len(times) > 1:
            print(f"\n  ⚠️ POSSIBLE CONTRADICTION: Fire start time")
            for source, time_val, text in time_reports[:5]:
                print(f"    • {source} says: {time_val}")
        else:
            print(f"\n  ✓ Fire start time consistent: {list(times)[0] if times else 'not specified'}")

    # =========================================================================
    # 6. ATTRIBUTION PATTERNS (Who quoted whom)
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. ATTRIBUTION PATTERNS")
    print("=" * 80)

    # Look for quotes and attributions
    attribution_patterns = [
        (r'according to ([^,]+)', 'according_to'),
        (r'([A-Z][a-z]+ [A-Z][a-z]+) said', 'person_said'),
        (r'([A-Z][a-z]+ [A-Z][a-z]+) told', 'person_told'),
        (r'the ([a-z]+ (?:department|ministry|government|office))', 'official_body'),
        (r'"([^"]+)"', 'direct_quote'),
    ]

    attributions = defaultdict(list)
    quotes = []

    for c in fire_claims:
        page = pages.get(c.page_id)
        source = 'Unknown'
        if page:
            for domain, (name, _) in SOURCE_TYPES.items():
                if domain in page.url.lower():
                    source = name
                    break

        text = c.text

        # Find attributions
        for pattern, attr_type in attribution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if attr_type == 'direct_quote' and len(match) > 20:
                    quotes.append((source, match[:80]))
                elif attr_type != 'direct_quote':
                    attributions[match.lower()].append(source)

    print("\nAttributed sources mentioned:")
    for attr, sources in sorted(attributions.items(), key=lambda x: -len(x[1]))[:10]:
        unique_sources = list(set(sources))
        print(f"  \"{attr}\" - mentioned by: {', '.join(unique_sources[:3])}")

    print(f"\nDirect quotes found: {len(quotes)}")
    for source, quote in quotes[:5]:
        print(f"  [{source}] \"{quote}...\"")

    # =========================================================================
    # 7. WHAT WE CAN'T DO (YET)
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. GAPS IN CURRENT DATA")
    print("=" * 80)

    print("""
    ❌ No user-contributed content
       - All sources are mainstream media
       - No "User A reported..." or "User B HERE shared..."
       - Need community contribution layer

    ❌ No witness accounts (direct)
       - We have quotes FROM witnesses via media
       - But no direct witness submissions
       - Need user submission flow

    ❌ Limited official sources
       - Government statements only via media reports
       - No direct gov.hk pages in dataset
       - Need official source ingestion

    ❌ No document/evidence uploads
       - No photos, videos, documents
       - All text-based claims
       - Need media handling

    ✓ CAN demonstrate:
       - Multi-source attribution
       - Temporal ordering
       - Death toll evolution (update chains)
       - Corroboration patterns
       - Basic contradiction detection
       - Quote extraction
    """)

    # =========================================================================
    # 8. SAMPLE EPISTEMIC NARRATIVE
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. SAMPLE EPISTEMIC NARRATIVE (Generated from data)")
    print("=" * 80)

    print("""
    ─────────────────────────────────────────────────────────────────────────
    HONG KONG FIRE - Wang Fuk Court, Tai Po
    ─────────────────────────────────────────────────────────────────────────
    """)

    # Build narrative from actual data
    # Sorted by time
    narrative_items = []

    for time, claim, page in claims_with_time[:20]:
        source = 'Unknown'
        source_type = 'unknown'
        url = ''
        if page:
            url = page.url
            for domain, (name, stype) in SOURCE_TYPES.items():
                if domain in page.url.lower():
                    source = name
                    source_type = stype
                    break

        # Find corroboration count
        text_lower = claim.text.lower()
        corr_count = 0
        for other_time, other_claim, other_page in claims_with_time:
            if other_claim.id != claim.id:
                other_text = other_claim.text.lower()
                # Simple overlap check
                words1 = set(text_lower.split())
                words2 = set(other_text.split())
                overlap = len(words1 & words2) / max(len(words1), 1)
                if overlap > 0.5:
                    corr_count += 1

        narrative_items.append({
            'time': time,
            'source': source,
            'type': source_type,
            'text': claim.text,
            'url': url,
            'corr': min(corr_count, 5)
        })

    # Print narrative
    type_icons = {
        'wire': '📡',
        'international': '📰',
        'local_news': '📰',
        'official': '🏛️',
        'magazine': '📖',
        'tabloid': '📰',
        'unknown': '📄'
    }

    for item in narrative_items[:12]:
        time_str = item['time'].strftime('%H:%M') if hasattr(item['time'], 'strftime') else '??:??'
        icon = type_icons.get(item['type'], '📄')
        corr_str = f"✓×{item['corr']}" if item['corr'] > 0 else ""

        # Truncate text nicely
        text = item['text']
        if len(text) > 70:
            text = text[:67] + "..."

        print(f"    [{time_str}] {icon} {item['source']:12} | {text}")
        if corr_str:
            print(f"             {corr_str} corroborated")
        print()

    print("    ─────────────────────────────────────────────────────────────────")
    print("    📊 DEATH TOLL EVOLUTION")
    counts = sorted(set(r['count'] for r in unique_reports))
    print(f"    {' → '.join(map(str, counts))}")
    print("    ─────────────────────────────────────────────────────────────────")


def main():
    snapshot = load_snapshot()
    extract_epistemic_structure(snapshot.claims, snapshot.pages, snapshot.entities)


if __name__ == "__main__":
    main()
