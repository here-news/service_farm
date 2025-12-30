"""
Generate Epistemic Data for Prototype UI

Processes existing claims from Neo4j to produce structured epistemic data:
- Voice/source classification
- Corroborations (similar claims from different sources)
- Tensions (competing framings)
- Gaps (missing source types)
- Attributed quotes

Outputs JSON files for the prototype UI.
"""
import os
import sys
import json
import re
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, '/app')

from neo4j import AsyncGraphDatabase
from rapidfuzz import fuzz

# Source classification
SOURCE_TYPES = {
    # Wire services
    'reuters.com': ('Reuters', 'wire'),
    'apnews.com': ('AP', 'wire'),
    'afp.com': ('AFP', 'wire'),
    # International
    'bbc.com': ('BBC', 'international'),
    'bbc.co.uk': ('BBC', 'international'),
    'theguardian.com': ('Guardian', 'international'),
    'dw.com': ('DW', 'international'),
    'aljazeera.com': ('Al Jazeera', 'international'),
    'cnn.com': ('CNN', 'international'),
    'nytimes.com': ('NY Times', 'international'),
    'washingtonpost.com': ('Washington Post', 'international'),
    'france24.com': ('France24', 'international'),
    # Local
    'scmp.com': ('SCMP', 'local'),
    'hk01.com': ('HK01', 'local'),
    'rthk.hk': ('RTHK', 'local'),
    'thestandard.com.hk': ('The Standard', 'local'),
    # Official
    'gov.hk': ('HK Government', 'official'),
    'info.gov.hk': ('HK Government', 'official'),
    'china-embassy': ('China Embassy', 'official'),
    # NGO
    'amnesty.org': ('Amnesty', 'ngo'),
    'hrw.org': ('HRW', 'ngo'),
    'rsf.org': ('RSF', 'ngo'),
    'cpj.org': ('CPJ', 'ngo'),
    'hongkongfp.com': ('HKFP', 'local'),
    # Other
    'newsweek.com': ('Newsweek', 'international'),
    'nypost.com': ('NY Post', 'international'),
    'independent.co.uk': ('Independent', 'international'),
    'telegraph.co.uk': ('Telegraph', 'international'),
    'abc.net.au': ('ABC Australia', 'international'),
    'channelnewsasia.com': ('CNA', 'international'),
    'straitstimes.com': ('Straits Times', 'international'),
}

# Epistemic pole signals for tension detection
TENSION_POLES = {
    'jimmy_lai': {
        'prosecution': {
            'signals': ['collu', 'sedition', 'national security', 'foreign forces', 'endanger', 'guilty'],
            'sources': ['official', 'china']
        },
        'defense': {
            'signals': ['press freedom', 'journalism', 'peaceful', 'political', 'persecution', 'innocent', 'prisoner of conscience'],
            'sources': ['ngo', 'defense']
        }
    },
    'hk_fire': {
        'official': {
            'signals': ['investigation', 'cause', 'safety', 'regulation'],
            'sources': ['official']
        },
        'victim': {
            'signals': ['victim', 'family', 'compensation', 'negligence'],
            'sources': ['local']
        }
    }
}


@dataclass
class Source:
    name: str
    type: str
    url: str
    domain: str


@dataclass
class Claim:
    id: str
    text: str
    source: Source
    event_time: Optional[str]
    page_id: str


@dataclass
class Corroboration:
    claim_text: str
    sources: List[str]
    count: int
    first_reported: str
    claim_ids: List[str]


@dataclass
class Tension:
    topic: str
    pole_a: Dict[str, Any]  # {label, quote, sources}
    pole_b: Dict[str, Any]
    claim_ids: List[str]


@dataclass
class Gap:
    type: str
    description: str
    priority: str
    bounty: int


@dataclass
class NarrativeBlock:
    type: str  # 'paragraph', 'tension', 'gap', 'quote'
    content: str
    voice: Optional[str]
    sources: List[str]
    source_type: Optional[str]
    timestamp: Optional[str]
    metadata: Optional[Dict]


@dataclass
class EpistemicEvent:
    id: str
    title: str
    event_type: str
    date_range: str
    claim_count: int
    source_count: int
    sources_by_type: Dict[str, List[str]]
    narrative_blocks: List[NarrativeBlock]
    corroborations: List[Corroboration]
    tensions: List[Tension]
    gaps: List[Gap]
    live_activity: List[Dict]
    completeness: float


def classify_source(url: str) -> Source:
    """Classify a URL into source name and type"""
    domain = ''
    try:
        domain = url.lower().split('/')[2].replace('www.', '')
    except:
        pass

    for pattern, (name, stype) in SOURCE_TYPES.items():
        if pattern in domain:
            return Source(name=name, type=stype, url=url, domain=domain)

    # Fallback
    name = domain.split('.')[0].upper() if domain else 'Unknown'
    return Source(name=name, type='other', url=url, domain=domain)


def find_corroborations(claims: List[Claim], threshold: float = 70.0) -> List[Corroboration]:
    """Find claims that say similar things from different sources"""
    corroborations = []
    used_claims = set()

    for i, claim_a in enumerate(claims):
        if claim_a.id in used_claims:
            continue

        similar_claims = [claim_a]

        for j, claim_b in enumerate(claims):
            if i >= j or claim_b.id in used_claims:
                continue
            if claim_a.source.name == claim_b.source.name:
                continue

            # Check similarity
            similarity = fuzz.token_sort_ratio(claim_a.text, claim_b.text)
            if similarity >= threshold:
                similar_claims.append(claim_b)
                used_claims.add(claim_b.id)

        if len(similar_claims) >= 2:
            used_claims.add(claim_a.id)
            sources = list(set(c.source.name for c in similar_claims))
            times = [c.event_time for c in similar_claims if c.event_time]
            first = min(times) if times else None

            corroborations.append(Corroboration(
                claim_text=claim_a.text[:200],
                sources=sources,
                count=len(sources),
                first_reported=first or 'Unknown',
                claim_ids=[c.id for c in similar_claims]
            ))

    # Sort by corroboration count
    corroborations.sort(key=lambda x: -x.count)
    return corroborations[:20]  # Top 20


def find_tensions(claims: List[Claim], event_type: str) -> List[Tension]:
    """Find competing framings in claims"""
    tensions = []
    poles_config = TENSION_POLES.get(event_type, {})

    if not poles_config:
        return tensions

    pole_names = list(poles_config.keys())
    if len(pole_names) < 2:
        return tensions

    pole_a_name, pole_b_name = pole_names[0], pole_names[1]
    pole_a_config = poles_config[pole_a_name]
    pole_b_config = poles_config[pole_b_name]

    pole_a_claims = []
    pole_b_claims = []

    for claim in claims:
        text_lower = claim.text.lower()

        # Check pole A signals
        a_score = sum(1 for sig in pole_a_config['signals'] if sig in text_lower)
        b_score = sum(1 for sig in pole_b_config['signals'] if sig in text_lower)

        if a_score > b_score and a_score > 0:
            pole_a_claims.append(claim)
        elif b_score > a_score and b_score > 0:
            pole_b_claims.append(claim)

    if pole_a_claims and pole_b_claims:
        # Find best representative from each pole
        best_a = max(pole_a_claims, key=lambda c: len(c.text))
        best_b = max(pole_b_claims, key=lambda c: len(c.text))

        tensions.append(Tension(
            topic="Framing of events",
            pole_a={
                'label': pole_a_name.replace('_', ' ').title(),
                'quote': best_a.text[:300],
                'sources': list(set(c.source.name for c in pole_a_claims))[:5]
            },
            pole_b={
                'label': pole_b_name.replace('_', ' ').title(),
                'quote': best_b.text[:300],
                'sources': list(set(c.source.name for c in pole_b_claims))[:5]
            },
            claim_ids=[best_a.id, best_b.id]
        ))

    return tensions


def find_gaps(sources_by_type: Dict[str, List[str]]) -> List[Gap]:
    """Identify missing source types"""
    gaps = []

    if not sources_by_type.get('official'):
        gaps.append(Gap(
            type='missing_source',
            description='No official government statements found',
            priority='high',
            bounty=15
        ))

    if not sources_by_type.get('ngo'):
        gaps.append(Gap(
            type='perspective_gap',
            description='No NGO or human rights organization perspective',
            priority='medium',
            bounty=20
        ))

    if not sources_by_type.get('wire'):
        gaps.append(Gap(
            type='missing_source',
            description='No wire service coverage (Reuters, AP, AFP)',
            priority='medium',
            bounty=10
        ))

    if len(sources_by_type.get('local', [])) < 2:
        gaps.append(Gap(
            type='missing_source',
            description='Limited local news coverage',
            priority='low',
            bounty=8
        ))

    return gaps


def extract_quotes(claims: List[Claim]) -> List[Dict]:
    """Extract direct quotes from claims"""
    quotes = []
    quote_pattern = r'"([^"]{20,})"'

    for claim in claims:
        matches = re.findall(quote_pattern, claim.text)
        for match in matches:
            quotes.append({
                'text': match,
                'source': claim.source.name,
                'source_type': claim.source.type,
                'claim_id': claim.id
            })

    return quotes[:10]


def build_narrative_blocks(claims: List[Claim], corroborations: List[Corroboration],
                           tensions: List[Tension], quotes: List[Dict]) -> List[NarrativeBlock]:
    """Build narrative blocks from claims, organized by topic"""
    blocks = []

    # Group claims by rough topic (using simple heuristics)
    used_claims = set()

    # First, add high-corroboration facts
    for corr in corroborations[:5]:
        if corr.count >= 3:
            blocks.append(NarrativeBlock(
                type='paragraph',
                content=corr.claim_text,
                voice='wire' if 'Reuters' in corr.sources or 'AP' in corr.sources else 'international',
                sources=corr.sources[:4],
                source_type='corroborated',
                timestamp=corr.first_reported,
                metadata={'corroboration_count': corr.count}
            ))
            used_claims.update(corr.claim_ids)

    # Add tensions
    for tension in tensions:
        blocks.append(NarrativeBlock(
            type='tension',
            content=tension.topic,
            voice=None,
            sources=[],
            source_type=None,
            timestamp=None,
            metadata={
                'pole_a': tension.pole_a,
                'pole_b': tension.pole_b
            }
        ))
        used_claims.update(tension.claim_ids)

    # Add some uncorroborated but unique claims
    for claim in claims:
        if claim.id in used_claims:
            continue
        if len(claim.text) > 100:  # Substantive claims
            blocks.append(NarrativeBlock(
                type='paragraph',
                content=claim.text[:300],
                voice=claim.source.type,
                sources=[claim.source.name],
                source_type='single',
                timestamp=claim.event_time,
                metadata=None
            ))
            used_claims.add(claim.id)
            if len(blocks) >= 12:
                break

    return blocks


async def fetch_event_claims(driver, event_id: str) -> List[Claim]:
    """Fetch all claims for an event from Neo4j"""
    claims = []

    query = """
    MATCH (e:Event {id: $event_id})-[:INTAKES]->(c:Claim)
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    RETURN c.id as id, c.text as text, c.event_time as event_time,
           p.id as page_id, p.url as url
    ORDER BY c.event_time DESC
    """

    async with driver.session() as session:
        result = await session.run(query, event_id=event_id)
        records = await result.data()

        for r in records:
            source = classify_source(r.get('url', ''))
            claims.append(Claim(
                id=r['id'],
                text=r['text'],
                source=source,
                event_time=str(r['event_time']) if r.get('event_time') else None,
                page_id=r.get('page_id', '')
            ))

    return claims


async def fetch_events(driver) -> List[Dict]:
    """Fetch all events"""
    query = """
    MATCH (e:Event)
    OPTIONAL MATCH (e)-[:INTAKES]->(c:Claim)
    WITH e, count(c) as claim_count
    WHERE claim_count > 5
    RETURN e.id as id, e.canonical_name as name, e.event_type as type,
           e.event_start as start, e.event_end as end, claim_count
    ORDER BY claim_count DESC
    LIMIT 5
    """

    async with driver.session() as session:
        result = await session.run(query)
        return await result.data()


async def generate_epistemic_event(driver, event_info: Dict) -> EpistemicEvent:
    """Generate full epistemic data for an event"""
    event_id = event_info['id']
    print(f"Processing event: {event_info['name']}")

    # Fetch claims
    claims = await fetch_event_claims(driver, event_id)
    print(f"  Found {len(claims)} claims")

    # Classify sources
    sources_by_type = defaultdict(list)
    for claim in claims:
        if claim.source.name not in sources_by_type[claim.source.type]:
            sources_by_type[claim.source.type].append(claim.source.name)

    # Find corroborations
    corroborations = find_corroborations(claims)
    print(f"  Found {len(corroborations)} corroborations")

    # Determine event type for tension detection
    event_type = 'jimmy_lai' if 'lai' in event_info['name'].lower() else \
                 'hk_fire' if 'fire' in event_info['name'].lower() else 'generic'

    # Find tensions
    tensions = find_tensions(claims, event_type)
    print(f"  Found {len(tensions)} tensions")

    # Find gaps
    gaps = find_gaps(dict(sources_by_type))
    print(f"  Found {len(gaps)} gaps")

    # Extract quotes
    quotes = extract_quotes(claims)

    # Build narrative
    narrative_blocks = build_narrative_blocks(claims, corroborations, tensions, quotes)

    # Calculate completeness
    expected_types = ['wire', 'international', 'local', 'official', 'ngo']
    covered = sum(1 for t in expected_types if sources_by_type.get(t))
    completeness = covered / len(expected_types)

    # Generate fake live activity
    live_activity = [
        {'type': 'source_added', 'text': f'New source: {claims[0].source.name}', 'time': '5 min ago'},
        {'type': 'corroboration', 'text': f'Claim corroborated by {corroborations[0].count} sources' if corroborations else 'Processing...', 'time': '12 min ago'},
    ]

    return EpistemicEvent(
        id=event_id,
        title=event_info['name'],
        event_type=event_info.get('type', 'event'),
        date_range=f"{event_info.get('start', 'Unknown')} - {event_info.get('end', 'Ongoing')}",
        claim_count=len(claims),
        source_count=sum(len(v) for v in sources_by_type.values()),
        sources_by_type=dict(sources_by_type),
        narrative_blocks=[asdict(b) for b in narrative_blocks],
        corroborations=[asdict(c) for c in corroborations],
        tensions=[asdict(t) for t in tensions],
        gaps=[asdict(g) for g in gaps],
        live_activity=live_activity,
        completeness=completeness
    )


async def main():
    # Connect to Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', '')

    print(f"Connecting to Neo4j at {neo4j_uri}")
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        # Fetch events
        events = await fetch_events(driver)
        print(f"Found {len(events)} events with >10 claims")

        # Generate epistemic data for each
        epistemic_events = []
        for event_info in events[:3]:  # Top 3 events
            epistemic_event = await generate_epistemic_event(driver, event_info)
            epistemic_events.append(asdict(epistemic_event))

        # Write output
        output_dir = '/app/prototypes/data'
        os.makedirs(output_dir, exist_ok=True)

        # Write individual event files
        for event in epistemic_events:
            filename = f"{output_dir}/{event['id']}.json"
            with open(filename, 'w') as f:
                json.dump(event, f, indent=2, default=str)
            print(f"Wrote {filename}")

        # Write index
        index = {
            'generated_at': datetime.utcnow().isoformat(),
            'events': [
                {'id': e['id'], 'title': e['title'], 'claim_count': e['claim_count']}
                for e in epistemic_events
            ]
        }
        with open(f"{output_dir}/index.json", 'w') as f:
            json.dump(index, f, indent=2)
        print(f"Wrote {output_dir}/index.json")

    finally:
        await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
