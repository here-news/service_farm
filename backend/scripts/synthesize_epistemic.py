"""
Epistemic Synthesis Pipeline

Takes raw claims and produces genuine epistemic structure:
1. Semantic clustering - group claims about the same facts
2. Narrative synthesis - LLM-generated prose with inline attribution
3. Tension detection - real semantic opposition, not keywords
4. Confidence scoring - based on source diversity and corroboration
5. Gap identification - what questions remain unanswered
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, '/app')

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

# Source classification
SOURCE_TYPES = {
    'reuters.com': ('Reuters', 'wire'),
    'apnews.com': ('AP', 'wire'),
    'afp.com': ('AFP', 'wire'),
    'bbc.com': ('BBC', 'international'),
    'bbc.co.uk': ('BBC', 'international'),
    'theguardian.com': ('Guardian', 'international'),
    'dw.com': ('DW', 'international'),
    'aljazeera.com': ('Al Jazeera', 'international'),
    'cnn.com': ('CNN', 'international'),
    'nytimes.com': ('NY Times', 'international'),
    'washingtonpost.com': ('Washington Post', 'international'),
    'france24.com': ('France24', 'international'),
    'scmp.com': ('SCMP', 'local'),
    'hk01.com': ('HK01', 'local'),
    'rthk.hk': ('RTHK', 'local'),
    'thestandard.com.hk': ('The Standard', 'local'),
    'hongkongfp.com': ('HKFP', 'local'),
    'gov.hk': ('HK Government', 'official'),
    'info.gov.hk': ('HK Government', 'official'),
    'amnesty.org': ('Amnesty', 'ngo'),
    'hrw.org': ('HRW', 'ngo'),
    'rsf.org': ('RSF', 'ngo'),
    'cpj.org': ('CPJ', 'ngo'),
    'newsweek.com': ('Newsweek', 'international'),
    'nypost.com': ('NY Post', 'international'),
    'independent.co.uk': ('Independent', 'international'),
    'abc.net.au': ('ABC Australia', 'international'),
    'channelnewsasia.com': ('CNA', 'international'),
    'straitstimes.com': ('Straits Times', 'international'),
    'christianitytoday.com': ('Christianity Today', 'international'),
    'dailymail.co.uk': ('Daily Mail', 'international'),
}


@dataclass
class Claim:
    id: str
    text: str
    source_name: str
    source_type: str
    source_url: str
    event_time: Optional[str]


@dataclass
class NarrativeSection:
    """A synthesized narrative section with epistemic metadata"""
    id: str
    heading: Optional[str]
    prose: str  # Synthesized narrative with inline [Source] attribution
    claims_used: List[str]  # Claim IDs that contributed
    sources: List[Dict]  # [{name, type, url}]
    confidence: float  # 0-1 based on source diversity
    epistemic_status: str  # 'established', 'reported', 'contested', 'uncertain'


@dataclass
class Tension:
    """A genuine semantic tension between perspectives"""
    id: str
    topic: str
    description: str
    positions: List[Dict]  # [{label, summary, sources, quote}]
    claims_involved: List[str]


@dataclass
class Gap:
    """An identified epistemic gap"""
    id: str
    question: str  # What we don't know
    why_important: str
    suggested_sources: List[str]
    priority: str


@dataclass
class EpistemicNarrative:
    """Complete epistemic structure for an event"""
    event_id: str
    title: str
    generated_at: str

    # Core narrative
    summary: str  # 2-3 sentence overview
    sections: List[NarrativeSection]

    # Epistemic structure
    tensions: List[Tension]
    gaps: List[Gap]

    # Metadata
    total_claims: int
    sources_by_type: Dict[str, List[str]]
    confidence_overall: float


def classify_source(url: str) -> tuple:
    """Classify URL into (name, type)"""
    if not url:
        return ('Unknown', 'other')

    domain = ''
    try:
        domain = url.lower().split('/')[2].replace('www.', '')
    except:
        pass

    for pattern, (name, stype) in SOURCE_TYPES.items():
        if pattern in domain:
            return (name, stype)

    # Fallback
    name = domain.split('.')[0].upper() if domain else 'Unknown'
    return (name, 'other')


async def fetch_event_claims(driver, event_id: str) -> tuple:
    """Fetch event info and claims from Neo4j"""

    # Get event info
    event_query = """
    MATCH (e:Event {id: $event_id})
    RETURN e.canonical_name as name, e.event_type as type,
           e.event_start as start, e.event_end as end
    """

    # Get claims
    claims_query = """
    MATCH (e:Event {id: $event_id})-[:INTAKES]->(c:Claim)
    OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
    RETURN c.id as id, c.text as text, c.event_time as event_time,
           p.url as url
    ORDER BY c.event_time DESC
    """

    async with driver.session() as session:
        # Event info
        result = await session.run(event_query, event_id=event_id)
        event_data = await result.data()
        event_info = event_data[0] if event_data else {'name': 'Unknown Event'}

        # Claims
        result = await session.run(claims_query, event_id=event_id)
        records = await result.data()

        claims = []
        for r in records:
            name, stype = classify_source(r.get('url', ''))
            claims.append(Claim(
                id=r['id'],
                text=r['text'],
                source_name=name,
                source_type=stype,
                source_url=r.get('url', ''),
                event_time=str(r['event_time']) if r.get('event_time') else None
            ))

        return event_info, claims


async def cluster_claims(client: AsyncOpenAI, claims: List[Claim], event_title: str) -> List[Dict]:
    """Use LLM to semantically cluster claims into topic groups"""

    # Prepare claims for LLM
    claims_text = "\n".join([
        f"[{i}] ({c.source_name}): {c.text}"
        for i, c in enumerate(claims[:60])  # Limit for context
    ])

    prompt = f"""Analyze these news claims about "{event_title}" and group them into semantic clusters.

CLAIMS:
{claims_text}

Group these claims into 4-7 thematic clusters. Each cluster should represent a distinct aspect or fact of the story.

Return JSON array:
[
  {{
    "topic": "Brief topic label",
    "aspect": "What aspect of the story this covers",
    "claim_indices": [0, 3, 7],  // Which claims belong here
    "is_contested": true/false,  // Do sources disagree?
    "key_fact": "The core factual assertion if sources agree"
  }}
]

Focus on:
- Grouping claims that discuss the SAME facts together
- Identifying where sources AGREE vs DISAGREE
- Separating distinct aspects of the story (what happened, reactions, background, etc.)

Return ONLY valid JSON array."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        # Handle both direct array and wrapped object
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'clusters' in result:
            return result['clusters']
        else:
            return list(result.values())[0] if result else []
    except:
        return []


async def synthesize_section(
    client: AsyncOpenAI,
    cluster: Dict,
    claims: List[Claim],
    section_num: int
) -> NarrativeSection:
    """Synthesize a narrative section from a cluster of claims"""

    # Get claims for this cluster
    indices = cluster.get('claim_indices', [])
    cluster_claims = [claims[i] for i in indices if i < len(claims)]

    if not cluster_claims:
        return None

    # Prepare claims with sources
    claims_text = "\n".join([
        f"- {c.text} (from: {c.source_name})"
        for c in cluster_claims
    ])

    sources = list(set(c.source_name for c in cluster_claims))
    source_types = list(set(c.source_type for c in cluster_claims))
    source_count = len(sources)
    is_contested = cluster.get('is_contested', False)

    prompt = f"""Synthesize these claims into clean narrative prose.

TOPIC: {cluster.get('topic', 'Unknown')}
IS CONTESTED: {is_contested}
NUMBER OF SOURCES: {source_count}

CLAIMS:
{claims_text}

Write 2-4 sentences of CLEAN, READABLE prose. Use MINIMAL markers - only where essential:

- {{⚡}} ONLY when sources actively contradict each other (not just different emphasis)
- {{❓}} ONLY for a key uncertain claim from single source

DO NOT mark every fact. Most sentences should have NO markers.
NO source names in brackets. NO numbers. Clean prose only.

EXAMPLE:
"The fire killed at least 17 people, making it one of Hong Kong's deadliest blazes in decades. Officials attribute the blaze to electrical failure{{⚡}}, though residents allege building code violations. The cause remains under investigation."

Return ONLY the clean paragraph."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    prose = response.choices[0].message.content.strip()

    # Calculate confidence based on source diversity
    type_count = len(source_types)
    source_count = len(sources)
    confidence = min(1.0, (type_count * 0.3 + source_count * 0.1))

    # Determine epistemic status
    if cluster.get('is_contested'):
        status = 'contested'
    elif type_count >= 3 or source_count >= 4:
        status = 'established'
    elif source_count >= 2:
        status = 'reported'
    else:
        status = 'uncertain'

    return NarrativeSection(
        id=f"section-{section_num}",
        heading=cluster.get('topic'),
        prose=prose,
        claims_used=[c.id for c in cluster_claims],
        sources=[{'name': c.source_name, 'type': c.source_type} for c in cluster_claims],
        confidence=confidence,
        epistemic_status=status
    )


async def identify_tensions(
    client: AsyncOpenAI,
    claims: List[Claim],
    clusters: List[Dict],
    event_title: str
) -> List[Tension]:
    """Identify genuine semantic tensions between sources"""

    # Find contested clusters
    contested = [c for c in clusters if c.get('is_contested')]

    if not contested:
        # Try to find tensions anyway
        claims_sample = "\n".join([
            f"({c.source_name}, {c.source_type}): {c.text}"
            for c in claims[:40]
        ])

        prompt = f"""Analyze these claims about "{event_title}" for tensions or disagreements between sources.

CLAIMS:
{claims_sample}

Identify 1-3 genuine tensions where different sources present conflicting information or framings.

Return JSON:
{{
  "tensions": [
    {{
      "topic": "What the tension is about",
      "description": "Brief description of the disagreement",
      "position_a": {{
        "label": "One perspective",
        "summary": "What this side says",
        "sources": ["Source1", "Source2"]
      }},
      "position_b": {{
        "label": "Other perspective",
        "summary": "What this side says",
        "sources": ["Source3"]
      }}
    }}
  ]
}}

Only include REAL tensions where sources genuinely disagree, not just different emphasis.
Return empty array if no clear tensions exist."""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response.choices[0].message.content)
            tensions_data = result.get('tensions', [])
        except:
            tensions_data = []
    else:
        tensions_data = []
        for cluster in contested[:3]:
            tensions_data.append({
                'topic': cluster.get('topic', 'Unknown'),
                'description': f"Sources disagree on {cluster.get('aspect', 'this aspect')}",
                'position_a': {'label': 'Position A', 'summary': '', 'sources': []},
                'position_b': {'label': 'Position B', 'summary': '', 'sources': []}
            })

    return [
        Tension(
            id=f"tension-{i}",
            topic=t.get('topic', ''),
            description=t.get('description', ''),
            positions=[t.get('position_a', {}), t.get('position_b', {})],
            claims_involved=[]
        )
        for i, t in enumerate(tensions_data)
    ]


async def identify_gaps(
    client: AsyncOpenAI,
    claims: List[Claim],
    sources_by_type: Dict[str, List[str]],
    event_title: str
) -> List[Gap]:
    """Identify epistemic gaps - what we don't know"""

    # Source type gaps
    gaps = []

    if not sources_by_type.get('official'):
        gaps.append(Gap(
            id="gap-official",
            question="What is the official government position or statement?",
            why_important="Official sources provide authoritative information on investigations, policies, and responses",
            suggested_sources=["Government press releases", "Official statements", "Press conferences"],
            priority="high"
        ))

    if not sources_by_type.get('ngo'):
        gaps.append(Gap(
            id="gap-ngo",
            question="What do civil society organizations say?",
            why_important="NGOs provide independent perspectives and human rights context",
            suggested_sources=["Amnesty International", "Human Rights Watch", "Local advocacy groups"],
            priority="medium"
        ))

    # Ask LLM for content gaps
    claims_summary = "\n".join([c.text for c in claims[:30]])

    prompt = f"""Given these claims about "{event_title}", identify 1-2 important UNANSWERED questions.

CLAIMS SUMMARY:
{claims_summary}

What important aspects of this story are NOT covered by these claims?

Return JSON:
{{
  "gaps": [
    {{
      "question": "What we don't know",
      "why_important": "Why this matters",
      "suggested_sources": ["Where to look"]
    }}
  ]
}}

Focus on genuinely missing information, not just minor details."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        for i, g in enumerate(result.get('gaps', [])[:2]):
            gaps.append(Gap(
                id=f"gap-content-{i}",
                question=g.get('question', ''),
                why_important=g.get('why_important', ''),
                suggested_sources=g.get('suggested_sources', []),
                priority="medium"
            ))
    except:
        pass

    return gaps


async def generate_summary(
    client: AsyncOpenAI,
    sections: List[NarrativeSection],
    event_title: str
) -> str:
    """Generate a 2-3 sentence summary of the event"""

    section_texts = "\n".join([s.prose for s in sections if s])

    prompt = f"""Summarize this news event in 2-3 clean sentences.

EVENT: {event_title}

NARRATIVE:
{section_texts}

Write a concise, readable summary. Use {{⚡}} ONLY if there's one major contested point.
Otherwise NO markers - just clean prose.
Return ONLY the summary."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


async def synthesize_event(driver, client: AsyncOpenAI, event_id: str) -> EpistemicNarrative:
    """Full epistemic synthesis for an event"""

    print(f"Synthesizing event: {event_id}")

    # Fetch data
    event_info, claims = await fetch_event_claims(driver, event_id)
    print(f"  Found {len(claims)} claims")

    if len(claims) < 5:
        print(f"  Too few claims, skipping")
        return None

    event_title = event_info.get('name', 'Unknown Event')

    # Classify sources
    sources_by_type = defaultdict(list)
    for c in claims:
        if c.source_name not in sources_by_type[c.source_type]:
            sources_by_type[c.source_type].append(c.source_name)

    # Step 1: Cluster claims semantically
    print("  Clustering claims...")
    clusters = await cluster_claims(client, claims, event_title)
    print(f"  Found {len(clusters)} clusters")

    # Step 2: Synthesize narrative sections
    print("  Synthesizing narrative...")
    sections = []
    for i, cluster in enumerate(clusters):
        section = await synthesize_section(client, cluster, claims, i)
        if section:
            sections.append(section)
    print(f"  Generated {len(sections)} sections")

    # Step 3: Identify tensions
    print("  Identifying tensions...")
    tensions = await identify_tensions(client, claims, clusters, event_title)
    print(f"  Found {len(tensions)} tensions")

    # Step 4: Identify gaps
    print("  Identifying gaps...")
    gaps = await identify_gaps(client, claims, dict(sources_by_type), event_title)
    print(f"  Found {len(gaps)} gaps")

    # Step 5: Generate summary
    print("  Generating summary...")
    summary = await generate_summary(client, sections, event_title)

    # Calculate overall confidence
    if sections:
        confidence = sum(s.confidence for s in sections) / len(sections)
    else:
        confidence = 0.0

    return EpistemicNarrative(
        event_id=event_id,
        title=event_title,
        generated_at=datetime.utcnow().isoformat(),
        summary=summary,
        sections=[asdict(s) for s in sections],
        tensions=[asdict(t) for t in tensions],
        gaps=[asdict(g) for g in gaps],
        total_claims=len(claims),
        sources_by_type=dict(sources_by_type),
        confidence_overall=confidence
    )


async def main():
    # Connect to Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', '')

    print(f"Connecting to Neo4j at {neo4j_uri}")
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # OpenAI client
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        # Fetch top events
        query = """
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[:INTAKES]->(c:Claim)
        WITH e, count(c) as claim_count
        WHERE claim_count > 10
        RETURN e.id as id, e.canonical_name as name, claim_count
        ORDER BY claim_count DESC
        LIMIT 3
        """

        async with driver.session() as session:
            result = await session.run(query)
            events = await result.data()

        print(f"Found {len(events)} events to process")

        # Process each event
        narratives = []
        for event in events:
            narrative = await synthesize_event(driver, client, event['id'])
            if narrative:
                narratives.append(asdict(narrative))

        # Write output
        output_dir = '/app/prototypes/data'
        os.makedirs(output_dir, exist_ok=True)

        for narrative in narratives:
            filename = f"{output_dir}/{narrative['event_id']}_epistemic.json"
            with open(filename, 'w') as f:
                json.dump(narrative, f, indent=2, default=str)
            print(f"Wrote {filename}")

        # Write index
        index = {
            'generated_at': datetime.utcnow().isoformat(),
            'type': 'epistemic_narratives',
            'events': [
                {
                    'id': n['event_id'],
                    'title': n['title'],
                    'total_claims': n['total_claims'],
                    'sections': len(n['sections']),
                    'confidence': n['confidence_overall']
                }
                for n in narratives
            ]
        }
        with open(f"{output_dir}/index_epistemic.json", 'w') as f:
            json.dump(index, f, indent=2)
        print(f"Wrote {output_dir}/index_epistemic.json")

    finally:
        await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
