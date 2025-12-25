"""
Test Belief Kernel with Full Domain Model Claims
=================================================

Uses proper Claim and Entity domain models.
Tests kernel with rich epistemic context.
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, '/app')

from datetime import datetime
from typing import List, Dict, Optional
from openai import AsyncOpenAI

from models.domain.claim import Claim
from models.domain.entity import Entity
from belief_kernel import BeliefKernel


def load_claims_from_cache(path: str = '/tmp/hk_event.json') -> List[Claim]:
    """
    Load claims from cached JSON using proper domain models.
    """
    with open(path) as f:
        data = json.load(f)

    claims = []
    for c in data.get('claims', []):
        # Build Entity domain objects
        entities = []
        for e in c.get('entities', []):
            entities.append(Entity(
                id=e.get('id', ''),
                canonical_name=e.get('canonical_name', ''),
                entity_type=e.get('entity_type', 'UNKNOWN')
            ))

        # Parse event_time
        event_time = None
        if c.get('event_time'):
            try:
                event_time = datetime.fromisoformat(c['event_time'].replace('Z', '+00:00'))
            except:
                pass

        # Build Claim domain object with all available fields
        claim = Claim(
            id=c.get('id', ''),
            page_id=c.get('page_id', ''),
            text=c.get('text', ''),
            event_time=event_time,
            confidence=c.get('confidence', 0.8),
            modality=c.get('modality', 'observation'),
            topic_key=c.get('topic_key'),
            updates_claim_id=c.get('updates_claim_id'),
            is_superseded=c.get('is_superseded', False),
            entities=entities,
            metadata={'source_name': c.get('source_name', 'unknown')}
        )
        claims.append(claim)

    return claims


def analyze_claim_richness(claims: List[Claim]):
    """Analyze what data is actually populated in claims."""
    print(f"\n{'='*60}")
    print("CLAIM DATA RICHNESS ANALYSIS")
    print(f"{'='*60}")
    print(f"Total claims: {len(claims)}")

    # Field coverage
    stats = {
        'event_time': sum(1 for c in claims if c.event_time),
        'confidence_varied': len(set(c.confidence for c in claims)) > 1,
        'modality_varied': len(set(c.modality for c in claims)) > 1,
        'topic_key': sum(1 for c in claims if c.topic_key),
        'updates_claim_id': sum(1 for c in claims if c.updates_claim_id),
        'entities': sum(1 for c in claims if c.entities),
        'is_factual': sum(1 for c in claims if c.is_factual),
    }

    print(f"\nField coverage:")
    print(f"  event_time:       {stats['event_time']}/{len(claims)} ({stats['event_time']/len(claims)*100:.0f}%)")
    print(f"  confidence:       varied={stats['confidence_varied']}")
    print(f"  modality:         varied={stats['modality_varied']}")
    print(f"  topic_key:        {stats['topic_key']}/{len(claims)}")
    print(f"  updates_claim_id: {stats['updates_claim_id']}/{len(claims)}")
    print(f"  entities:         {stats['entities']}/{len(claims)}")
    print(f"  is_factual:       {stats['is_factual']}/{len(claims)}")

    # Modality distribution
    modalities = {}
    for c in claims:
        modalities[c.modality] = modalities.get(c.modality, 0) + 1
    print(f"\nModality distribution:")
    for m, count in sorted(modalities.items(), key=lambda x: -x[1]):
        print(f"  {m}: {count}")

    # Entity stats
    entity_count = sum(len(c.entities) for c in claims if c.entities)
    unique_entities = set()
    entity_types = {}
    for c in claims:
        if c.entities:
            for e in c.entities:
                unique_entities.add(e.id)
                entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1

    print(f"\nEntity coverage:")
    print(f"  Total mentions: {entity_count}")
    print(f"  Unique entities: {len(unique_entities)}")
    for t, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"    {t}: {count}")


def claim_to_rich_text(claim: Claim) -> str:
    """
    Convert Claim to enriched text for kernel input.
    Includes all available epistemic context.
    """
    parts = []

    # Temporal context
    if claim.event_time:
        parts.append(f"[{claim.event_time.strftime('%Y-%m-%d %H:%M')}]")

    # Epistemic context
    parts.append(f"[{claim.modality}]")
    if claim.confidence != 0.8:  # Only if not default
        parts.append(f"[conf:{claim.confidence:.2f}]")

    # Topic grouping
    if claim.topic_key:
        parts.append(f"[topic:{claim.topic_key}]")

    # The claim text
    parts.append(claim.text)

    return " ".join(parts)


def get_claim_source(claim: Claim) -> str:
    """Extract source from claim metadata."""
    return claim.metadata.get('source_name', 'unknown')


async def generate_prose_with_model(kernel: BeliefKernel, llm, model: str = "gpt-4o") -> str:
    """
    Generate prose using a specified model.
    Uses the same prompt as kernel but with better model.
    """
    if not kernel.beliefs:
        return "Awaiting information..."

    # Group beliefs by confidence
    high_conf = [b for b in kernel.beliefs if len(b.sources) >= 3]
    medium_conf = [b for b in kernel.beliefs if len(b.sources) == 2]
    low_conf = [b for b in kernel.beliefs if len(b.sources) == 1]

    beliefs_text = ""
    if high_conf:
        beliefs_text += "CONFIRMED (3+ sources):\n"
        beliefs_text += "\n".join(f"- {b.text}" for b in high_conf[:8])
        beliefs_text += "\n\n"
    if medium_conf:
        beliefs_text += "CORROBORATED (2 sources):\n"
        beliefs_text += "\n".join(f"- {b.text}" for b in medium_conf[:8])
        beliefs_text += "\n\n"
    if low_conf[:5]:
        beliefs_text += "REPORTED (1 source):\n"
        beliefs_text += "\n".join(f"- {b.text}" for b in low_conf[:5])

    if kernel.conflicts:
        beliefs_text += "\n\nUNRESOLVED (conflicting reports):\n"
        for c in kernel.conflicts[:3]:
            beliefs_text += f"- {c['new_claim'][:80]}...\n"

    prompt = f"""Write a comprehensive news article based ONLY on these verified beliefs. Do NOT invent any details.

{beliefs_text}

FORMAT: Markdown with sections
- ## Headline (factual, based on confirmed facts)
- ### Key Facts (bullet points of confirmed information)
- ### Details (prose narrative with proper attribution)
- ### Developing (if there are unresolved conflicts or single-source claims)

RULES:
- ONLY include facts that appear in the beliefs above
- For CONFIRMED (3+ sources): state as established fact
- For CORROBORATED (2 sources): attribute to "multiple sources"
- For REPORTED (1 source): note "according to one report" or similar
- Include specific numbers, times, and names from beliefs
- If there are UNRESOLVED conflicts, explain the discrepancy
- Be thorough - include all significant facts

Return the markdown article."""

    try:
        # gpt-5.2 uses max_completion_tokens instead of max_tokens
        response = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating narrative: {e}"


async def run_kernel_test(claims: List[Claim], limit: int = None, use_rich_text: bool = False):
    """
    Stream claims through kernel and evaluate topology.

    Args:
        claims: List of Claim domain objects
        limit: Optional limit on claims to process
        use_rich_text: If True, include temporal/epistemic context in claim text
    """
    llm = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    kernel = BeliefKernel()

    test_claims = claims[:limit] if limit else claims
    mode = "RICH" if use_rich_text else "PLAIN"
    print(f"\nProcessing {len(test_claims)} claims ({mode} mode)...")
    print("=" * 60)

    # Stream claims
    for i, claim in enumerate(test_claims):
        # Get enriched or plain text
        claim_text = claim_to_rich_text(claim) if use_rich_text else claim.text
        source = get_claim_source(claim)

        result = await kernel.process(
            claim=claim_text,
            source=source,
            llm=llm
        )

        rel = result.get('relation', '?')
        symbol = {
            'COMPATIBLE': '+',
            'REDUNDANT': '=',
            'REFINES': '↑',
            'SUPERSEDES': '→',
            'CONFLICTS': '!'
        }.get(rel, '?')
        print(symbol, end='', flush=True)

        if (i + 1) % 50 == 0:
            s = kernel.summary()
            print(f" [{i+1}] {s['beliefs']}b/{len(s['unresolved_conflicts'])}c")

    print("\n" + "=" * 60)

    # Results
    summary = kernel.summary()
    total = len(test_claims)

    print(f"\nTOPOLOGY RESULTS")
    print(f"=" * 40)
    print(f"Claims processed: {total}")
    print(f"Final beliefs:    {len(summary['current_beliefs'])}")
    print(f"Compression:      {total / max(len(summary['current_beliefs']), 1):.1f}x")
    print(f"Conflicts:        {len(summary['unresolved_conflicts'])}")
    print(f"Coherence:        {kernel.compute_coherence():.2%}")
    print(f"Entropy:          {kernel.compute_entropy():.2%}")

    print(f"\nRelation breakdown:")
    for rel, count in sorted(summary['relations'].items()):
        pct = count / total * 100
        bar = '█' * int(pct / 5)
        print(f"  {rel:11s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Show high-confidence beliefs
    print(f"\nHigh-confidence beliefs (2+ sources):")
    multi_source = [b for b in kernel.beliefs if len(b.sources) >= 2]
    for b in multi_source[:5]:
        print(f"  [{len(b.sources)} src] {b.text[:70]}...")

    # Show conflicts
    if summary['unresolved_conflicts']:
        print(f"\nUnresolved conflicts:")
        for c in summary['unresolved_conflicts'][:3]:
            print(f"  ! {c['new_claim'][:60]}...")
            if c.get('existing_belief'):
                print(f"    vs: {c['existing_belief'][:60]}...")

    # Generate prose narrative (use better model for final output)
    print(f"\n{'='*60}")
    print("GENERATED PROSE (gpt-5.2)")
    print(f"{'='*60}")

    # Override prose generation with better model
    prose = await generate_prose_with_model(kernel, llm, model="gpt-5.2")
    print(prose)

    # Advanced metrics
    print(f"\n{'='*60}")
    print("ADVANCED METRICS")
    print(f"{'='*60}")

    # Source agreement: beliefs with multiple sources
    single_source = sum(1 for b in kernel.beliefs if len(b.sources) == 1)
    multi_source = sum(1 for b in kernel.beliefs if len(b.sources) >= 2)
    high_conf = sum(1 for b in kernel.beliefs if len(b.sources) >= 3)

    print(f"Source agreement:")
    print(f"  Single source: {single_source} ({single_source/len(kernel.beliefs)*100:.0f}%)")
    print(f"  Multi-source:  {multi_source} ({multi_source/len(kernel.beliefs)*100:.0f}%)")
    print(f"  High-conf (3+): {high_conf} ({high_conf/len(kernel.beliefs)*100:.0f}%)")

    # Update chains (supersession depth)
    superseded = [b for b in kernel.beliefs if b.supersedes]
    print(f"\nEvolution tracking:")
    print(f"  Beliefs with history: {len(superseded)}")
    for b in superseded[:3]:
        print(f"    NOW: {b.text[:50]}...")
        print(f"    WAS: {b.supersedes[:50]}...")

    # Epistemic quality score (composite)
    # High quality = high coherence, low entropy, few conflicts, good compression
    quality_score = (
        summary['relations'].get('REDUNDANT', 0) * 2 +  # Corroboration is good
        summary['relations'].get('SUPERSEDES', 0) * 1.5 +  # Updates are good
        summary['relations'].get('REFINES', 0) * 1 -  # Refinements are okay
        summary['relations'].get('CONFLICTS', 0) * 3  # Conflicts are bad
    ) / len(test_claims)

    print(f"\nEpistemic quality score: {quality_score:.2f}")
    print(f"  (Higher = better. Weights: REDUNDANT×2, SUPERSEDES×1.5, REFINES×1, CONFLICTS×-3)")

    return kernel, summary, prose


async def main():
    """Main test runner"""
    print("=" * 60)
    print("BELIEF KERNEL FULL TEST (Domain Models)")
    print("=" * 60)

    # Load claims using proper domain models
    claims = load_claims_from_cache()

    # Analyze data richness
    analyze_claim_richness(claims)

    # Parse args
    limit = None
    use_rich = '--rich' in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            limit = int(arg)

    # Run test
    kernel, summary, prose = await run_kernel_test(claims, limit, use_rich_text=use_rich)

    # Save FULL TOPOLOGY - this is the epistemic state
    # Prose is just a view on this; can regenerate anytime
    topology = {
        'meta': {
            'mode': 'RICH' if use_rich else 'PLAIN',
            'claims_processed': limit or len(claims),
            'timestamp': datetime.now().isoformat()
        },
        'metrics': {
            'coherence': kernel.compute_coherence(),
            'entropy': kernel.compute_entropy(),
            'belief_count': len(kernel.beliefs),
            'conflict_count': len(kernel.conflicts),
            'compression': (limit or len(claims)) / max(len(kernel.beliefs), 1)
        },
        'relations': summary['relations'],
        # FULL belief objects - not just text
        'beliefs': [
            {
                'text': b.text,
                'sources': b.sources,
                'source_count': len(b.sources),
                'supersedes': b.supersedes,
                'confidence': b.confidence,
                'updated_count': b.updated_count
            }
            for b in kernel.beliefs
        ],
        'conflicts': kernel.conflicts,
        # Processing history for debugging/analysis
        'history': kernel.history
    }

    mode_suffix = 'rich' if use_rich else 'plain'
    output_path = f'/app/test_eu/results/topology_{mode_suffix}.json'
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)

    print(f"\nTopology saved to results/topology_{mode_suffix}.json")
    print(f"  → {len(kernel.beliefs)} beliefs, {len(kernel.conflicts)} conflicts")
    print(f"  → Can regenerate prose anytime from this state")


if __name__ == '__main__':
    asyncio.run(main())
