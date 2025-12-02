"""
Test Section-Aware Event Growth

Process Hong Kong fire pages to demonstrate section detection and promotion scoring
"""
import asyncio
import asyncpg
import os
import json
from openai import AsyncOpenAI
from workers.section_aware_enrichment import SectionAwareEnricher
from workers.holistic_enrichment import FactBelief
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def test_section_growth():
    """Test section-aware enrichment with Hong Kong fire pages"""

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enricher = SectionAwareEnricher(client)

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    async with db_pool.acquire() as conn:
        # Get Hong Kong fire pages in chronological order
        pages_raw = await conn.fetch("""
            SELECT
                id, title, description, content_text, url,
                pub_time, created_at
            FROM core.pages
            WHERE title ILIKE '%hong kong%fire%' OR title ILIKE '%tai po%'
            ORDER BY created_at
        """)

        pages = [dict(p) for p in pages_raw]

    logger.info(f"\n{'='*80}")
    logger.info(f"SECTION-AWARE EVENT GROWTH TEST")
    logger.info(f"Processing {len(pages)} pages chronologically")
    logger.info(f"{'='*80}\n")

    # Initialize event state
    event_state = {
        'event_id': 'test-event-hk-fire',
        'sections': {
            'main': {
                'name': 'Main Event',
                'semantic_type': 'primary',
                'ontology': {},
                'page_ids': [],
                'page_count': 0,
                'promotion_score': 0.0,
                'created_at': pages[0]['created_at'].isoformat() if pages else None,
                'updated_at': None,
                'enrichment_timeline': []
            }
        },
        'artifact_count': 0,
        'enrichment_timeline': []
    }

    # Process each page
    for i, page in enumerate(pages, 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"[{i}/{len(pages)}] Processing: {page['title'][:60]}...")
        logger.info(f"   URL: {page['url']}")
        logger.info(f"   Time: {page['created_at']}")
        logger.info(f"{'─'*80}")

        try:
            result = await enricher.enrich_event_with_page(event_state, page)

            logger.info(f"\n   ✅ Enrichment complete")
            logger.info(f"      Section: {result['section_name']} ({result['section_key']})")

            if 'promotion_score' in result and result['promotion_score']['total'] > 0:
                score = result['promotion_score']
                logger.info(f"      Promotion: {score['total']:.3f}")
                logger.info(f"         Temporal: {score['signals']['temporal_gap']:.2f}")
                logger.info(f"         Entity: {score['signals']['entity_divergence']:.2f}")
                logger.info(f"         Semantic: {score['signals']['semantic_shift']:.2f}")
                logger.info(f"         Density: {score['signals']['page_density']:.2f}")

        except Exception as e:
            logger.error(f"   ❌ Error: {e}", exc_info=True)

        # Show event state after each page
        logger.info(f"\n   📊 Event State:")
        logger.info(f"      Total pages: {event_state['artifact_count']}")
        logger.info(f"      Sections: {len(event_state['sections'])}")

        for key, section in event_state['sections'].items():
            logger.info(f"         {key}: {section['name']} ({section['page_count']} pages)")
            if key != 'main' and section.get('promotion_score'):
                ps_data = section['promotion_score']
                ps = ps_data if isinstance(ps_data, (int, float)) else ps_data.get('total', 0.0)
                status = "🔴 PROMOTABLE" if ps >= 0.7 else "🟡 REVIEW" if ps >= 0.5 else "🟢 STABLE"
                logger.info(f"            Score: {ps:.3f} {status}")

    # Final summary
    logger.info(f"\n\n{'='*80}")
    logger.info(f"FINAL EVENT STATE")
    logger.info(f"{'='*80}\n")

    logger.info(f"Total artifacts processed: {event_state['artifact_count']}")
    logger.info(f"Total sections: {len(event_state['sections'])}")

    # Show each section in detail
    for key, section in event_state['sections'].items():
        logger.info(f"\n{'─'*80}")
        logger.info(f"Section: {section['name']} ({key})")
        logger.info(f"{'─'*80}")
        logger.info(f"Type: {section['semantic_type']}")
        logger.info(f"Pages: {section['page_count']}")

        if key != 'main':
            ps_data = section.get('promotion_score', 0.0)
            ps = ps_data if isinstance(ps_data, (int, float)) else ps_data.get('total', 0.0)
            signals_data = section.get('promotion_signals', {})
            signals = signals_data if isinstance(signals_data, dict) else ps_data.get('signals', {}) if isinstance(ps_data, dict) else {}
            logger.info(f"\nPromotion Score: {ps:.3f}")
            logger.info(f"  Temporal gap: {signals.get('temporal_gap', 0):.2f}")
            logger.info(f"  Entity divergence: {signals.get('entity_divergence', 0):.2f}")
            logger.info(f"  Semantic shift: {signals.get('semantic_shift', 0):.2f}")
            logger.info(f"  Page density: {signals.get('page_density', 0):.2f}")
            logger.info(f"  Human weight: {signals.get('human_weight', 0):.2f}")

            if ps >= 0.7:
                logger.warning(f"  ⚠️  PROMOTABLE - Should become separate event!")
            elif ps >= 0.5:
                logger.info(f"  ⚡ REVIEW - Consider for promotion")
            else:
                logger.info(f"  ✅ STABLE - Stays within event")

        # Show ontology summary
        ontology = section.get('ontology', {})
        if ontology:
            logger.info(f"\nOntology:")

            # Casualties
            if 'casualties' in ontology:
                logger.info(f"  Casualties:")
                for subtype, belief in ontology['casualties'].items():
                    if isinstance(belief, FactBelief):
                        logger.info(f"    {subtype}: {belief.current_value} (confidence: {belief.current_confidence:.2f}, {len(belief.timeline)} sources)")

            # Timeline
            if 'timeline' in ontology:
                logger.info(f"  Timeline:")
                timeline = ontology['timeline']
                if 'start' in timeline:
                    start = timeline['start']
                    if isinstance(start, FactBelief):
                        logger.info(f"    Start: {start.current_value}")

                if 'milestones' in timeline and timeline['milestones']:
                    logger.info(f"    Milestones: {len(timeline['milestones'])}")
                    for m in timeline['milestones'][:3]:  # Show first 3
                        logger.info(f"      • {m.get('event', 'N/A')} at {m.get('time', 'N/A')}")

            # Story
            if 'story' in ontology:
                story = ontology['story']
                if 'description' in story:
                    logger.info(f"  Story:")
                    logger.info(f"    {story['description'][:200]}...")

    # Export to JSON for inspection
    output_file = '/tmp/section_growth_result.json'

    # Serialize for JSON export
    def serialize_event_state(state):
        """Convert FactBeliefs to dicts"""
        from datetime import datetime

        def serialize_obj(obj):
            if isinstance(obj, FactBelief):
                return obj.to_dict()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_obj(item) for item in obj]
            else:
                return obj

        return serialize_obj(state)

    serialized_state = serialize_event_state(event_state)

    with open(output_file, 'w') as f:
        json.dump(serialized_state, f, indent=2, default=str)

    logger.info(f"\n{'='*80}")
    logger.info(f"Event state exported to: {output_file}")
    logger.info(f"{'='*80}\n")

    await db_pool.close()


if __name__ == '__main__':
    asyncio.run(test_section_growth())
