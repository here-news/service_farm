#!/usr/bin/env python3
"""
Embedding-based Event Matching Experiment

Tests embedding similarity for event matching with current demo data:
- Generates embeddings for all pages and events
- Tests different similarity thresholds
- Compares quality vs current entity-based matching
- Saves similarity scores for analysis
"""

import asyncio
import asyncpg
import json
import os
from openai import AsyncOpenAI
from datetime import datetime

openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Database connection
DB_CONFIG = {
    'host': 'demo-postgres',
    'port': 5432,
    'user': 'demo_user',
    'password': 'demo_pass',
    'database': 'demo_phi_here'
}


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI text-embedding-3-small"""
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",  # Cheaper, smaller: 1536 dims, $0.00002/1K tokens
        input=text[:8000]  # Limit to ~8K chars to stay under token limit
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0.0


async def get_pages_and_events(conn):
    """Fetch all pages and events from database"""
    pages = await conn.fetch("""
        SELECT id, url, title, content_text, status
        FROM pages
        WHERE status IN ('complete', 'entities_extracted')
        ORDER BY created_at DESC
    """)

    events = await conn.fetch("""
        SELECT e.id, e.title, e.summary, e.event_type,
               COUNT(DISTINCT pe.page_id) as article_count,
               array_agg(DISTINCT en.canonical_name) as entities
        FROM events e
        LEFT JOIN page_events pe ON e.id = pe.event_id
        LEFT JOIN event_entities ee ON e.id = ee.event_id
        LEFT JOIN entities en ON ee.entity_id = en.id
        GROUP BY e.id
        ORDER BY e.created_at DESC
    """)

    return pages, events


async def embed_page(page) -> dict:
    """Create embedding for a page"""
    # Use title + first 1000 words of content
    text = page['title'] or ""
    if page['content_text']:
        words = page['content_text'].split()[:1000]
        text += " " + " ".join(words)

    embedding = await generate_embedding(text)
    return {
        'id': str(page['id']),
        'url': page['url'],
        'title': page['title'],
        'embedding': embedding,
        'text_preview': text[:200]
    }


async def embed_event(event) -> dict:
    """Create embedding for an event"""
    # Use title + summary + key entities
    text = event['title'] or ""
    if event['summary']:
        text += " " + event['summary']
    if event['entities']:
        entities_str = ", ".join([e for e in event['entities'] if e])
        text += " Entities: " + entities_str

    embedding = await generate_embedding(text)
    return {
        'id': str(event['id']),
        'title': event['title'],
        'article_count': event['article_count'],
        'embedding': embedding,
        'text_preview': text[:200]
    }


async def test_matching(page_embeddings, event_embeddings, conn):
    """Test different matching strategies and thresholds"""

    # Get ground truth (current page-event links)
    ground_truth = {}
    current_links = await conn.fetch("""
        SELECT pe.page_id, pe.event_id, e.title as event_title
        FROM page_events pe
        JOIN events e ON pe.event_id = e.id
    """)
    for link in current_links:
        ground_truth[str(link['page_id'])] = {
            'event_id': str(link['event_id']),
            'event_title': link['event_title']
        }

    # Test different thresholds
    thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]
    results = {}

    for threshold in thresholds:
        matches = []
        correct = 0
        incorrect = 0
        missed = 0

        for page_emb in page_embeddings:
            page_id = page_emb['id']
            best_match = None
            best_similarity = 0.0

            # Find most similar event
            for event_emb in event_embeddings:
                similarity = cosine_similarity(page_emb['embedding'], event_emb['embedding'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = event_emb

            # Record match if above threshold
            if best_similarity >= threshold and best_match:
                matches.append({
                    'page_id': page_id,
                    'page_title': page_emb['title'],
                    'matched_event_id': best_match['id'],
                    'matched_event_title': best_match['title'],
                    'similarity': best_similarity
                })

                # Check against ground truth
                if page_id in ground_truth:
                    if ground_truth[page_id]['event_id'] == best_match['id']:
                        correct += 1
                    else:
                        incorrect += 1
                        print(f"❌ Mismatch at {threshold}: {page_emb['title'][:40]}")
                        print(f"   Predicted: {best_match['title'][:40]} ({best_similarity:.3f})")
                        print(f"   Actual: {ground_truth[page_id]['event_title'][:40]}")
            else:
                # Didn't match to anything
                if page_id in ground_truth:
                    missed += 1

        results[threshold] = {
            'correct': correct,
            'incorrect': incorrect,
            'missed': missed,
            'precision': correct / (correct + incorrect) if (correct + incorrect) > 0 else 0,
            'recall': correct / (correct + missed) if (correct + missed) > 0 else 0,
            'matches': matches
        }

    return results, ground_truth


async def analyze_failed_cases(page_embeddings, event_embeddings, ground_truth):
    """Analyze cases where current entity matching worked but embedding might not"""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: Current vs Embedding Matching")
    print("="*80)

    for page_emb in page_embeddings:
        page_id = page_emb['id']

        # Calculate similarities to all events
        similarities = []
        for event_emb in event_embeddings:
            sim = cosine_similarity(page_emb['embedding'], event_emb['embedding'])
            similarities.append({
                'event_id': event_emb['id'],
                'event_title': event_emb['title'],
                'similarity': sim
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Show top 3 matches for each page
        print(f"\n📄 Page: {page_emb['title'][:60]}")
        print(f"   Current match: {ground_truth.get(page_id, {}).get('event_title', 'NONE')}")
        print(f"   Top embedding matches:")
        for i, sim in enumerate(similarities[:3], 1):
            indicator = "✅" if ground_truth.get(page_id, {}).get('event_id') == sim['event_id'] else "  "
            print(f"   {indicator} {i}. {sim['event_title'][:50]} (sim: {sim['similarity']:.3f})")


async def main():
    print("🧪 Embedding-based Event Matching Experiment")
    print("=" * 80)

    # Connect to database
    conn = await asyncpg.connect(**DB_CONFIG)

    try:
        # Step 1: Fetch data
        print("\n📊 Fetching pages and events...")
        pages, events = await get_pages_and_events(conn)
        print(f"   Found {len(pages)} pages, {len(events)} events")

        # Step 2: Generate embeddings
        print("\n🔄 Generating embeddings...")
        print(f"   Embedding {len(pages)} pages...")
        page_embeddings = []
        for i, page in enumerate(pages):
            page_emb = await embed_page(page)
            page_embeddings.append(page_emb)
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i + 1}/{len(pages)}")

        print(f"\n   Embedding {len(events)} events...")
        event_embeddings = []
        for i, event in enumerate(events):
            event_emb = await embed_event(event)
            event_embeddings.append(event_emb)
            print(f"   {i + 1}. {event['title'][:50]}")

        # Step 3: Test matching with different thresholds
        print("\n🎯 Testing matching with different thresholds...")
        results, ground_truth = await test_matching(page_embeddings, event_embeddings, conn)

        # Step 4: Display results
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"{'Threshold':<12} {'Correct':<10} {'Incorrect':<12} {'Missed':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 80)
        for threshold, res in sorted(results.items()):
            print(f"{threshold:<12.2f} {res['correct']:<10} {res['incorrect']:<12} {res['missed']:<10} "
                  f"{res['precision']:<12.2%} {res['recall']:<10.2%}")

        # Step 5: Detailed analysis
        await analyze_failed_cases(page_embeddings, event_embeddings, ground_truth)

        # Step 6: Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'pages': len(pages),
                'events': len(events),
                'ground_truth_links': len(ground_truth)
            },
            'results': {
                str(k): {
                    'correct': v['correct'],
                    'incorrect': v['incorrect'],
                    'missed': v['missed'],
                    'precision': v['precision'],
                    'recall': v['recall']
                } for k, v in results.items()
            },
            'page_embeddings': [{
                'id': p['id'],
                'title': p['title'],
                'text_preview': p['text_preview']
            } for p in page_embeddings],
            'event_embeddings': [{
                'id': e['id'],
                'title': e['title'],
                'article_count': e['article_count'],
                'text_preview': e['text_preview']
            } for e in event_embeddings]
        }

        with open('/tmp/embedding_experiment_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n\n✅ Results saved to /tmp/embedding_experiment_results.json")
        print("\n💡 Recommendation will be based on best precision/recall balance")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
