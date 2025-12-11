"""
Test ABC News disambiguation with domain context
"""
import asyncio
import sys
import logging

sys.path.insert(0, '/app')

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('services.wikidata_client')
logger.setLevel(logging.INFO)

from services.wikidata_client import WikidataClient


async def test_abc_disambiguation():
    """Test ABC News disambiguation using P856 domain matching"""

    client = WikidataClient()

    print("=" * 80)
    print("🧪 Testing ABC News Publisher Disambiguation (P856)")
    print("=" * 80)

    # Test 1: Without domain (old behavior - should be ambiguous)
    print("\n📋 Test 1: Without domain context (generic search)")
    print("-" * 40)
    result = await client.search_entity(
        "ABC News",
        entity_type='ORGANIZATION'
    )
    if result:
        print(f"   Result: {result.get('label')} ({result.get('qid')})")
        print(f"   Accepted: {result.get('accepted')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
    else:
        print("   Result: None (rejected as ambiguous) ✓ Expected")

    # Test 2: Australian ABC with domain (should match via P856)
    print("\n📋 Test 2: search_publisher with abc.net.au")
    print("-" * 40)
    result = await client.search_publisher(
        name="ABC News",
        domain="abc.net.au"
    )
    if result:
        print(f"   Result: {result.get('label')} ({result.get('qid')})")
        print(f"   Accepted: {result.get('accepted')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Match type: {result.get('match_type', 'bayesian')}")
        print(f"   Description: {result.get('description', '')[:80]}...")
        if result.get('qid') == 'Q4650197':
            print("   ✓ Correctly matched Australian ABC!")
    else:
        print("   Result: None (not matched)")

    # Test 3: American ABC with domain
    print("\n📋 Test 3: search_publisher with abcnews.go.com")
    print("-" * 40)
    result = await client.search_publisher(
        name="ABC News",
        domain="abcnews.go.com"
    )
    if result:
        print(f"   Result: {result.get('label')} ({result.get('qid')})")
        print(f"   Accepted: {result.get('accepted')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Match type: {result.get('match_type', 'bayesian')}")
        print(f"   Description: {result.get('description', '')[:80]}...")
        if result.get('qid') == 'Q287171':
            print("   ✓ Correctly matched American ABC!")
    else:
        print("   Result: None (not matched)")

    await client.close()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_abc_disambiguation())
