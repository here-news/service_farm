"""
Analyze entities in Neo4j graph database

Investigate:
1. What entities exist
2. Why they can't be matched to Wikidata
3. Distribution by type
4. Common patterns in unmatched entities
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService


async def main():
    neo4j = Neo4jService()
    await neo4j.connect()

    print("="*80)
    print("ENTITY ANALYSIS IN NEO4J")
    print("="*80)

    # Get entity statistics
    stats = await neo4j._execute_read("""
        MATCH (e:Entity)
        RETURN
            count(e) as total,
            sum(CASE WHEN e.wikidata_qid IS NOT NULL THEN 1 ELSE 0 END) as enriched,
            sum(CASE WHEN e.wikidata_qid IS NULL THEN 1 ELSE 0 END) as not_enriched
    """, {})

    stat = stats[0]
    print(f"\n📊 Entity Statistics:")
    print(f"   Total entities: {stat['total']}")
    print(f"   ✅ Enriched with QID: {stat['enriched']} ({stat['enriched']/stat['total']*100:.1f}%)")
    print(f"   ❌ Not enriched: {stat['not_enriched']} ({stat['not_enriched']/stat['total']*100:.1f}%)")

    # Distribution by type
    print("\n" + "="*80)
    print("ENTITY TYPE DISTRIBUTION")
    print("="*80)

    by_type = await neo4j._execute_read("""
        MATCH (e:Entity)
        RETURN
            e.entity_type as type,
            count(e) as count,
            sum(CASE WHEN e.wikidata_qid IS NOT NULL THEN 1 ELSE 0 END) as enriched,
            sum(CASE WHEN e.wikidata_qid IS NULL THEN 1 ELSE 0 END) as not_enriched
        ORDER BY count DESC
    """, {})

    for row in by_type:
        enrichment_rate = (row['enriched'] / row['count'] * 100) if row['count'] > 0 else 0
        print(f"\n{row['type'] or 'NULL'}:")
        print(f"   Total: {row['count']}")
        print(f"   ✅ Enriched: {row['enriched']} ({enrichment_rate:.1f}%)")
        print(f"   ❌ Not enriched: {row['not_enriched']}")

    # Show enriched entities
    print("\n" + "="*80)
    print("SUCCESSFULLY ENRICHED ENTITIES")
    print("="*80)

    enriched_entities = await neo4j._execute_read("""
        MATCH (e:Entity)
        WHERE e.wikidata_qid IS NOT NULL
        RETURN
            e.canonical_name as name,
            e.entity_type as type,
            e.wikidata_qid as qid,
            e.wikidata_description as desc,
            e.mention_count as mentions
        ORDER BY e.mention_count DESC
    """, {})

    for ent in enriched_entities:
        print(f"\n✅ {ent['name']} ({ent['type']})")
        print(f"   QID: {ent['qid']}")
        print(f"   Desc: {ent['desc'][:80]}")
        print(f"   Mentions: {ent['mentions']}")

    # Show problematic unenriched entities (high mention count)
    print("\n" + "="*80)
    print("UNENRICHED ENTITIES (High Priority)")
    print("="*80)

    unenriched = await neo4j._execute_read("""
        MATCH (e:Entity)
        WHERE e.wikidata_qid IS NULL
        RETURN
            e.canonical_name as name,
            e.entity_type as type,
            e.mention_count as mentions,
            e.description as description
        ORDER BY e.mention_count DESC
        LIMIT 20
    """, {})

    for i, ent in enumerate(unenriched, 1):
        print(f"\n{i}. ❌ {ent['name']} ({ent['type']})")
        print(f"   Mentions: {ent['mentions']}")
        if ent['description']:
            print(f"   Description: {ent['description'][:100]}...")

    # Analyze why entities failed
    print("\n" + "="*80)
    print("FAILURE ANALYSIS")
    print("="*80)

    # Check for NULL or empty names
    null_names = await neo4j._execute_read("""
        MATCH (e:Entity)
        WHERE e.canonical_name IS NULL OR e.canonical_name = ''
        RETURN count(e) as count
    """, {})

    print(f"\n🔍 Entities with NULL/empty canonical_name: {null_names[0]['count']}")

    if null_names[0]['count'] > 0:
        print("   ⚠️  These cannot be matched to Wikidata!")
        null_examples = await neo4j._execute_read("""
            MATCH (e:Entity)
            WHERE e.canonical_name IS NULL OR e.canonical_name = ''
            RETURN e.id as id, e.entity_type as type, e.mention_count as mentions
            LIMIT 5
        """, {})
        for ex in null_examples:
            print(f"      - ID: {ex['id']}, Type: {ex['type']}, Mentions: {ex['mentions']}")

    # Check for very generic names
    generic_patterns = ['Block 6', 'Building', 'Department', 'Court', 'House']
    print(f"\n🔍 Generic entity names (hard to disambiguate):")

    for pattern in generic_patterns:
        generic = await neo4j._execute_read("""
            MATCH (e:Entity)
            WHERE e.canonical_name CONTAINS $pattern
            RETURN count(e) as count
        """, {'pattern': pattern})

        if generic[0]['count'] > 0:
            print(f"   - Containing '{pattern}': {generic[0]['count']}")

    # Check entities with profile_summary but no QID
    print(f"\n🔍 Entities with profile_summary but no QID:")

    with_profile_no_qid = await neo4j._execute_read("""
        MATCH (e:Entity)
        WHERE e.profile_summary IS NOT NULL
            AND e.profile_summary <> ''
            AND e.wikidata_qid IS NULL
        RETURN
            e.canonical_name as name,
            e.entity_type as type,
            e.profile_summary as profile,
            e.mention_count as mentions
        ORDER BY e.mention_count DESC
        LIMIT 20
    """, {})

    for ent in with_profile_no_qid:
        print(f"\n   - {ent['name']} ({ent['type']}) - {ent['mentions']} mentions")
        print(f"     Profile: {ent['profile'][:150]}...")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\n1. Fix NULL canonical_name entities")
    print("   - These have entity_type but no name")
    print("   - Cannot be matched to Wikidata without a name")
    print("   - Likely bug in semantic extraction")

    print("\n2. Lower confidence threshold for high-quality matches")
    print("   - Current threshold: 0.65 (very conservative)")
    print("   - Consider 0.55 for entities with good context")
    print("   - Or 0.60 as middle ground")

    print("\n3. Improve context scoring for generic names")
    print("   - 'John Lee', 'Block 6' need strong context")
    print("   - Description should mention 'Hong Kong', 'Chief Executive', etc.")
    print("   - Boost score when context matches geographic/temporal markers")

    print("\n4. Manual review and enrichment")
    print("   - High-mention entities without QIDs")
    print("   - These are important for event formation")
    print("   - Consider manual Wikidata links for top 10")

    await neo4j.close()


if __name__ == '__main__':
    asyncio.run(main())
