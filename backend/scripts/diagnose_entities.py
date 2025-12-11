#!/usr/bin/env python3
"""
Entity Diagnostic Script - Analyze Knowledge Graph entity quality

Checks for:
1. Missing wikidata_label (has QID but no label)
2. Duplicate QIDs (shouldn't exist)
3. Generic/invalid entity names
4. Encoding issues in aliases
5. High-mention entities without QID
6. Publisher entities without proper flags
"""
import asyncio
import os
from neo4j import AsyncGraphDatabase

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'herenews_neo4j_pass')


async def run_diagnostics():
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    problems = []

    async with driver.session() as session:
        # Problem 1: Missing wikidata_label
        result = await session.run("""
            MATCH (e:Entity)
            WHERE e.wikidata_qid IS NOT NULL AND e.wikidata_label IS NULL
            RETURN count(*) as count, collect(e.canonical_name)[0..5] as examples
        """)
        record = await result.single()
        if record['count'] > 0:
            problems.append({
                'issue': 'Missing wikidata_label (has QID but no authoritative name)',
                'count': record['count'],
                'examples': record['examples'],
                'fix': 'EntityManager.update_qid() should set wikidata_label'
            })

        # Problem 2: Duplicate QIDs
        result = await session.run("""
            MATCH (e:Entity)
            WHERE e.wikidata_qid IS NOT NULL
            WITH e.wikidata_qid as qid, collect(e) as entities
            WHERE size(entities) > 1
            RETURN qid, [ent IN entities | {id: ent.id, name: ent.canonical_name}] as duplicates
        """)
        records = [r async for r in result]
        if records:
            problems.append({
                'issue': 'CRITICAL: Duplicate QIDs (violates single truth)',
                'count': len(records),
                'examples': [{'qid': r['qid'], 'entities': r['duplicates']} for r in records[:5]],
                'fix': 'Run EntityManager.merge_entities() to consolidate'
            })

        # Problem 3: Generic entity names
        result = await session.run("""
            MATCH (e:Entity)
            WHERE e.canonical_name =~ '(?i)^(fire service|construction firm|the|a|an)$'
               OR size(e.canonical_name) <= 2
            RETURN e.id as id, e.canonical_name as name, e.entity_type as type
        """)
        records = [r async for r in result]
        if records:
            problems.append({
                'issue': 'Generic/invalid entity names',
                'count': len(records),
                'examples': [dict(r) for r in records[:5]],
                'fix': 'Improve extraction to skip generic terms'
            })

        # Problem 4: Encoding issues in aliases
        result = await session.run("""
            MATCH (e:Entity)
            WHERE any(a IN coalesce(e.aliases, []) WHERE a =~ '^[?]+$' OR a = '???' OR a = '????')
            RETURN e.id as id, e.canonical_name as name, e.aliases as aliases
        """)
        records = [r async for r in result]
        if records:
            problems.append({
                'issue': 'Corrupted aliases (encoding issues)',
                'count': len(records),
                'examples': [dict(r) for r in records[:5]],
                'fix': 'Check source encoding, filter invalid aliases'
            })

        # Problem 5: High-mention entities without QID
        result = await session.run("""
            MATCH (e:Entity)
            WHERE e.wikidata_qid IS NULL
              AND e.mention_count >= 3
              AND e.is_publisher IS NULL
            RETURN e.id as id, e.canonical_name as name, e.entity_type as type, e.mention_count as mentions
            ORDER BY e.mention_count DESC
            LIMIT 10
        """)
        records = [r async for r in result]
        if records:
            problems.append({
                'issue': 'High-mention entities without Wikidata QID',
                'count': len(records),
                'examples': [dict(r) for r in records],
                'fix': 'Re-run Wikidata search or manual resolution'
            })

        # Summary stats
        result = await session.run("""
            MATCH (e:Entity)
            RETURN count(*) as total,
                   sum(CASE WHEN e.wikidata_qid IS NOT NULL THEN 1 ELSE 0 END) as with_qid,
                   sum(CASE WHEN e.wikidata_label IS NOT NULL THEN 1 ELSE 0 END) as with_label,
                   sum(CASE WHEN e.is_publisher = true THEN 1 ELSE 0 END) as publishers
        """)
        stats = await result.single()

    await driver.close()

    # Print report
    print("\n" + "="*70)
    print("ENTITY DIAGNOSTIC REPORT")
    print("="*70)

    print(f"\n📊 SUMMARY STATS:")
    print(f"   Total entities: {stats['total']}")
    print(f"   With Wikidata QID: {stats['with_qid']} ({100*stats['with_qid']/stats['total']:.1f}%)")
    print(f"   With wikidata_label: {stats['with_label']} ({100*stats['with_label']/stats['total']:.1f}%)")
    print(f"   Publishers: {stats['publishers']}")

    if not problems:
        print("\n✅ No problems detected!")
    else:
        print(f"\n⚠️  PROBLEMS DETECTED: {len(problems)}")
        for i, p in enumerate(problems, 1):
            print(f"\n{i}. {p['issue']}")
            print(f"   Count: {p['count']}")
            print(f"   Examples: {p['examples'][:3]}")
            print(f"   Fix: {p['fix']}")

    print("\n" + "="*70)
    return problems


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
