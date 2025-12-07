#!/usr/bin/env python3
"""
Merge duplicate entities based on normalized names.

Usage:
    python merge_duplicate_entities.py --preview  # Show duplicates without merging
    python merge_duplicate_entities.py --merge    # Actually merge duplicates
"""
import asyncio
import os
import sys
import re
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.neo4j_service import Neo4jService


def normalize_name(name: str) -> str:
    """Normalize entity name for matching."""
    normalized = name.lower()
    # Remove possessives
    normalized = re.sub(r"['']s\b", "", normalized)
    # Remove punctuation except hyphens
    normalized = re.sub(r"[^\w\s\-]", "", normalized)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def canonical_name_score(name: str) -> int:
    """
    Score for selecting best canonical name.
    Higher = better. Prefer names without possessives, with proper case.
    """
    score = 0
    # Penalize possessives
    if "'s" in name or "'s" in name:
        score -= 10
    # Prefer title case (proper capitalization)
    if name[0].isupper():
        score += 5
    # Prefer longer names (more specific)
    score += len(name) // 10
    return score


async def find_duplicates(neo4j: Neo4jService):
    """Find all duplicate entity groups."""
    results = await neo4j._execute_read("""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)<-[:INVOLVES]-(ev:Event)
        WITH e, count(ev) as event_count
        RETURN e.id as id,
               e.canonical_name as canonical_name,
               e.entity_type as entity_type,
               e.mention_count as mention_count,
               e.wikidata_qid as wikidata_qid,
               event_count
    """, {})

    # Group by normalized name + type
    groups = defaultdict(list)
    for row in results:
        normalized = normalize_name(row['canonical_name'])
        key = (normalized, row['entity_type'])
        groups[key].append(row)

    # Filter to only duplicates and sort each group
    duplicates = []
    for key, entities in groups.items():
        if len(entities) > 1:
            # Sort by: wikidata_qid presence, mention_count, canonical_name_score
            entities.sort(key=lambda e: (
                -1 if e.get('wikidata_qid') else 0,  # Prefer enriched
                -e.get('mention_count', 0),          # Higher mentions
                -canonical_name_score(e['canonical_name'])  # Cleaner name
            ))
            duplicates.append({
                'normalized': key[0],
                'type': key[1],
                'entities': entities
            })

    return sorted(duplicates, key=lambda x: -len(x['entities']))


async def merge_group(neo4j: Neo4jService, group: dict):
    """Merge a group of duplicate entities."""
    entities = group['entities']
    canonical = entities[0]  # Keep first (highest mention_count)
    duplicates = entities[1:]

    print(f"  📌 Keeping: {canonical['canonical_name']} (mentions: {canonical['mention_count']}, events: {canonical['event_count']})")

    for dup in duplicates:
        print(f"     🔄 Merging: {dup['canonical_name']} → {canonical['canonical_name']}")

        # Step 1: Transfer INVOLVES relationships from Events
        await neo4j._execute_write("""
            MATCH (dup:Entity {id: $dup_id})<-[r:INVOLVES]-(e:Event)
            MATCH (canonical:Entity {id: $canonical_id})
            MERGE (e)-[:INVOLVES]->(canonical)
            DELETE r
        """, {'dup_id': dup['id'], 'canonical_id': canonical['id']})

        # Step 2: Transfer any other relationships
        await neo4j._execute_write("""
            MATCH (dup:Entity {id: $dup_id})-[r]-(other)
            WHERE NOT other:Entity OR other.id <> $canonical_id
            MATCH (canonical:Entity {id: $canonical_id})
            WITH dup, canonical, type(r) as rel_type, other, r
            // We can't dynamically create relationship types in plain Cypher
            // So just delete the duplicate after transferring INVOLVES
            RETURN count(r) as remaining
        """, {'dup_id': dup['id'], 'canonical_id': canonical['id']})

        # Step 3: Delete duplicate entity
        await neo4j._execute_write("""
            MATCH (dup:Entity {id: $dup_id})
            DETACH DELETE dup
        """, {'dup_id': dup['id']})

        print(f"     ✓ Deleted {dup['canonical_name']}")


async def main():
    parser = argparse.ArgumentParser(description='Merge duplicate entities')
    parser.add_argument('--preview', action='store_true', help='Preview duplicates without merging')
    parser.add_argument('--merge', action='store_true', help='Actually merge duplicates')
    args = parser.parse_args()

    if not args.preview and not args.merge:
        parser.print_help()
        return

    # Connect to Neo4j
    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        duplicates = await find_duplicates(neo4j)

        if not duplicates:
            print("✅ No duplicate entities found!")
            return

        print(f"\n📊 Found {len(duplicates)} groups of duplicate entities:\n")

        for group in duplicates:
            print(f"[{group['type']}] '{group['normalized']}' ({len(group['entities'])} duplicates):")
            for e in group['entities']:
                qid = f" → {e['wikidata_qid']}" if e.get('wikidata_qid') else ""
                print(f"    - {e['canonical_name']} (mentions: {e['mention_count']}, events: {e['event_count']}){qid}")
            print()

        if args.merge:
            confirm = input(f"\n⚠️  Merge {sum(len(g['entities'])-1 for g in duplicates)} duplicate entities? [y/N]: ")
            if confirm.lower() == 'y':
                print("\n🔄 Merging duplicates...\n")
                for group in duplicates:
                    await merge_group(neo4j, group)
                print(f"\n✅ Merged {sum(len(g['entities'])-1 for g in duplicates)} duplicate entities!")
            else:
                print("Cancelled.")

    finally:
        await neo4j.close()


if __name__ == '__main__':
    asyncio.run(main())
