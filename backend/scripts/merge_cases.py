#!/usr/bin/env python3
"""
Merge Duplicate Cases
=====================

Groups and merges Cases that are about the same story based on:
1. Similar titles (fuzzy matching)
2. Shared core entities
3. Related incidents

Usage:
    docker exec herenews-app python scripts/merge_cases.py
    docker exec herenews-app python scripts/merge_cases.py --dry-run
"""

import asyncio
import argparse
import logging
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.neo4j_service import Neo4jService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def title_similarity(t1: str, t2: str) -> float:
    """Compute title similarity using SequenceMatcher."""
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio()


def entity_overlap(e1: list, e2: list) -> float:
    """Compute Jaccard overlap between entity lists."""
    if not e1 or not e2:
        return 0.0
    s1, s2 = set(e1), set(e2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


async def find_merge_groups(neo4j: Neo4jService) -> list[list[dict]]:
    """Find groups of cases that should be merged."""

    # Load all cases
    result = await neo4j._execute_read('''
        MATCH (c:Case)
        OPTIONAL MATCH (c)-[:CONTAINS]->(i:Incident)
        WITH c, count(i) as incident_count
        RETURN c.id as id, c.title as title, c.core_entities as entities, incident_count
        ORDER BY incident_count DESC
    ''')

    cases = [dict(r) for r in result]
    logger.info(f"Loaded {len(cases)} cases")

    # Group by high similarity
    used = set()
    groups = []

    for i, c1 in enumerate(cases):
        if c1['id'] in used:
            continue

        group = [c1]
        used.add(c1['id'])

        for j, c2 in enumerate(cases[i+1:], i+1):
            if c2['id'] in used:
                continue

            # Check title similarity
            t_sim = title_similarity(c1['title'], c2['title'])

            # Check entity overlap
            e_overlap = entity_overlap(c1['entities'] or [], c2['entities'] or [])

            # Merge if high title similarity OR significant entity overlap
            if t_sim >= 0.6 or e_overlap >= 0.4:
                group.append(c2)
                used.add(c2['id'])
                logger.debug(f"  Grouping {c2['id']} with {c1['id']} (title_sim={t_sim:.2f}, entity_overlap={e_overlap:.2f})")

        if len(group) > 1:
            groups.append(group)

    return groups


async def merge_group(neo4j: Neo4jService, group: list[dict], dry_run: bool = False):
    """Merge a group of cases into the one with most incidents."""

    # Sort by incident count descending - keep the largest
    group.sort(key=lambda x: x['incident_count'], reverse=True)
    primary = group[0]
    others = group[1:]

    logger.info(f"Merging into {primary['id']} ({primary['title'][:50]})")
    for other in others:
        logger.info(f"  ← {other['id']} ({other['title'][:50] if other['title'] else 'No title'})")

    if dry_run:
        return

    # Move all incidents from other cases to primary
    for other in others:
        # Transfer incidents
        await neo4j._execute_write('''
            MATCH (primary:Case {id: $primary_id})
            MATCH (other:Case {id: $other_id})-[r:CONTAINS]->(i:Incident)
            DELETE r
            MERGE (primary)-[:CONTAINS]->(i)
        ''', {'primary_id': primary['id'], 'other_id': other['id']})

        # Merge core_entities
        await neo4j._execute_write('''
            MATCH (primary:Case {id: $primary_id})
            MATCH (other:Case {id: $other_id})
            SET primary.core_entities = primary.core_entities + other.core_entities
        ''', {'primary_id': primary['id'], 'other_id': other['id']})

        # Delete other case
        await neo4j._execute_write('''
            MATCH (c:Case {id: $id})
            DETACH DELETE c
        ''', {'id': other['id']})


async def main(dry_run: bool = False):
    """Main merge routine."""
    neo4j = Neo4jService()
    await neo4j.connect()

    try:
        groups = await find_merge_groups(neo4j)

        if not groups:
            logger.info("No duplicate cases found to merge")
            return

        logger.info(f"Found {len(groups)} groups to merge")

        total_merged = 0
        for group in groups:
            await merge_group(neo4j, group, dry_run)
            total_merged += len(group) - 1

        # Report final state
        result = await neo4j._execute_read('MATCH (c:Case) RETURN count(c) as count')
        logger.info(f"✅ Merged {total_merged} cases. Remaining: {result[0]['count']}")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge duplicate cases")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be merged without doing it')
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
