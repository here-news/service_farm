#!/usr/bin/env python3
"""
Neo4j Migration: Local -> Remote
Issue #21: Infrastructure consolidation

Migrates all nodes and relationships from local Neo4j to remote.
Uses neo4j Python driver for reliable data transfer.

Usage:
    python scripts/migrate_neo4j.py [--dry-run]
"""

import os
import sys
from neo4j import GraphDatabase

# Local Neo4j
LOCAL_URI = os.environ.get("LOCAL_NEO4J_URI", "bolt://localhost:7687")
LOCAL_USER = os.environ.get("LOCAL_NEO4J_USER", "neo4j")
LOCAL_PASS = os.environ.get("LOCAL_NEO4J_PASS", "herenews_neo4j_pass")

# Remote Neo4j (IPv6)
REMOTE_URI = os.environ.get("REMOTE_NEO4J_URI", "bolt://[200:d9ce:5252:7e25:c770:1715:a84e:ab3a]:7687")
REMOTE_USER = os.environ.get("REMOTE_NEO4J_USER", "neo4j")
REMOTE_PASS = os.environ.get("REMOTE_NEO4J_PASS", "ITai3wrEOWoOwh3XKltIkS612HyAu1A2wa5Xwx2XUA8=")


def get_node_counts(session):
    """Get count of nodes by label"""
    result = session.run("""
        MATCH (n)
        RETURN labels(n)[0] as label, count(*) as count
        ORDER BY count DESC
    """)
    return {r["label"]: r["count"] for r in result}


def export_nodes(session, label):
    """Export all nodes of a given label"""
    result = session.run(f"""
        MATCH (n:{label})
        RETURN properties(n) as props, n.id as id
    """)
    return [{"props": dict(r["props"]), "id": r["id"]} for r in result]


def export_relationships(session):
    """Export all relationships"""
    result = session.run("""
        MATCH (a)-[r]->(b)
        RETURN
            labels(a)[0] as from_label,
            a.id as from_id,
            type(r) as rel_type,
            properties(r) as rel_props,
            labels(b)[0] as to_label,
            b.id as to_id
    """)
    return [dict(r) for r in result]


def clear_remote(session):
    """Clear all data from remote Neo4j"""
    print("  Clearing remote Neo4j...")
    session.run("MATCH (n) DETACH DELETE n")
    print("  ✓ Cleared")


def import_nodes(session, label, nodes):
    """Import nodes to remote"""
    if not nodes:
        return 0

    # Create constraints first (if not exists)
    try:
        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
    except:
        pass

    # Batch import
    count = 0
    batch_size = 100
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        session.run(f"""
            UNWIND $nodes as node
            CREATE (n:{label})
            SET n = node.props
        """, nodes=batch)
        count += len(batch)

    return count


def import_relationships(session, rels):
    """Import relationships to remote"""
    if not rels:
        return 0

    count = 0
    for rel in rels:
        try:
            session.run(f"""
                MATCH (a:{rel['from_label']} {{id: $from_id}})
                MATCH (b:{rel['to_label']} {{id: $to_id}})
                CREATE (a)-[r:{rel['rel_type']}]->(b)
                SET r = $props
            """, from_id=rel['from_id'], to_id=rel['to_id'], props=rel['rel_props'] or {})
            count += 1
        except Exception as e:
            print(f"    Warning: Could not create relationship {rel['rel_type']}: {e}")

    return count


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 50)
    print("Neo4j Migration: Local -> Remote")
    print("=" * 50)

    if dry_run:
        print("DRY RUN - no changes will be made to remote")

    # Connect to local
    print("\n[1] Connecting to local Neo4j...")
    local_driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PASS))

    with local_driver.session() as session:
        local_counts = get_node_counts(session)
        print(f"  Local node counts: {local_counts}")

    # Connect to remote
    print("\n[2] Connecting to remote Neo4j...")
    remote_driver = GraphDatabase.driver(REMOTE_URI, auth=(REMOTE_USER, REMOTE_PASS))

    with remote_driver.session() as session:
        remote_counts = get_node_counts(session)
        print(f"  Remote node counts (before): {remote_counts}")

    if dry_run:
        print("\nDry run complete. To migrate, run without --dry-run")
        return

    # Confirm
    print("\n⚠️  This will DELETE all data on remote and replace with local data.")
    confirm = input("Continue? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Clear remote
    print("\n[3] Clearing remote...")
    with remote_driver.session() as session:
        clear_remote(session)

    # Export and import nodes
    print("\n[4] Migrating nodes...")
    with local_driver.session() as local_session:
        for label in local_counts.keys():
            print(f"  Exporting {label}...", end=" ")
            nodes = export_nodes(local_session, label)
            print(f"{len(nodes)} nodes")

            with remote_driver.session() as remote_session:
                imported = import_nodes(remote_session, label, nodes)
                print(f"    ✓ Imported {imported} {label} nodes")

    # Export and import relationships
    print("\n[5] Migrating relationships...")
    with local_driver.session() as local_session:
        rels = export_relationships(local_session)
        print(f"  Exported {len(rels)} relationships")

    with remote_driver.session() as remote_session:
        imported = import_relationships(remote_session, rels)
        print(f"  ✓ Imported {imported} relationships")

    # Verify
    print("\n[6] Verification...")
    with remote_driver.session() as session:
        final_counts = get_node_counts(session)
        print(f"  Remote node counts (after): {final_counts}")

    # Summary
    print("\n" + "=" * 50)
    print("Migration Complete!")
    print("=" * 50)

    local_driver.close()
    remote_driver.close()


if __name__ == "__main__":
    main()
