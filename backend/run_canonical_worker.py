#!/usr/bin/env python3
"""
Canonical Worker Runner
========================

Runs the CanonicalWorker that continuously builds canonical
events and entities from L2 surfaces.

Output: Event nodes + enriched Entity nodes in Neo4j

Usage:
    python run_canonical_worker.py
    python run_canonical_worker.py --once  # Single rebuild, then exit
"""

import asyncio
import os
import sys
import logging
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workers.canonical_worker import CanonicalWorker, main
import asyncpg
from services.neo4j_service import Neo4jService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_once():
    """Run a single canonical rebuild."""
    logger.info("Running single canonical rebuild...")

    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'phi_here'),
        user=os.getenv('POSTGRES_USER', 'phi_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'phi_password_dev'),
        min_size=2,
        max_size=10,
    )

    neo4j = Neo4jService()
    await neo4j.connect()

    worker = CanonicalWorker(db_pool=db_pool, neo4j=neo4j)

    try:
        await worker.rebuild_canonical_layer()
        logger.info(
            f"✅ Done: {worker.events_count} events, {worker.entities_count} entities"
        )
    finally:
        await db_pool.close()
        await neo4j.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Worker")
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run single rebuild and exit'
    )
    args = parser.parse_args()

    if args.once:
        asyncio.run(run_once())
    else:
        asyncio.run(main())
