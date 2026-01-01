#!/usr/bin/env python3
"""
Weaver Worker Runner
====================

Runs the WeaverWorker that consumes claims from Redis queue
and weaves them into Surfaces (L2 identity clusters).

Queue: claims:pending
Output: Surface nodes in Neo4j + centroids in PostgreSQL

Usage:
    python run_weaver_worker.py
    python run_weaver_worker.py --bootstrap  # Reprocess all claims
"""

import asyncio
import os
import sys
import logging

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workers.weaver_worker import main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    asyncio.run(main())
