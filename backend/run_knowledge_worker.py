#!/usr/bin/env python3
"""
Run Knowledge Worker - Unified extraction, identification, and linking pipeline

Replaces semantic_worker + wikidata_worker with a single atomic operation.
Listens to queue:semantic:high for pages to process.
"""
import asyncio
import logging
from workers.knowledge_worker import run_knowledge_worker

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🧠 Starting Knowledge Worker...")
    print("   Pipeline: Extraction → Identification → Linking")
    print("   Listening on: queue:semantic:high")
    print("   Press Ctrl+C to stop\n")

    asyncio.run(run_knowledge_worker())
