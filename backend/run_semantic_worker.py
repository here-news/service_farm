"""
Run Semantic Worker - Standalone entry point

Listens to queue:semantic:high for pages to process
"""
import asyncio
import logging
from workers.semantic_worker import run_semantic_worker

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🧠 Starting Gen2 Semantic Worker...")
    print("   Listening on: queue:semantic:high")
    print("   Press Ctrl+C to stop\n")

    asyncio.run(run_semantic_worker())
