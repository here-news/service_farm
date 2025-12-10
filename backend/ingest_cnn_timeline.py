"""
Ingest CNN timeline page directly via job queue
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.job_queue import JobQueue


async def main():
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    url = "https://ix.cnn.io/dailygraphics/graphics/20251128-hong-kong-fire-timeline/index.html"

    print("=" * 80)
    print("📥 INGESTING CNN TIMELINE PAGE")
    print("=" * 80)
    print()
    print(f"URL: {url}")
    print()

    # Enqueue to extraction worker (high priority)
    await job_queue.enqueue('queue:extraction:high', {
        'url': url,
        'priority': 'high'
    })

    print("✅ Enqueued to extraction worker")
    print()
    print("⏳ Page will go through:")
    print("   1. Extraction Worker → Scrape HTML content")
    print("   2. Knowledge Worker → Extract claims, identify entities")
    print("   3. Event Worker → Match to events")
    print()
    print("Monitor with:")
    print("   docker logs herenews-worker-extraction-1 -f")
    print("   docker logs herenews-worker-knowledge-1 -f")
    print("   docker logs herenews-worker-event -f")

    await job_queue.close()


if __name__ == "__main__":
    asyncio.run(main())
