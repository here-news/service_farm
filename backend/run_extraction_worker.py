"""
Run Extraction Worker

Launches extraction worker that processes URLs from Redis queue
"""
import os
import asyncio
import logging
import asyncpg
from services.job_queue import JobQueue
from workers.extraction_worker import ExtractionWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main worker entry point"""
    # Get worker ID from environment (for scaling)
    worker_id = int(os.getenv('WORKER_ID', '1'))

    # Connect to PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=2,
        max_size=5
    )

    # Connect to Redis job queue
    job_queue = JobQueue(os.getenv('REDIS_URL', 'redis://redis:6379'))
    await job_queue.connect()

    # Create and start worker
    worker = ExtractionWorker(db_pool, job_queue, worker_id=worker_id)

    logger.info(f"🚀 Starting extraction worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        # Cleanup
        await db_pool.close()
        await job_queue.disconnect()
        logger.info("Worker shut down cleanly")


if __name__ == '__main__':
    asyncio.run(main())
