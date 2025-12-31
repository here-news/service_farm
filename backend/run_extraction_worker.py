"""
Run Extraction Worker

Launches extraction worker that processes URLs from Redis queue
"""
import os
from pathlib import Path

# Load .env from project root (one level up from backend/)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

import asyncio
import logging
from config import create_postgres_pool, create_job_queue
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
    db_pool = await create_postgres_pool(min_size=2, max_size=5)

    # Connect to Redis job queue
    job_queue = await create_job_queue()

    # Create and start worker
    worker = ExtractionWorker(db_pool, job_queue, worker_id=worker_id)

    logger.info(f"Starting extraction worker {worker_id}")

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        # Cleanup
        await db_pool.close()
        await job_queue.close()
        logger.info("Worker shut down cleanly")


if __name__ == '__main__':
    asyncio.run(main())
