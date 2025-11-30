"""
Base worker class for Gen2 workers

Combines:
- Demo pattern: Redis queue consumption (BRPOP)
- Gen1 pattern: Signal handling, error handling
- Gen2 pattern: Autonomous decision-making, confidence thresholds
"""
import asyncio
import signal
import logging
from typing import Optional, Tuple
import asyncpg
from services.job_queue import JobQueue

logger = logging.getLogger(__name__)


class BaseWorker:
    """
    Base class for all Gen2 workers

    Borrowed from Gen1:
    - Signal handling (graceful shutdown)
    - Health checks
    - Metrics logging

    Borrowed from Demo:
    - Redis queue consumption (BRPOP)
    - Simple, efficient worker loop

    New for Gen2:
    - Autonomous decision-making (should_process)
    - Confidence-threshold driven
    - Schema-aware (core.* tables)
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        job_queue: JobQueue,
        worker_name: str,
        queue_name: str
    ):
        self.pool = pool
        self.job_queue = job_queue
        self.worker_name = worker_name
        self.queue_name = queue_name
        self.running = False
        self.jobs_processed = 0
        self.jobs_failed = 0

    async def start(self):
        """
        Main worker loop (from demo pattern)

        Continuously:
        1. BRPOP from queue (blocks until job available)
        2. Fetch current state from PostgreSQL
        3. Decide if should process (autonomous)
        4. Process if threshold met
        5. Enqueue next jobs
        """
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        self.running = True
        logger.info(f"[{self.worker_name}] Started, listening on {self.queue_name}")

        while self.running:
            try:
                # Blocking pop from queue
                job = await self.job_queue.dequeue(self.queue_name, timeout=5)

                if job:
                    logger.debug(f"[{self.worker_name}] Received job: {job}")

                    try:
                        # Fetch current state from PostgreSQL
                        state = await self.get_state(job)

                        # Autonomous decision
                        should_process, confidence = await self.should_process(state)

                        if should_process:
                            logger.info(
                                f"[{self.worker_name}] Processing job (confidence={confidence:.2f})"
                            )
                            await self.process(job, state)
                            self.jobs_processed += 1
                        else:
                            logger.info(
                                f"[{self.worker_name}] Skipping job (confidence={confidence:.2f} below threshold)"
                            )

                    except Exception as e:
                        self.jobs_failed += 1
                        logger.error(f"[{self.worker_name}] Job failed: {e}", exc_info=True)
                        await self.handle_error(job, e)

            except asyncio.CancelledError:
                logger.info(f"[{self.worker_name}] Received cancellation signal")
                break
            except Exception as e:
                logger.error(f"[{self.worker_name}] Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info(
            f"[{self.worker_name}] Shutting down. "
            f"Processed: {self.jobs_processed}, Failed: {self.jobs_failed}"
        )

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGTERM/SIGINT"""
        def shutdown_handler(signum, frame):
            logger.info(f"[{self.worker_name}] Received signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

    async def process(self, job: dict, state: dict):
        """
        Override in subclass - do the actual work

        Args:
            job: Job data from queue
            state: Current state from PostgreSQL
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement process()")

    async def should_process(self, state: dict) -> Tuple[bool, float]:
        """
        Autonomous decision based on state

        Override in subclass to implement decision logic

        Returns:
            (should_process, confidence)

        Example:
            if state['word_count'] > 100 and state['language'] == 'en':
                return (True, 0.9)
            return (False, 0.2)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement should_process()")

    async def get_state(self, job: dict) -> dict:
        """
        Fetch current state from PostgreSQL

        Override in subclass to implement state fetching logic

        Args:
            job: Job data from queue

        Returns:
            Current state dict
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_state()")

    async def handle_error(self, job: dict, error: Exception):
        """
        Handle job processing error

        Default: Log error
        Override in subclass for custom error handling (e.g., retry logic)

        Args:
            job: Job that failed
            error: Exception that occurred
        """
        logger.error(
            f"[{self.worker_name}] Error processing job {job}: {error}",
            exc_info=True
        )
