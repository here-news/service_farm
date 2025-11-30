"""
Redis-based job queue system for Gen2 workers

Uses LPUSH/BRPOP for efficient queue consumption
"""
import json
import redis.asyncio as redis
from typing import Optional


class JobQueue:
    """
    Redis-based job queue system

    Queues (one per worker type):
    - 'queue:extraction:high'  → Extraction workers consume
    - 'queue:semantic:high'    → Semantic workers consume
    - 'queue:event:high'       → Event workers consume
    - 'queue:enrichment:high'  → Enrichment workers consume

    Workers use BRPOP (blocking pop) for efficient consumption
    Each job is consumed by exactly ONE worker (round-robin)
    """

    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url

    async def connect(self):
        """Initialize Redis connection"""
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()

    async def enqueue(self, queue_name: str, job: dict):
        """
        Add job to queue

        Example:
            await queue.enqueue('queue:extraction:high', {
                'page_id': '...',
                'url': 'https://...'
            })
        """
        await self.redis.lpush(queue_name, json.dumps(job))

    async def dequeue(self, queue_name: str, timeout: int = 5) -> Optional[dict]:
        """
        Blocking pop from queue (BRPOP)

        Blocks until job available or timeout
        Returns None on timeout

        Args:
            queue_name: Name of queue to pop from
            timeout: Timeout in seconds

        Returns:
            Job dict or None
        """
        result = await self.redis.brpop(queue_name, timeout=timeout)
        if result:
            # result is a tuple: (queue_name, job_json)
            return json.loads(result[1])
        return None

    async def queue_length(self, queue_name: str) -> int:
        """Get current queue length"""
        return await self.redis.llen(queue_name)
