"""
Redis-based job queue system for Gen2 workers

Uses LPUSH/BRPOP for efficient queue consumption

Queue types:
- Work queues: queue:extraction:high, queue:event:high, etc.
- Command queue: queue:event:command - for sending commands to living events
"""
import json
import redis.asyncio as redis
from typing import Optional, List


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

    async def dequeue_nonblocking(self, queue_name: str) -> Optional[dict]:
        """
        Non-blocking pop from queue (RPOP)

        Returns immediately with job or None.
        Used for command queues where we don't want to block.
        """
        result = await self.redis.rpop(queue_name)
        if result:
            return json.loads(result)
        return None

    async def dequeue_multiple(self, queue_names: List[str], timeout: int = 5) -> Optional[dict]:
        """
        Blocking pop from multiple queues (BRPOP)

        Blocks until job available on any queue or timeout.
        Returns None on timeout.

        Args:
            queue_names: List of queue names to pop from (priority order)
            timeout: Timeout in seconds

        Returns:
            Job dict or None
        """
        result = await self.redis.brpop(queue_names, timeout=timeout)
        if result:
            # result is a tuple: (queue_name, job_json)
            return json.loads(result[1])
        return None

    # Event command helpers
    COMMAND_QUEUE = 'queue:event:command'

    async def send_event_command(self, event_id: str, command: str, params: dict = None):
        """
        Send a command to a living event.

        Commands are MCP-like paths that instruct events to perform actions.

        Args:
            event_id: Target event ID (e.g., 'ev_pth3a8dc')
            command: Command path (e.g., '/retopologize', '/regenerate', '/hibernate')
            params: Optional parameters for the command

        Example:
            await queue.send_event_command('ev_pth3a8dc', '/retopologize')
            await queue.send_event_command('ev_xyz', '/regenerate', {'force': True})
        """
        job = {
            'event_id': event_id,
            'command': command,
            'params': params or {},
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        }
        await self.enqueue(self.COMMAND_QUEUE, job)
