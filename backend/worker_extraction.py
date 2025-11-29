#!/usr/bin/env python3
"""
Extraction Worker - Runs independently
"""
import asyncio
import json
import asyncpg
import redis.asyncio as redis
from workers import ExtractionWorker

async def main():
    pool = await asyncpg.create_pool(
        host='demo-postgres',
        port=5432,
        user='demo_user',
        password='demo_pass',
        database='demo_phi_here',
        min_size=1,
        max_size=3
    )

    redis_client = await redis.from_url('redis://demo-redis:6379')
    worker = ExtractionWorker(pool, redis_client)

    print("[ExtractionWorker] Started, listening on queue:extraction:high")

    while True:
        try:
            job_data = await redis_client.brpop('queue:extraction:high', timeout=5)
            if job_data:
                job = json.loads(job_data[1])
                await worker.process(job)
        except Exception as e:
            print(f"[ExtractionWorker] Error: {e}")
            await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())
