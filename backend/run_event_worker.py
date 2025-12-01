"""
Run Event Worker

Launches event worker that processes claims into events via multi-pass clustering
"""
import asyncio
from workers.event_worker import main

if __name__ == '__main__':
    asyncio.run(main())
