"""
Event Worker Runner - Uses EventService for recursive event formation
"""
import asyncio
from workers.event_worker import main

if __name__ == "__main__":
    asyncio.run(main())
