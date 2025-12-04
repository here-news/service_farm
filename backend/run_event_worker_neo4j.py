"""
Event Worker Runner (Neo4j-based)
"""
import asyncio
from workers.event_worker_neo4j import main

if __name__ == "__main__":
    asyncio.run(main())
