"""Runner for Entity Merge Worker"""
from workers.entity_merge_worker import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
