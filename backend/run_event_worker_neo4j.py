"""
Event Worker Runner - Uses EventService for recursive event formation
"""
import os
from pathlib import Path

# Load .env from project root (one level up from backend/)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

import asyncio
from workers.event_worker import main

if __name__ == "__main__":
    asyncio.run(main())
