"""Runner for Event Consolidation Worker - Periodic Bayesian Re-evaluation"""
from workers.event_consolidation_worker import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
