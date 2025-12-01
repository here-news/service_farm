#!/usr/bin/env python3
"""
Wikidata Enrichment Worker Entry Point

Runs the high-accuracy Wikidata entity linking worker.
"""
import asyncio
import sys
import os

# Add backend to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Set working directory to backend
os.chdir(backend_dir)

from workers.wikidata_worker import main

if __name__ == "__main__":
    asyncio.run(main())
