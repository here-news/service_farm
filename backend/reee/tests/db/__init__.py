"""
Test Database Infrastructure
============================

Provides test Neo4j and PostgreSQL connections for kernel validation.

Usage:
    from reee.tests.db import TestNeo4jManager

    async with TestNeo4jManager() as neo4j:
        await neo4j.load_fixture("golden_corpus.json")
        # run tests...
"""

from .test_neo4j import TestNeo4jManager

__all__ = ['TestNeo4jManager']
