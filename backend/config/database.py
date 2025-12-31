"""
Database Configuration
======================

Centralized database connection configuration for all workers and services.
Handles PostgreSQL, Neo4j, and Redis connections with proper env var handling.
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration."""
    host: str
    port: int
    user: str
    password: str
    database: str
    min_size: int = 2
    max_size: int = 10

    @classmethod
    def from_env(cls, min_size: int = 2, max_size: int = 10) -> 'PostgresConfig':
        """Create config from environment variables."""
        host = os.getenv('POSTGRES_HOST')
        if not host:
            raise ValueError("POSTGRES_HOST environment variable is required")

        return cls(
            host=host,
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            user=os.getenv('POSTGRES_USER', 'phi_user'),
            password=os.getenv('POSTGRES_PASSWORD', ''),
            database=os.getenv('POSTGRES_DB', 'phi_here'),
            min_size=min_size,
            max_size=max_size,
        )

    def to_asyncpg_kwargs(self) -> dict:
        """Convert to asyncpg.create_pool kwargs."""
        return {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'min_size': self.min_size,
            'max_size': self.max_size,
        }


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create config from environment variables."""
        uri = os.getenv('NEO4J_URI')
        if not uri:
            raise ValueError("NEO4J_URI environment variable is required")

        return cls(
            uri=uri,
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', ''),
        )


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    url: str

    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create config from environment variables."""
        url = os.getenv('REDIS_URL')
        if not url:
            raise ValueError("REDIS_URL environment variable is required")

        return cls(url=url)


def get_postgres_config(min_size: int = 2, max_size: int = 10) -> PostgresConfig:
    """Get PostgreSQL configuration from environment."""
    return PostgresConfig.from_env(min_size=min_size, max_size=max_size)


def get_neo4j_config() -> Neo4jConfig:
    """Get Neo4j configuration from environment."""
    return Neo4jConfig.from_env()


def get_redis_config() -> RedisConfig:
    """Get Redis configuration from environment."""
    return RedisConfig.from_env()


async def create_postgres_pool(min_size: int = 2, max_size: int = 10):
    """Create PostgreSQL connection pool from environment config."""
    import asyncpg
    config = get_postgres_config(min_size=min_size, max_size=max_size)
    return await asyncpg.create_pool(**config.to_asyncpg_kwargs())


async def create_job_queue():
    """Create and connect Redis job queue from environment config."""
    from services.job_queue import JobQueue
    config = get_redis_config()
    queue = JobQueue(config.url)
    await queue.connect()
    return queue


async def create_neo4j_service():
    """Create and connect Neo4j service from environment config."""
    from services.neo4j_service import Neo4jService
    # Neo4jService reads from env vars internally
    service = Neo4jService()
    await service.connect()
    return service
