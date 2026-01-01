"""
Configuration module for database and service connections.
"""
from .database import (
    PostgresConfig,
    Neo4jConfig,
    RedisConfig,
    get_postgres_config,
    get_neo4j_config,
    get_redis_config,
    create_postgres_pool,
    create_job_queue,
    create_neo4j_service,
)
from .settings import Settings, get_settings

__all__ = [
    # Database config
    'PostgresConfig',
    'Neo4jConfig',
    'RedisConfig',
    'get_postgres_config',
    'get_neo4j_config',
    'get_redis_config',
    'create_postgres_pool',
    'create_job_queue',
    'create_neo4j_service',
    # App settings (OAuth, JWT, etc.)
    'Settings',
    'get_settings',
]
