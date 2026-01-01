from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can come from:
    - docker-compose.yml environment section
    - .env file (for secrets like API keys)
    - System environment

    Variable names match docker-compose conventions:
    - POSTGRES_HOST, POSTGRES_PORT, etc. (for database)
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (for Neo4j)
    - GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET (for OAuth)
    """

    # Environment
    environment: str = "development"

    # Google OAuth (from .env)
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: Optional[str] = None

    # JWT - uses SECRET_KEY from .env or generates default
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # PostgreSQL (from docker-compose)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "herenews_user"
    postgres_password: str = "herenews_pass"
    postgres_db: str = "herenews"
    database_url: Optional[str] = None

    # Neo4j (from docker-compose)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "herenews_neo4j_pass"
    neo4j_database: str = "neo4j"

    # OpenAI (from .env)
    openai_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Service Farm
    service_farm_url: str = "http://localhost:8080"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars

    @field_validator('jwt_secret_key', mode='before')
    @classmethod
    def get_jwt_secret(cls, v):
        """Use SECRET_KEY from env if JWT_SECRET_KEY not set"""
        if v and v != "dev-secret-key-change-in-production":
            return v
        # Fall back to SECRET_KEY (used in .env)
        return os.getenv('SECRET_KEY', v or 'dev-secret-key-change-in-production')

    @field_validator('database_url', mode='before')
    @classmethod
    def construct_database_url(cls, v, info):
        """Construct database URL from components if not explicitly set"""
        if v:
            return v

        data = info.data
        host = data.get('postgres_host', 'localhost')
        port = data.get('postgres_port', 5432)
        user = data.get('postgres_user', 'herenews_user')
        password = data.get('postgres_password', 'herenews_pass')
        db = data.get('postgres_db', 'herenews')

        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
