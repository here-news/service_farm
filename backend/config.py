from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Environment
    environment: str = "development"

    # Google OAuth
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: Optional[str] = None  # Optional: will be auto-constructed from request host

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Database connection components
    db_host: Optional[str] = None
    db_port: int = 5432
    db_user: str = "phi_user"
    db_password: str = "phi_password_dev"
    db_name: str = "phi_here"
    database_url: Optional[str] = None

    # Neo4j connection components
    neo4j_host: Optional[str] = None
    neo4j_bolt_port: int = 7687
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    neo4j_uri: Optional[str] = None

    # OpenAI
    openai_api_key: str = ""

    # Service Farm (extraction worker)
    service_farm_url: str = "http://localhost:8080"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @field_validator('database_url', mode='before')
    @classmethod
    def construct_database_url(cls, v, info):
        """
        Construct database URL based on environment
        - Production: Use localhost (running on same server)
        - Development: Use IPv6 address from db_host
        """
        if v:
            # If DATABASE_URL is explicitly set, use it
            return v

        data = info.data
        environment = data.get('environment', 'development')
        db_user = data.get('db_user', 'phi_user')
        db_password = data.get('db_password', 'phi_password_dev')
        db_port = data.get('db_port', 5432)
        db_name = data.get('db_name', 'phi_here')

        if environment == 'production':
            # Production: Use localhost
            host = 'localhost'
        else:
            # Development: Use IPv6 address
            db_host = data.get('db_host')
            if not db_host:
                raise ValueError("db_host must be set for development environment")
            host = f'[{db_host}]'  # IPv6 addresses need brackets

        return f"postgresql+asyncpg://{db_user}:{db_password}@{host}:{db_port}/{db_name}"

    @field_validator('neo4j_uri', mode='before')
    @classmethod
    def construct_neo4j_uri(cls, v, info):
        """
        Construct Neo4j URI based on environment
        - Production: Use localhost (running on same server)
        - Development: Use IPv6 address from neo4j_host
        """
        if v:
            # If NEO4J_URI is explicitly set, use it
            return v

        data = info.data
        environment = data.get('environment', 'development')
        neo4j_bolt_port = data.get('neo4j_bolt_port', 7687)

        if environment == 'production':
            # Production: Use localhost
            host = 'localhost'
        else:
            # Development: Use IPv6 address
            neo4j_host = data.get('neo4j_host')
            if not neo4j_host:
                raise ValueError("neo4j_host must be set for development environment")
            host = f'[{neo4j_host}]'  # IPv6 addresses need brackets

        return f"bolt://{host}:{neo4j_bolt_port}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
