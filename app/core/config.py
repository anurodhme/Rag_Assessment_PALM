"""Application configuration using Pydantic settings."""

from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    database_url: str = Field(..., description="PostgreSQL database URL")
    
    # Redis Configuration
    redis_url: str = Field(..., description="Redis connection URL")
    
    # Milvus Configuration
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")
    
    # Google Gemini API
    google_api_key: str = Field(..., description="Google API key for Gemini")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    
    # Email Configuration
    smtp_host: str = Field(..., description="SMTP host for email")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: str = Field(..., description="SMTP username")
    smtp_password: str = Field(..., description="SMTP password")
    from_email: str = Field(..., description="From email address")
    
    # Application Settings
    app_name: str = Field(default="RAG Assessment API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Security
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration time")
    
    # Chunking Configuration
    default_chunk_size: int = Field(default=800, description="Default chunk size in tokens")
    default_chunk_overlap: int = Field(default=200, description="Default chunk overlap")
    semantic_similarity_threshold: float = Field(
        default=0.8, description="Threshold for semantic chunking"
    )
    
    # Retrieval Configuration
    default_top_k: int = Field(default=3, description="Default number of chunks to retrieve")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model name"
    )
    
    # File Upload Configuration
    max_file_size_mb: int = Field(default=20, description="Maximum file size in MB")
    allowed_extensions: str = Field(
        default="pdf,txt", description="Allowed file extensions (comma-separated)"
    )
    
    def get_allowed_extensions(self) -> List[str]:
        """Parse allowed extensions from comma-separated string."""
        return [ext.strip() for ext in self.allowed_extensions.split(',')]

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
