"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    LATE = "late"
    SEMANTIC = "semantic"


class SearchStrategy(str, Enum):
    """Available search strategies."""
    COSINE = "cosine"
    HYBRID = "hybrid"


# Ingestion Models
class IngestionRequest(BaseModel):
    """Request model for document ingestion."""
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.LATE,
        description="Chunking strategy to use"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        description="Override default chunk size"
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="Override default chunk overlap"
    )


class ChunkInfo(BaseModel):
    """Information about a processed chunk."""
    chunk_id: str
    content: str
    start_pos: int
    end_pos: int
    embedding_id: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    file_id: str
    filename: str
    file_size: int
    chunking_strategy: ChunkingStrategy
    total_chunks: int
    chunks: List[ChunkInfo]
    processing_time_seconds: float
    created_at: datetime


# Chat Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Session identifier for conversation memory")
    question: str = Field(..., description="User question")
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.COSINE,
        description="Search strategy to use for retrieval"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve"
    )
    # Optional booking fields
    name: Optional[str] = Field(default=None, description="Name for interview booking")
    email: Optional[str] = Field(default=None, description="Email for interview booking")
    date: Optional[str] = Field(default=None, description="Preferred interview date")
    time: Optional[str] = Field(default=None, description="Preferred interview time")


class RetrievedChunk(BaseModel):
    """Information about a retrieved chunk."""
    chunk_id: str
    content: str
    file_id: str
    filename: str
    similarity_score: float
    search_strategy: SearchStrategy


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    session_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    response_time_seconds: float
    is_booking_request: bool = False
    booking_id: Optional[str] = None


# Interview Booking Models
class InterviewBooking(BaseModel):
    """Interview booking information."""
    booking_id: str
    name: str
    email: str
    date: str
    time: str
    status: str = "confirmed"
    created_at: datetime


class BookingConfirmation(BaseModel):
    """Booking confirmation response."""
    booking_id: str
    message: str
    email_sent: bool


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health Check Models
class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]
