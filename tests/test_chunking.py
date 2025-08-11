"""Tests for chunking service."""

from __future__ import annotations

import pytest
from app.services.chunking import (
    LateChunkingService,
    SemanticChunkingService,
    ChunkingFactory,
    chunk_document
)
from app.models.schemas import ChunkingStrategy


class TestLateChunkingService:
    """Test late chunking service."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = LateChunkingService()
        text = "This is a test document. It has multiple sentences. We want to chunk it properly."
        
        chunks = chunker.chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, tuple) and len(chunk) == 3 for chunk in chunks)
        assert all(chunk[0].strip() for chunk in chunks)  # No empty chunks
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = LateChunkingService()
        chunks = chunker.chunk_text("", chunk_size=50, overlap=10)
        assert chunks == []
    
    def test_chunk_text_short(self):
        """Test chunking very short text."""
        chunker = LateChunkingService()
        text = "Short text"
        chunks = chunker.chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0][0] == text
        assert chunks[0][1] == 0
        assert chunks[0][2] == len(text)


class TestSemanticChunkingService:
    """Test semantic chunking service."""
    
    def test_chunk_text_basic(self):
        """Test basic semantic chunking."""
        chunker = SemanticChunkingService()
        text = "Machine learning is great. Deep learning is a subset. Neural networks are powerful. Computer vision uses CNNs."
        
        chunks = chunker.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, tuple) and len(chunk) == 3 for chunk in chunks)
    
    def test_chunk_text_single_sentence(self):
        """Test chunking single sentence falls back to late chunking."""
        chunker = SemanticChunkingService()
        text = "This is a single sentence without periods"
        
        chunks = chunker.chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0][0] == text


class TestChunkingFactory:
    """Test chunking factory."""
    
    def test_create_late_chunker(self):
        """Test creating late chunker."""
        chunker = ChunkingFactory.create_chunker(ChunkingStrategy.LATE)
        assert isinstance(chunker, LateChunkingService)
    
    def test_create_semantic_chunker(self):
        """Test creating semantic chunker."""
        chunker = ChunkingFactory.create_chunker(ChunkingStrategy.SEMANTIC)
        assert isinstance(chunker, SemanticChunkingService)
    
    def test_invalid_strategy(self):
        """Test invalid chunking strategy."""
        with pytest.raises(ValueError):
            ChunkingFactory.create_chunker("invalid")


class TestChunkDocument:
    """Test chunk_document convenience function."""
    
    def test_chunk_document_late(self):
        """Test document chunking with late strategy."""
        text = "This is a test document with multiple sentences. It should be chunked properly."
        
        chunks = chunk_document(text, ChunkingStrategy.LATE, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, tuple) and len(chunk) == 3 for chunk in chunks)
    
    def test_chunk_document_semantic(self):
        """Test document chunking with semantic strategy."""
        text = "Machine learning is powerful. Deep learning uses neural networks. Computer vision processes images."
        
        chunks = chunk_document(text, ChunkingStrategy.SEMANTIC, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, tuple) and len(chunk) == 3 for chunk in chunks)
