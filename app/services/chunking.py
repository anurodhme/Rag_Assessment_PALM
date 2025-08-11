"""Document chunking service with multiple strategies."""

from __future__ import annotations

import re
from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ChunkingStrategy

logger = get_logger(__name__)


class ChunkingService(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
        """
        Chunk text into segments.
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of tuples (chunk_content, start_pos, end_pos)
        """
        pass


class LateChunkingService(ChunkingService):
    """Fixed window chunking strategy."""
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
        """
        Implement fixed window chunking with overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters (approximating tokens)
            overlap: Overlap between chunks in characters
            
        Returns:
            List of tuples (chunk_content, start_pos, end_pos)
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size, text_length)
            
            # Try to break at word boundaries if possible
            if end < text_length:
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunks.append((chunk_content, start, end))
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        logger.info(f"Late chunking created {len(chunks)} chunks")
        return chunks


class SemanticChunkingService(ChunkingService):
    """Semantic similarity-based chunking strategy."""
    
    def __init__(self):
        """Initialize semantic chunking with sentence transformer."""
        self.model = SentenceTransformer(settings.embedding_model)
        self.similarity_threshold = settings.semantic_similarity_threshold
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting - could be improved with spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
        """
        Implement semantic chunking based on sentence similarity.
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            overlap: Minimum overlap in characters (used as fallback)
            
        Returns:
            List of tuples (chunk_content, start_pos, end_pos)
        """
        if not text.strip():
            return []
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            # Fallback to late chunking for very short texts
            late_chunker = LateChunkingService()
            return late_chunker.chunk_text(text, chunk_size, overlap)
        
        # Generate embeddings for sentences
        sentence_embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_start = 0
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and chunk
            current_chunk_text = ' '.join(current_chunk_sentences)
            current_embedding = self.model.encode([current_chunk_text])
            sentence_embedding = sentence_embeddings[i:i+1]
            
            similarity = cosine_similarity(current_embedding, sentence_embedding)[0][0]
            
            # Check if we should continue the current chunk
            should_continue = (
                similarity >= self.similarity_threshold and 
                len(current_chunk_text) + len(sentences[i]) < chunk_size * 1.5  # Allow some flexibility
            )
            
            if should_continue:
                current_chunk_sentences.append(sentences[i])
            else:
                # Finalize current chunk
                chunk_content = ' '.join(current_chunk_sentences)
                chunk_end = current_start + len(chunk_content)
                chunks.append((chunk_content, current_start, chunk_end))
                
                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_start = chunk_end + 1  # +1 for space
        
        # Add the last chunk
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences)
            chunk_end = current_start + len(chunk_content)
            chunks.append((chunk_content, current_start, min(chunk_end, len(text))))
        
        logger.info(f"Semantic chunking created {len(chunks)} chunks")
        return chunks


class ChunkingFactory:
    """Factory for creating chunking services."""
    
    @staticmethod
    def create_chunker(strategy: ChunkingStrategy) -> ChunkingService:
        """
        Create a chunking service based on strategy.
        
        Args:
            strategy: Chunking strategy to use
            
        Returns:
            Chunking service instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy == ChunkingStrategy.LATE:
            return LateChunkingService()
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunkingService()
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")


def chunk_document(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.LATE,
    chunk_size: int | None = None,
    overlap: int | None = None
) -> List[Tuple[str, int, int]]:
    """
    Convenience function to chunk a document.
    
    Args:
        text: Input text to chunk
        strategy: Chunking strategy to use
        chunk_size: Override default chunk size
        overlap: Override default overlap
        
    Returns:
        List of tuples (chunk_content, start_pos, end_pos)
    """
    chunker = ChunkingFactory.create_chunker(strategy)
    
    # Use defaults if not provided
    chunk_size = chunk_size or settings.default_chunk_size
    overlap = overlap or settings.default_chunk_overlap
    
    return chunker.chunk_text(text, chunk_size, overlap)
