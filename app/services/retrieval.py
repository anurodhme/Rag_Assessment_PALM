"""Retrieval service with cosine similarity and hybrid search strategies."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import SearchStrategy, RetrievedChunk
from app.services.storage_milvus import get_milvus_service

logger = get_logger(__name__)


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        pass


class CosineRetrievalStrategy(RetrievalStrategy):
    """Cosine similarity-based retrieval using Milvus vector search."""
    
    def __init__(self):
        """Initialize cosine retrieval strategy."""
        self.milvus_service = get_milvus_service()
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using cosine similarity search in Milvus.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of retrieved chunks with cosine similarity scores
        """
        try:
            results = self.milvus_service.search_similar_chunks(
                query_text=query,
                top_k=top_k,
                document_ids=document_ids
            )
            
            # Convert to standard format
            retrieved_chunks = []
            for result in results:
                retrieved_chunks.append({
                    "chunk_id": result["chunk_id"],
                    "content": result["content"],
                    "file_id": result["document_id"],
                    "filename": result["filename"],
                    "similarity_score": result["similarity_score"],
                    "search_strategy": SearchStrategy.COSINE,
                    "chunking_strategy": result.get("chunking_strategy", "unknown")
                })
            
            logger.info(f"Cosine retrieval returned {len(retrieved_chunks)} chunks")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Cosine retrieval failed: {e}")
            return []


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining cosine similarity and BM25."""
    
    def __init__(self):
        """Initialize hybrid retrieval strategy."""
        self.milvus_service = get_milvus_service()
        self.cosine_weight = 0.7  # Weight for cosine similarity
        self.bm25_weight = 0.3   # Weight for BM25
    
    def _get_all_chunks_for_bm25(self, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all chunks for BM25 scoring.
        This is a simplified implementation - in production, you might want to cache this.
        
        Args:
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of all chunks with content
        """
        try:
            # Get a large number of chunks to build BM25 corpus
            # In a real implementation, you might want to maintain a separate BM25 index
            all_chunks = self.milvus_service.search_similar_chunks(
                query_text="",  # Empty query to get diverse results
                top_k=1000,     # Large number to get comprehensive corpus
                document_ids=document_ids
            )
            return all_chunks
        except Exception as e:
            logger.error(f"Failed to get chunks for BM25: {e}")
            return []
    
    def _compute_bm25_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute BM25 scores for chunks.
        
        Args:
            query: Search query
            chunks: List of chunks to score
            
        Returns:
            Dictionary mapping chunk_id to BM25 score
        """
        if not chunks:
            return {}
        
        try:
            # Prepare corpus for BM25
            corpus = [chunk["content"] for chunk in chunks]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            
            # Initialize BM25
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Compute scores
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Create mapping from chunk_id to score
            score_map = {}
            for i, chunk in enumerate(chunks):
                score_map[chunk["chunk_id"]] = float(bm25_scores[i])
            
            return score_map
            
        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}")
            return {}
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using hybrid cosine + BM25 approach.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of retrieved chunks with hybrid scores
        """
        try:
            # Step 1: Get cosine similarity results (more than needed for fusion)
            cosine_results = self.milvus_service.search_similar_chunks(
                query_text=query,
                top_k=min(top_k * 3, 50),  # Get more candidates for reranking
                document_ids=document_ids
            )
            
            if not cosine_results:
                logger.warning("No cosine results found")
                return []
            
            # Step 2: Get BM25 corpus (could be optimized with caching)
            all_chunks = self._get_all_chunks_for_bm25(document_ids)
            
            # Step 3: Compute BM25 scores
            bm25_scores = self._compute_bm25_scores(query, all_chunks)
            
            # Step 4: Combine scores
            hybrid_results = []
            for result in cosine_results:
                chunk_id = result["chunk_id"]
                cosine_score = result["similarity_score"]
                bm25_score = bm25_scores.get(chunk_id, 0.0)
                
                # Normalize BM25 score (simple min-max normalization)
                if bm25_scores:
                    max_bm25 = max(bm25_scores.values())
                    min_bm25 = min(bm25_scores.values())
                    if max_bm25 > min_bm25:
                        bm25_score = (bm25_score - min_bm25) / (max_bm25 - min_bm25)
                    else:
                        bm25_score = 0.0
                
                # Compute hybrid score
                hybrid_score = (
                    self.cosine_weight * cosine_score +
                    self.bm25_weight * bm25_score
                )
                
                hybrid_results.append({
                    "chunk_id": result["chunk_id"],
                    "content": result["content"],
                    "file_id": result["document_id"],
                    "filename": result["filename"],
                    "similarity_score": hybrid_score,
                    "cosine_score": cosine_score,
                    "bm25_score": bm25_score,
                    "search_strategy": SearchStrategy.HYBRID,
                    "chunking_strategy": result.get("chunking_strategy", "unknown")
                })
            
            # Step 5: Sort by hybrid score and return top_k
            hybrid_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            final_results = hybrid_results[:top_k]
            
            logger.info(f"Hybrid retrieval returned {len(final_results)} chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # Fallback to cosine similarity
            cosine_strategy = CosineRetrievalStrategy()
            return cosine_strategy.retrieve(query, top_k, document_ids)


class RetrievalService:
    """Main retrieval service that orchestrates different strategies."""
    
    def __init__(self):
        """Initialize retrieval service with available strategies."""
        self.strategies = {
            SearchStrategy.COSINE: CosineRetrievalStrategy(),
            SearchStrategy.HYBRID: HybridRetrievalStrategy()
        }
    
    def retrieve_chunks(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.COSINE,
        top_k: int | None = None,
        document_ids: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using the specified strategy.
        
        Args:
            query: Search query
            strategy: Retrieval strategy to use
            top_k: Number of top results to return (uses default if None)
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of RetrievedChunk objects
            
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unsupported search strategy: {strategy}")
        
        # Use default top_k if not provided
        if top_k is None:
            top_k = settings.default_top_k
        
        # Get retrieval strategy and perform search
        retrieval_strategy = self.strategies[strategy]
        results = retrieval_strategy.retrieve(query, top_k, document_ids)
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for result in results:
            chunk = RetrievedChunk(
                chunk_id=result["chunk_id"],
                content=result["content"],
                file_id=result["file_id"],
                filename=result["filename"],
                similarity_score=result["similarity_score"],
                search_strategy=result["search_strategy"]
            )
            retrieved_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks using {strategy} strategy")
        return retrieved_chunks


# Global retrieval service instance
_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    """
    Get the global retrieval service instance.
    
    Returns:
        RetrievalService instance
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
