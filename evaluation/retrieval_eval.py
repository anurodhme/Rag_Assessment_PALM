from typing import List, Dict, Any, Optional, Tuple
from app.services.retrieval import RetrievalService
from app.models.schemas import SearchStrategy, RetrievedChunk
from .metrics import RetrievalMetrics
import time


class RetrievalEvaluator:
    """Class for evaluating retrieval methods."""
    
    def __init__(self):
        """Initialize retrieval service."""
        self.retrieval_service = RetrievalService()
    
    def evaluate_retrieval_method(
        self, 
        query: str, 
        relevant_doc_ids: List[str],
        strategy: SearchStrategy, 
        top_k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a retrieval method on a query.
        
        Args:
            query: Search query
            relevant_doc_ids: List of relevant document IDs
            strategy: Retrieval strategy to evaluate
            top_k: Number of top results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            Dictionary with evaluation results
        """
        # Start timing
        start_time = time.time()
        
        # Perform retrieval
        try:
            retrieved_chunks = self.retrieval_service.retrieve_chunks(
                query=query,
                strategy=strategy,
                top_k=top_k,
                document_ids=document_ids
            )
            
            # Extract document IDs from retrieved chunks
            retrieved_doc_ids = [chunk.file_id for chunk in retrieved_chunks]
            
            # End timing
            end_time = time.time()
            latency = RetrievalMetrics.compute_latency(start_time, end_time)
            
            # Compute metrics
            basic_metrics = RetrievalMetrics.compute_basic_metrics(relevant_doc_ids, retrieved_doc_ids)
            rank_metrics = RetrievalMetrics.compute_rank_based_metrics(relevant_doc_ids, retrieved_doc_ids)
            
            return {
                "strategy": strategy.value,
                "query": query,
                "latency": latency,
                "num_retrieved": len(retrieved_chunks),
                "retrieved_doc_ids": retrieved_doc_ids,
                **basic_metrics,
                **rank_metrics
            }
        except Exception as e:
            # End timing
            end_time = time.time()
            latency = RetrievalMetrics.compute_latency(start_time, end_time)
            
            return {
                "strategy": strategy.value,
                "query": query,
                "latency": latency,
                "error": str(e),
                "num_retrieved": 0,
                "retrieved_doc_ids": [],
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mrr": 0.0,
                "map": 0.0
            }
    
    def compare_retrieval_methods(
        self, 
        queries: List[Tuple[str, List[str]]],  # List of (query, relevant_doc_ids) tuples
        top_k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare retrieval methods on a set of queries.
        
        Args:
            queries: List of (query, relevant_doc_ids) tuples
            top_k: Number of top results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            Dictionary with comparison results
        """
        strategies = [SearchStrategy.COSINE, SearchStrategy.HYBRID]
        results = {}
        
        for strategy in strategies:
            strategy_results = []
            
            for query, relevant_doc_ids in queries:
                result = self.evaluate_retrieval_method(
                    query, relevant_doc_ids, strategy, top_k, document_ids
                )
                strategy_results.append(result)
            
            # Aggregate results
            from .metrics import EvaluationAggregator
            aggregated_metrics = EvaluationAggregator.aggregate_metrics([
                {k: v for k, v in r.items() if k not in ["strategy", "query", "retrieved_doc_ids", "error"]}
                for r in strategy_results
            ])
            
            avg_latency = sum(r["latency"] for r in strategy_results) / len(strategy_results)
            
            results[strategy.value] = {
                "avg_latency": avg_latency,
                "aggregated_metrics": aggregated_metrics,
                "individual_results": strategy_results
            }
        
        return results
