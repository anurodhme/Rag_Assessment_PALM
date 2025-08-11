from typing import List, Dict, Any, Tuple
from .chunking_eval import ChunkingEvaluator
from .retrieval_eval import RetrievalEvaluator
from .metrics import EvaluationAggregator
from app.models.schemas import ChunkingStrategy, SearchStrategy


class EvaluationRunner:
    """Main class for running evaluations of chunking and retrieval methods."""
    
    def __init__(self):
        """Initialize evaluators."""
        self.chunking_evaluator = ChunkingEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()
    
    def run_chunking_evaluation(
        self, 
        texts: List[str], 
        chunk_size: int = 800, 
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Run chunking evaluation.
        
        Args:
            texts: List of texts to evaluate on
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            Dictionary with evaluation results
        """
        print("Running chunking evaluation...")
        results = self.chunking_evaluator.compare_chunking_methods(texts, chunk_size, overlap)
        return results
    
    def run_retrieval_evaluation(
        self, 
        queries: List[tuple],  # List of (query, relevant_doc_ids) tuples
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Run retrieval evaluation.
        
        Args:
            queries: List of (query, relevant_doc_ids) tuples
            top_k: Number of top results to return
            
        Returns:
            Dictionary with evaluation results
        """
        print("Running retrieval evaluation...")
        results = self.retrieval_evaluator.compare_retrieval_methods(queries, top_k)
        return results
    
    def run_complete_evaluation(
        self, 
        texts: List[str],
        queries: List[tuple],
        chunk_size: int = 800, 
        overlap: int = 200,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete evaluation of both chunking and retrieval methods.
        
        Args:
            texts: List of texts to evaluate chunking on
            queries: List of (query, relevant_doc_ids) tuples for retrieval evaluation
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            top_k: Number of top results to return
            
        Returns:
            Dictionary with all evaluation results
        """
        print("Running complete evaluation...")
        
        chunking_results = self.run_chunking_evaluation(texts, chunk_size, overlap)
        retrieval_results = self.run_retrieval_evaluation(queries, top_k)
        
        return {
            "chunking_evaluation": chunking_results,
            "retrieval_evaluation": retrieval_results
        }
    
    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted evaluation report.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*80)
        print("RAG SYSTEM EVALUATION REPORT")
        print("="*80)
        
        # Chunking Evaluation Results
        print("\n1. CHUNKING EVALUATION RESULTS")
        print("-"*40)
        
        chunking_results = results.get("chunking_evaluation", {})
        for strategy, metrics in chunking_results.items():
            print(f"\n{strategy.upper()} CHUNKING:")
            print(f"  Average Number of Chunks: {metrics['avg_num_chunks']:.2f}")
            print(f"  Average Chunk Length: {metrics['avg_chunk_length']:.2f} characters")
            print(f"  Average Latency: {metrics['avg_latency']:.4f} seconds")
        
        # Retrieval Evaluation Results
        print("\n2. RETRIEVAL EVALUATION RESULTS")
        print("-"*40)
        
        retrieval_results = results.get("retrieval_evaluation", {})
        for strategy, metrics in retrieval_results.items():
            print(f"\n{strategy.upper()} SEARCH:")
            print(f"  Average Latency: {metrics['avg_latency']:.4f} seconds")
            
            # Print aggregated metrics
            agg_metrics = metrics.get('aggregated_metrics', {})
            if agg_metrics:
                print(f"  Accuracy: {agg_metrics.get('accuracy', 0):.4f}")
                print(f"  Precision: {agg_metrics.get('precision', 0):.4f}")
                print(f"  Recall: {agg_metrics.get('recall', 0):.4f}")
                print(f"  F1-Score: {agg_metrics.get('f1_score', 0):.4f}")
                print(f"  MRR: {agg_metrics.get('mrr', 0):.4f}")
                print(f"  MAP: {agg_metrics.get('map', 0):.4f}")
        
        print("\n" + "="*80)
        
        # Summary
        print("\nSUMMARY AND RECOMMENDATIONS")
        print("-"*40)
        
        # Find best chunking method based on latency
        if chunking_results:
            best_chunking = min(chunking_results.items(), key=lambda x: x[1]['avg_latency'])
            print(f"\nBest chunking method (lowest latency): {best_chunking[0]} ({best_chunking[1]['avg_latency']:.4f}s)")
        
        # Find best retrieval method based on F1-score
        if retrieval_results:
            best_retrieval = max(retrieval_results.items(), 
                               key=lambda x: x[1]['aggregated_metrics'].get('f1_score', 0))
            best_f1 = best_retrieval[1]['aggregated_metrics'].get('f1_score', 0)
            print(f"Best retrieval method (highest F1-score): {best_retrieval[0]} ({best_f1:.4f})")
        
        print("\n" + "="*80)
