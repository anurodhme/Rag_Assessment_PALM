from typing import List, Tuple, Dict, Any
from app.services.chunking import LateChunkingService, SemanticChunkingService, ChunkingFactory
from app.models.schemas import ChunkingStrategy
from .metrics import RetrievalMetrics
import time


class ChunkingEvaluator:
    """Class for evaluating chunking methods."""
    
    def __init__(self):
        """Initialize chunking evaluators."""
        self.late_chunker = LateChunkingService()
        self.semantic_chunker = SemanticChunkingService()
    
    def evaluate_chunking_method(
        self, 
        text: str, 
        strategy: ChunkingStrategy, 
        chunk_size: int = 800, 
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Evaluate a chunking method on a text.
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy to evaluate
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            Dictionary with evaluation results
        """
        # Start timing
        start_time = time.time()
        
        # Create chunker
        chunker = ChunkingFactory.create_chunker(strategy)
        
        # Perform chunking
        chunks = chunker.chunk_text(text, chunk_size, overlap)
        
        # End timing
        end_time = time.time()
        latency = RetrievalMetrics.compute_latency(start_time, end_time)
        
        # Compute metrics
        num_chunks = len(chunks)
        
        # Calculate average chunk length
        avg_chunk_length = 0
        if chunks:
            total_length = sum(len(chunk[0]) for chunk in chunks)
            avg_chunk_length = total_length / len(chunks)
        
        # Calculate overlap statistics
        overlaps = []
        for i in range(1, len(chunks)):
            prev_end = chunks[i-1][2]  # End position of previous chunk
            curr_start = chunks[i][1]  # Start position of current chunk
            if prev_end > curr_start:  # There is overlap
                overlap_amount = prev_end - curr_start
                overlaps.append(overlap_amount)
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        
        return {
            "strategy": strategy.value,
            "num_chunks": num_chunks,
            "avg_chunk_length": avg_chunk_length,
            "avg_overlap": avg_overlap,
            "latency": latency,
            "chunks": chunks
        }
    
    def compare_chunking_methods(
        self, 
        texts: List[str], 
        chunk_size: int = 800, 
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Compare chunking methods on a set of texts.
        
        Args:
            texts: List of texts to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            Dictionary with comparison results
        """
        strategies = [ChunkingStrategy.LATE, ChunkingStrategy.SEMANTIC]
        results = {}
        
        for strategy in strategies:
            strategy_results = []
            
            for text in texts:
                result = self.evaluate_chunking_method(text, strategy, chunk_size, overlap)
                strategy_results.append(result)
            
            # Aggregate results
            avg_num_chunks = sum(r["num_chunks"] for r in strategy_results) / len(strategy_results)
            avg_chunk_length = sum(r["avg_chunk_length"] for r in strategy_results) / len(strategy_results)
            avg_latency = sum(r["latency"] for r in strategy_results) / len(strategy_results)
            
            results[strategy.value] = {
                "avg_num_chunks": avg_num_chunks,
                "avg_chunk_length": avg_chunk_length,
                "avg_latency": avg_latency,
                "individual_results": strategy_results
            }
        
        return results
