"""Evaluation script for benchmarking chunking and search strategies."""

from __future__ import annotations

import time
import json
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ChunkingStrategy, SearchStrategy
from app.services.retrieval import get_retrieval_service
from app.services.embedding import get_embedding_service

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    chunking_strategy: ChunkingStrategy
    search_strategy: SearchStrategy
    precision: float
    recall: float
    f1_score: float
    average_latency_ms: float
    total_queries: int
    successful_queries: int


@dataclass
class QueryGroundTruth:
    """Ground truth data for a single query."""
    query: str
    relevant_chunks: List[str]  # List of chunk IDs that are relevant
    expected_answer: str  # Expected answer for semantic evaluation


class RAGEvaluator:
    """Evaluator for RAG system performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.retrieval_service = get_retrieval_service()
        self.embedding_service = get_embedding_service()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_threshold = 0.8
        
    def load_evaluation_dataset(self, dataset_path: str) -> List[QueryGroundTruth]:
        """
        Load evaluation dataset from JSON file.
        
        Args:
            dataset_path: Path to the evaluation dataset
            
        Returns:
            List of QueryGroundTruth objects
        """
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            dataset = []
            for item in data:
                ground_truth = QueryGroundTruth(
                    query=item['query'],
                    relevant_chunks=item['relevant_chunks'],
                    expected_answer=item.get('expected_answer', '')
                )
                dataset.append(ground_truth)
            
            logger.info(f"Loaded {len(dataset)} queries from {dataset_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def create_sample_dataset(self) -> List[QueryGroundTruth]:
        """
        Create a sample evaluation dataset for demonstration.
        
        Returns:
            List of sample QueryGroundTruth objects
        """
        sample_data = [
            {
                "query": "What is machine learning?",
                "relevant_chunks": ["chunk_1", "chunk_2"],
                "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
            },
            {
                "query": "How does neural network training work?",
                "relevant_chunks": ["chunk_3", "chunk_4"],
                "expected_answer": "Neural network training involves adjusting weights and biases through backpropagation to minimize the loss function."
            },
            {
                "query": "What are the benefits of deep learning?",
                "relevant_chunks": ["chunk_5", "chunk_6"],
                "expected_answer": "Deep learning can automatically learn complex patterns from large amounts of data and achieve state-of-the-art performance in many tasks."
            }
        ]
        
        dataset = []
        for item in sample_data:
            ground_truth = QueryGroundTruth(
                query=item['query'],
                relevant_chunks=item['relevant_chunks'],
                expected_answer=item['expected_answer']
            )
            dataset.append(ground_truth)
        
        return dataset
    
    def compute_retrieval_metrics(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score for retrieval.
        
        Args:
            retrieved_chunks: List of retrieved chunk IDs
            relevant_chunks: List of relevant chunk IDs
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not retrieved_chunks:
            return 0.0, 0.0, 0.0
        
        retrieved_set = set(retrieved_chunks)
        relevant_set = set(relevant_chunks)
        
        # True positives: chunks that are both retrieved and relevant
        tp = len(retrieved_set.intersection(relevant_set))
        
        # False positives: chunks that are retrieved but not relevant
        fp = len(retrieved_set - relevant_set)
        
        # False negatives: chunks that are relevant but not retrieved
        fn = len(relevant_set - retrieved_set)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        try:
            embeddings = self.semantic_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute semantic similarity: {e}")
            return 0.0
    
    async def evaluate_single_query(
        self,
        query_gt: QueryGroundTruth,
        chunking_strategy: ChunkingStrategy,
        search_strategy: SearchStrategy,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate a single query with given strategies.
        
        Args:
            query_gt: Query ground truth data
            chunking_strategy: Chunking strategy to evaluate
            search_strategy: Search strategy to evaluate
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        try:
            # Retrieve chunks
            retrieved_chunks = self.retrieval_service.retrieve_chunks(
                query=query_gt.query,
                strategy=search_strategy,
                top_k=top_k
            )
            
            # Extract chunk IDs
            retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
            
            # Compute retrieval metrics
            precision, recall, f1_score = self.compute_retrieval_metrics(
                retrieved_chunk_ids,
                query_gt.relevant_chunks
            )
            
            # Compute semantic similarity if expected answer is provided
            semantic_similarity = 0.0
            if query_gt.expected_answer and retrieved_chunks:
                # Combine retrieved content
                retrieved_content = " ".join([chunk.content for chunk in retrieved_chunks])
                semantic_similarity = self.compute_semantic_similarity(
                    retrieved_content,
                    query_gt.expected_answer
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "semantic_similarity": semantic_similarity,
                "latency_ms": latency_ms,
                "retrieved_chunks": len(retrieved_chunks),
                "query": query_gt.query
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate query '{query_gt.query}': {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query_gt.query
            }
    
    async def evaluate_strategy_combination(
        self,
        dataset: List[QueryGroundTruth],
        chunking_strategy: ChunkingStrategy,
        search_strategy: SearchStrategy,
        top_k: int = 3
    ) -> EvaluationResult:
        """
        Evaluate a combination of chunking and search strategies.
        
        Args:
            dataset: List of ground truth queries
            chunking_strategy: Chunking strategy to evaluate
            search_strategy: Search strategy to evaluate
            top_k: Number of chunks to retrieve
            
        Returns:
            EvaluationResult object
        """
        logger.info(f"Evaluating {chunking_strategy} + {search_strategy}")
        
        results = []
        successful_queries = 0
        
        # Evaluate each query
        for query_gt in dataset:
            result = await self.evaluate_single_query(
                query_gt, chunking_strategy, search_strategy, top_k
            )
            results.append(result)
            
            if result["success"]:
                successful_queries += 1
        
        # Aggregate results
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            avg_precision = np.mean([r["precision"] for r in successful_results])
            avg_recall = np.mean([r["recall"] for r in successful_results])
            avg_f1 = np.mean([r["f1_score"] for r in successful_results])
            avg_latency = np.mean([r["latency_ms"] for r in successful_results])
        else:
            avg_precision = avg_recall = avg_f1 = avg_latency = 0.0
        
        return EvaluationResult(
            chunking_strategy=chunking_strategy,
            search_strategy=search_strategy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            average_latency_ms=avg_latency,
            total_queries=len(dataset),
            successful_queries=successful_queries
        )
    
    async def run_full_evaluation(
        self,
        dataset: List[QueryGroundTruth],
        top_k: int = 3
    ) -> List[EvaluationResult]:
        """
        Run full evaluation across all strategy combinations.
        
        Args:
            dataset: List of ground truth queries
            top_k: Number of chunks to retrieve
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info("Starting full RAG evaluation")
        
        results = []
        
        # All combinations of strategies
        chunking_strategies = [ChunkingStrategy.LATE, ChunkingStrategy.SEMANTIC]
        search_strategies = [SearchStrategy.COSINE, SearchStrategy.HYBRID]
        
        for chunking_strategy in chunking_strategies:
            for search_strategy in search_strategies:
                result = await self.evaluate_strategy_combination(
                    dataset, chunking_strategy, search_strategy, top_k
                )
                results.append(result)
        
        logger.info(f"Completed evaluation of {len(results)} strategy combinations")
        return results
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        output_path: str = "evaluation/evaluation_report.md"
    ) -> None:
        """
        Generate evaluation report in Markdown format.
        
        Args:
            results: List of evaluation results
            output_path: Path to save the report
        """
        try:
            # Sort results by F1 score
            results.sort(key=lambda x: x.f1_score, reverse=True)
            
            # Generate report content
            report_content = self._generate_report_content(results)
            
            # Save report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Evaluation report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def _generate_report_content(self, results: List[EvaluationResult]) -> str:
        """Generate the content of the evaluation report."""
        from datetime import datetime
        
        report = f"""# RAG System Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the evaluation results for different combinations of chunking and search strategies in the RAG system.

## Methodology

- **Chunking Strategies:** Late-chunking (fixed window) vs Semantic chunking (similarity-based)
- **Search Strategies:** Cosine similarity vs Hybrid (cosine + BM25)
- **Metrics:** Precision, Recall, F1-score, Average Latency
- **Top-K:** 3 chunks retrieved per query

## Results Summary

| Rank | Chunking | Search | Precision | Recall | F1-Score | Latency (ms) | Success Rate |
|------|----------|--------|-----------|--------|----------|--------------|--------------|
"""
        
        for i, result in enumerate(results, 1):
            success_rate = result.successful_queries / result.total_queries * 100
            report += f"| {i} | {result.chunking_strategy.value} | {result.search_strategy.value} | {result.precision:.3f} | {result.recall:.3f} | {result.f1_score:.3f} | {result.average_latency_ms:.1f} | {success_rate:.1f}% |\n"
        
        report += f"""

## Detailed Analysis

### Best Performing Configuration
- **Chunking:** {results[0].chunking_strategy.value}
- **Search:** {results[0].search_strategy.value}
- **F1-Score:** {results[0].f1_score:.3f}
- **Average Latency:** {results[0].average_latency_ms:.1f} ms

### Key Findings

"""
        
        # Add analysis based on results
        if len(results) >= 4:
            late_cosine = next((r for r in results if r.chunking_strategy == ChunkingStrategy.LATE and r.search_strategy == SearchStrategy.COSINE), None)
            late_hybrid = next((r for r in results if r.chunking_strategy == ChunkingStrategy.LATE and r.search_strategy == SearchStrategy.HYBRID), None)
            semantic_cosine = next((r for r in results if r.chunking_strategy == ChunkingStrategy.SEMANTIC and r.search_strategy == SearchStrategy.COSINE), None)
            semantic_hybrid = next((r for r in results if r.chunking_strategy == ChunkingStrategy.SEMANTIC and r.search_strategy == SearchStrategy.HYBRID), None)
            
            report += "#### Chunking Strategy Comparison\n"
            if late_cosine and semantic_cosine:
                if semantic_cosine.f1_score > late_cosine.f1_score:
                    report += "- Semantic chunking outperforms late-chunking with cosine search\n"
                else:
                    report += "- Late-chunking outperforms semantic chunking with cosine search\n"
            
            report += "\n#### Search Strategy Comparison\n"
            if late_cosine and late_hybrid:
                if late_hybrid.f1_score > late_cosine.f1_score:
                    report += "- Hybrid search outperforms cosine-only search with late-chunking\n"
                else:
                    report += "- Cosine-only search outperforms hybrid search with late-chunking\n"
        
        report += f"""

## Performance Metrics

### Precision
Measures the proportion of retrieved chunks that are relevant.

### Recall  
Measures the proportion of relevant chunks that were retrieved.

### F1-Score
Harmonic mean of precision and recall, providing a balanced measure.

### Latency
Average time taken to process each query in milliseconds.

## Recommendations

Based on the evaluation results:

1. **Primary Configuration:** Use {results[0].chunking_strategy.value} chunking with {results[0].search_strategy.value} search for optimal F1-score
2. **Low Latency:** Consider the configuration with the lowest latency if response time is critical
3. **High Precision:** Use the configuration with the highest precision if false positives are costly
4. **High Recall:** Use the configuration with the highest recall if missing relevant information is costly

## Configuration Details

- **Chunk Size:** {settings.default_chunk_size} tokens
- **Chunk Overlap:** {settings.default_chunk_overlap} tokens  
- **Embedding Model:** {settings.embedding_model}
- **Top-K Retrieved:** 3 chunks per query

---

*This report was generated automatically by the RAG Assessment evaluation system.*
"""
        
        return report


async def main():
    """Main evaluation function."""
    evaluator = RAGEvaluator()
    
    # Create or load dataset
    dataset = evaluator.create_sample_dataset()
    
    if not dataset:
        logger.error("No evaluation dataset available")
        return
    
    # Run evaluation
    results = await evaluator.run_full_evaluation(dataset)
    
    # Generate report
    evaluator.generate_report(results)
    
    # Print summary
    print("\n=== RAG Evaluation Results ===")
    for result in results:
        print(f"{result.chunking_strategy.value} + {result.search_strategy.value}: "
              f"F1={result.f1_score:.3f}, Latency={result.average_latency_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
