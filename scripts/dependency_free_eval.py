#!/usr/bin/env python3
"""
Dependency-free script to evaluate chunking methods.
"""

import time
import re
from typing import List, Tuple
from enum import Enum


class ChunkingStrategy(Enum):
    LATE = "late"
    SEMANTIC = "semantic"


class LateChunkingService:
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
            end = min(start + chunk_size, text_length)
            
            # Find last space to avoid cutting words
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append((chunk_content, start, end))
            
            start = max(start + 1, end - overlap)
            if start >= text_length:
                break
        
        return chunks


class SimpleSemanticChunkingService:
    """Simplified semantic-like chunking strategy using sentence boundaries."""
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
        """
        Implement simplified semantic chunking based on sentence boundaries.
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            overlap: Minimum overlap in characters (used as fallback)
            
        Returns:
            List of tuples (chunk_content, start_pos, end_pos)
        """
        if not text.strip():
            return []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+\s+', text.strip())
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            # Fallback to late chunking for very short texts
            late_chunker = LateChunkingService()
            return late_chunker.chunk_text(text, chunk_size, overlap)
        
        # Group sentences into chunks based on size
        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        current_chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, create a chunk
            if current_chunk_sentences and current_chunk_length + sentence_length > chunk_size:
                chunk_content = ' '.join(current_chunk_sentences)
                # Find the end position
                chunk_end = text.find(current_chunk_sentences[-1]) + len(current_chunk_sentences[-1])
                chunks.append((chunk_content, current_chunk_start, chunk_end))
                
                # Start new chunk with some overlap
                # Find overlap sentences
                overlap_sentences = max(1, len(current_chunk_sentences) // 3)
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_chunk_length = sum(len(s) for s in current_chunk_sentences)
                # Find new start position
                if current_chunk_sentences:
                    current_chunk_start = text.find(current_chunk_sentences[0])
                else:
                    current_chunk_start = text.find(sentence)
            
            # Add sentence to current chunk
            if not current_chunk_sentences:
                current_chunk_start = text.find(sentence)
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length
        
        # Add the last chunk
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences)
            # Find the end position
            chunk_end = text.find(current_chunk_sentences[-1]) + len(current_chunk_sentences[-1])
            chunks.append((chunk_content, current_chunk_start, chunk_end))
        
        return chunks


class ChunkingFactory:
    """Factory for creating chunking services."""
    
    @staticmethod
    def create_chunker(strategy: ChunkingStrategy):
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
            return SimpleSemanticChunkingService()
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")


class ChunkingEvaluator:
    """Class for evaluating chunking methods."""
    
    def __init__(self):
        """Initialize chunking evaluators."""
        self.late_chunker = LateChunkingService()
        self.semantic_chunker = SimpleSemanticChunkingService()
    
    def evaluate_chunking_method(
        self, 
        text: str, 
        strategy: ChunkingStrategy, 
        chunk_size: int = 800, 
        overlap: int = 200
    ) -> dict:
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
        latency = end_time - start_time
        
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
    ) -> dict:
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


def main():
    """Run the dependency-free chunking evaluation."""
    print("Running dependency-free chunking evaluation...")
    
    # Initialize the chunking evaluator
    chunking_evaluator = ChunkingEvaluator()
    
    # Sample texts for chunking evaluation
    sample_texts = [
        """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine learning techniques are a core part of AI. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.""",
        
        """Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.""",
        
        """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, and that using a multilayered network with non-linear activation functions allows approximations of arbitrary functions."""
    ]
    
    # Run chunking evaluation
    print("\n1. Running chunking evaluation...")
    chunking_results = chunking_evaluator.compare_chunking_methods(sample_texts)
    
    # Print chunking evaluation results
    print("\nCHUNKING EVALUATION RESULTS")
    print("="*50)
    
    for strategy, metrics in chunking_results.items():
        print(f"\n{strategy.upper()} CHUNKING:")
        print(f"  Average Number of Chunks: {metrics['avg_num_chunks']:.2f}")
        print(f"  Average Chunk Length: {metrics['avg_chunk_length']:.2f} characters")
        print(f"  Average Latency: {metrics['avg_latency']:.6f} seconds")
    
    # Summary
    print("\n" + "="*50)
    print("\nSUMMARY AND RECOMMENDATIONS")
    print("-"*30)
    
    # Find best chunking method based on latency
    if chunking_results:
        best_chunking = min(chunking_results.items(), key=lambda x: x[1]['avg_latency'])
        print(f"\nBest chunking method (lowest latency): {best_chunking[0]} ({best_chunking[1]['avg_latency']:.6f}s)")
        
        # Compare the methods
        late_metrics = chunking_results.get('late', {})
        semantic_metrics = chunking_results.get('semantic', {})
        
        if late_metrics and semantic_metrics:
            print(f"\nCOMPARISON:")
            print(f"  Late chunking produces {late_metrics['avg_num_chunks']:.2f} chunks on average")
            print(f"  Semantic chunking produces {semantic_metrics['avg_num_chunks']:.2f} chunks on average")
            
            if late_metrics['avg_num_chunks'] > semantic_metrics['avg_num_chunks']:
                print(f"  Semantic chunking creates {((late_metrics['avg_num_chunks'] / semantic_metrics['avg_num_chunks']) - 1)*100:.1f}% fewer chunks")
            else:
                print(f"  Late chunking creates {((semantic_metrics['avg_num_chunks'] / late_metrics['avg_num_chunks']) - 1)*100:.1f}% fewer chunks")
            
            late_latency = late_metrics['avg_latency']
            semantic_latency = semantic_metrics['avg_latency']
            
            if late_latency < semantic_latency:
                print(f"  Late chunking is {((semantic_latency / late_latency) - 1)*100:.1f}% faster")
            else:
                print(f"  Semantic chunking is {((late_latency / semantic_latency) - 1)*100:.1f}% faster")
    
    print("\n" + "="*50)
    
    # Optionally, save results to a file
    import json
    with open("dependency_free_evaluation_results.json", "w") as f:
        json.dump(chunking_results, f, indent=2)
    
    print("\nDetailed results saved to dependency_free_evaluation_results.json")


if __name__ == "__main__":
    main()
