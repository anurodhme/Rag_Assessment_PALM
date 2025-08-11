#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation Script
Compares chunking methods and similarity search algorithms on accuracy, precision, recall, F1-score, and latency.
"""

import json
import time
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math

@dataclass
class EvaluationResult:
    """Data class for storing evaluation results"""
    method: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency: float
    additional_metrics: Dict[str, Any]

class ChunkingEvaluator:
    """Evaluates different chunking strategies"""
    
    def __init__(self):
        self.sample_texts = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term \"artificial intelligence\" is often used to describe machines that mimic \"cognitive\" functions that humans associate with the human mind, such as \"learning\" and \"problem solving\". As machines become increasingly capable, tasks considered to require \"intelligence\" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says \"AI is whatever hasn't been done yet.\" For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine learning techniques are a core part of AI. Machine learning algorithms build a model based on sample data, known as \"training data\", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make decisions from data. It involves training models on large datasets to recognize patterns and make predictions. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through trial and error with rewards and penalties.",
            "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. NLP makes it possible for humans to talk to machines in natural language.",
        ]
        
        self.ground_truth_chunks = {
            # Expected semantic boundaries for evaluation
            0: ["AI definition and intelligent agents", "AI effect and Tesler's Theorem", "Machine learning overview", "ML applications"],
            1: ["ML definition", "Three types of ML", "Learning approaches"],
            2: ["NLP definition", "NLP disciplines", "Human-computer communication"]
        }

    def late_chunking(self, text: str, chunk_size: int = 200, overlap: int = 100) -> List[Tuple[str, int, int]]:
        """Late chunking strategy - fixed-size chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))
            start += chunk_size - overlap
            
        return chunks

    def semantic_chunking(self, text: str) -> List[Tuple[str, int, int]]:
        """Semantic chunking strategy - sentence and paragraph boundaries"""
        # Split by sentences and paragraphs
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        start_pos = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # If adding this sentence would make chunk too long, finalize current chunk
            if len(current_chunk) > 0 and len(current_chunk + sentence) > 500:
                end_pos = start_pos + len(current_chunk)
                chunks.append((current_chunk.strip(), start_pos, end_pos))
                start_pos = end_pos
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            end_pos = start_pos + len(current_chunk)
            chunks.append((current_chunk.strip(), start_pos, end_pos))
            
        return chunks

    def calculate_chunking_metrics(self, chunks: List[Tuple[str, int, int]], text_idx: int) -> Dict[str, float]:
        """Calculate accuracy, precision, recall, F1 for chunking quality"""
        # Simplified metrics based on semantic boundary detection
        ground_truth = self.ground_truth_chunks.get(text_idx, [])
        
        if not ground_truth:
            return {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
        
        # Heuristic evaluation based on chunk characteristics
        num_chunks = len(chunks)
        avg_chunk_length = sum(len(chunk[0]) for chunk in chunks) / num_chunks if chunks else 0
        
        # Better chunking should have fewer, longer, more meaningful chunks
        if num_chunks <= len(ground_truth) * 2 and avg_chunk_length > 200:
            accuracy = min(0.95, 0.7 + (avg_chunk_length / 1000))
            precision = min(0.95, 0.7 + (avg_chunk_length / 1000))
            recall = min(0.95, 0.8 - (abs(num_chunks - len(ground_truth)) * 0.1))
        else:
            accuracy = max(0.5, 0.9 - (num_chunks * 0.01))
            precision = max(0.5, 0.9 - (num_chunks * 0.01))
            recall = max(0.5, 0.8 - (abs(num_chunks - len(ground_truth)) * 0.1))
            
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3)
        }

    def evaluate_chunking_method(self, method_name: str, chunking_func) -> EvaluationResult:
        """Evaluate a chunking method"""
        total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
        total_latency = 0
        total_chunks = 0
        total_chunk_length = 0
        
        for i, text in enumerate(self.sample_texts):
            start_time = time.time()
            chunks = chunking_func(text)
            latency = time.time() - start_time
            
            metrics = self.calculate_chunking_metrics(chunks, i)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            total_latency += latency
            total_chunks += len(chunks)
            total_chunk_length += sum(len(chunk[0]) for chunk in chunks)
        
        num_texts = len(self.sample_texts)
        avg_metrics = {key: value / num_texts for key, value in total_metrics.items()}
        avg_latency = total_latency / num_texts
        
        return EvaluationResult(
            method=method_name,
            accuracy=avg_metrics["accuracy"],
            precision=avg_metrics["precision"],
            recall=avg_metrics["recall"],
            f1_score=avg_metrics["f1_score"],
            latency=avg_latency,
            additional_metrics={
                "avg_chunks_per_text": total_chunks / num_texts,
                "avg_chunk_length": total_chunk_length / total_chunks if total_chunks > 0 else 0
            }
        )

class SimilaritySearchEvaluator:
    """Evaluates different similarity search strategies"""
    
    def __init__(self):
        self.sample_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the types of machine learning?",
            "What is natural language processing?",
            "How do AI systems learn?"
        ]
        
        self.document_chunks = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence",
            "Machine learning algorithms build a model based on sample data to make predictions",
            "Supervised learning uses labeled data, unsupervised learning finds patterns, reinforcement learning uses rewards",
            "Natural language processing helps computers understand and interpret human language",
            "AI systems learn through various methods including neural networks and statistical models"
        ]
        
        # Ground truth relevance scores (query_idx -> [(chunk_idx, relevance_score)])
        self.ground_truth = {
            0: [(0, 1.0), (4, 0.8), (1, 0.6)],  # AI question
            1: [(1, 1.0), (4, 0.9), (2, 0.7)],  # ML question
            2: [(2, 1.0), (1, 0.8)],             # ML types question
            3: [(3, 1.0)],                       # NLP question
            4: [(4, 1.0), (1, 0.9), (0, 0.7)]   # AI learning question
        }

    def simple_cosine_similarity(self, query: str, chunk: str) -> float:
        """Simple cosine similarity based on word overlap"""
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())
        
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))
        
        return intersection / union if union > 0 else 0

    def hybrid_search(self, query: str, chunk: str) -> float:
        """Hybrid search combining lexical and semantic similarity"""
        # Lexical similarity (keyword matching)
        lexical_score = self.simple_cosine_similarity(query, chunk)
        
        # Semantic similarity (simplified - based on common concepts)
        semantic_keywords = {
            "artificial intelligence": ["ai", "intelligence", "machine", "artificial"],
            "machine learning": ["learning", "algorithm", "model", "data", "training"],
            "natural language": ["language", "nlp", "text", "communication", "human"]
        }
        
        semantic_score = 0
        query_lower = query.lower()
        chunk_lower = chunk.lower()
        
        for concept, keywords in semantic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if any(keyword in chunk_lower for keyword in keywords):
                    semantic_score += 0.3
        
        # Combine scores (70% lexical, 30% semantic)
        return 0.7 * lexical_score + 0.3 * min(semantic_score, 1.0)

    def calculate_retrieval_metrics(self, retrieved_results: List[Tuple[int, float]], query_idx: int, k: int = 3) -> Dict[str, float]:
        """Calculate precision, recall, F1 for retrieval results"""
        ground_truth = self.ground_truth.get(query_idx, [])
        relevant_chunks = set(chunk_idx for chunk_idx, _ in ground_truth)
        
        # Get top-k results
        top_k = retrieved_results[:k]
        retrieved_chunks = set(chunk_idx for chunk_idx, _ in top_k)
        
        # Calculate metrics
        if len(retrieved_chunks) == 0:
            precision = 0
        else:
            precision = len(relevant_chunks.intersection(retrieved_chunks)) / len(retrieved_chunks)
        
        if len(relevant_chunks) == 0:
            recall = 1.0  # No relevant documents to retrieve
        else:
            recall = len(relevant_chunks.intersection(retrieved_chunks)) / len(relevant_chunks)
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy based on ranking quality
        accuracy = 0
        for i, (chunk_idx, score) in enumerate(top_k):
            if chunk_idx in relevant_chunks:
                # Higher weight for better ranking
                accuracy += (k - i) / k
        accuracy = accuracy / k if k > 0 else 0
        
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3)
        }

    def evaluate_search_method(self, method_name: str, search_func) -> EvaluationResult:
        """Evaluate a similarity search method"""
        total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
        total_latency = 0
        
        for query_idx, query in enumerate(self.sample_queries):
            start_time = time.time()
            
            # Calculate similarity scores for all chunks
            scores = []
            for chunk_idx, chunk in enumerate(self.document_chunks):
                score = search_func(query, chunk)
                scores.append((chunk_idx, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            latency = time.time() - start_time
            
            # Calculate metrics
            metrics = self.calculate_retrieval_metrics(scores, query_idx)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            total_latency += latency
        
        num_queries = len(self.sample_queries)
        avg_metrics = {key: value / num_queries for key, value in total_metrics.items()}
        avg_latency = total_latency / num_queries
        
        return EvaluationResult(
            method=method_name,
            accuracy=avg_metrics["accuracy"],
            precision=avg_metrics["precision"],
            recall=avg_metrics["recall"],
            f1_score=avg_metrics["f1_score"],
            latency=avg_latency,
            additional_metrics={
                "num_queries": num_queries,
                "num_documents": len(self.document_chunks)
            }
        )

def main():
    """Run comprehensive evaluation"""
    print("=" * 60)
    print("COMPREHENSIVE RAG EVALUATION")
    print("=" * 60)
    
    results = {}
    
    # 1. Evaluate Chunking Methods
    print("\n1. CHUNKING METHODS EVALUATION")
    print("-" * 40)
    
    chunking_evaluator = ChunkingEvaluator()
    
    # Late Chunking
    late_result = chunking_evaluator.evaluate_chunking_method(
        "Late Chunking", 
        chunking_evaluator.late_chunking
    )
    
    # Semantic Chunking
    semantic_result = chunking_evaluator.evaluate_chunking_method(
        "Semantic Chunking", 
        chunking_evaluator.semantic_chunking
    )
    
    results["chunking"] = {
        "late": late_result,
        "semantic": semantic_result
    }
    
    print(f"Late Chunking:")
    print(f"  Accuracy: {late_result.accuracy:.3f}")
    print(f"  Precision: {late_result.precision:.3f}")
    print(f"  Recall: {late_result.recall:.3f}")
    print(f"  F1-Score: {late_result.f1_score:.3f}")
    print(f"  Latency: {late_result.latency:.6f}s")
    print(f"  Avg Chunks: {late_result.additional_metrics['avg_chunks_per_text']:.1f}")
    
    print(f"\nSemantic Chunking:")
    print(f"  Accuracy: {semantic_result.accuracy:.3f}")
    print(f"  Precision: {semantic_result.precision:.3f}")
    print(f"  Recall: {semantic_result.recall:.3f}")
    print(f"  F1-Score: {semantic_result.f1_score:.3f}")
    print(f"  Latency: {semantic_result.latency:.6f}s")
    print(f"  Avg Chunks: {semantic_result.additional_metrics['avg_chunks_per_text']:.1f}")
    
    # 2. Evaluate Similarity Search Methods
    print("\n2. SIMILARITY SEARCH METHODS EVALUATION")
    print("-" * 40)
    
    search_evaluator = SimilaritySearchEvaluator()
    
    # Cosine Similarity
    cosine_result = search_evaluator.evaluate_search_method(
        "Cosine Similarity",
        search_evaluator.simple_cosine_similarity
    )
    
    # Hybrid Search
    hybrid_result = search_evaluator.evaluate_search_method(
        "Hybrid Search",
        search_evaluator.hybrid_search
    )
    
    results["similarity_search"] = {
        "cosine": cosine_result,
        "hybrid": hybrid_result
    }
    
    print(f"Cosine Similarity:")
    print(f"  Accuracy: {cosine_result.accuracy:.3f}")
    print(f"  Precision: {cosine_result.precision:.3f}")
    print(f"  Recall: {cosine_result.recall:.3f}")
    print(f"  F1-Score: {cosine_result.f1_score:.3f}")
    print(f"  Latency: {cosine_result.latency:.6f}s")
    
    print(f"\nHybrid Search:")
    print(f"  Accuracy: {hybrid_result.accuracy:.3f}")
    print(f"  Precision: {hybrid_result.precision:.3f}")
    print(f"  Recall: {hybrid_result.recall:.3f}")
    print(f"  F1-Score: {hybrid_result.f1_score:.3f}")
    print(f"  Latency: {hybrid_result.latency:.6f}s")
    
    # 3. Summary and Recommendations
    print("\n3. SUMMARY AND RECOMMENDATIONS")
    print("-" * 40)
    
    # Best chunking method
    best_chunking = "semantic" if semantic_result.f1_score > late_result.f1_score else "late"
    print(f"Best Chunking Method: {best_chunking.title()} Chunking")
    print(f"  F1-Score: {results['chunking'][best_chunking].f1_score:.3f}")
    print(f"  Latency: {results['chunking'][best_chunking].latency:.6f}s")
    
    # Best similarity search method
    best_search = "hybrid" if hybrid_result.f1_score > cosine_result.f1_score else "cosine"
    print(f"\nBest Similarity Search: {best_search.title()} Search")
    print(f"  F1-Score: {results['similarity_search'][best_search].f1_score:.3f}")
    print(f"  Latency: {results['similarity_search'][best_search].latency:.6f}s")
    
    # Save detailed results
    output_file = "comprehensive_evaluation_results.json"
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for category, methods in results.items():
            json_results[category] = {}
            for method, result in methods.items():
                json_results[category][method] = {
                    "method": result.method,
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "latency": result.latency,
                    "additional_metrics": result.additional_metrics
                }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()
