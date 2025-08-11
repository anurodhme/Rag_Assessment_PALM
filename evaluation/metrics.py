from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np


class RetrievalMetrics:
    """Class for computing retrieval evaluation metrics."""
    
    @staticmethod
    def compute_basic_metrics(relevant_docs: List[str], retrieved_docs: List[str]) -> Dict[str, float]:
        """
        Compute basic retrieval metrics: accuracy, precision, recall, F1-score.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            
        Returns:
            Dictionary with metric names and values
        """
        # Create sets for easier computation
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        
        # Create binary labels for all unique documents
        all_docs = list(relevant_set.union(retrieved_set))
        true_labels = [1 if doc in relevant_set else 0 for doc in all_docs]
        pred_labels = [1 if doc in retrieved_set else 0 for doc in all_docs]
        
        # Handle edge cases
        if not relevant_set and not retrieved_set:
            return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0}
        
        if not relevant_set:
            return {"accuracy": accuracy_score(true_labels, pred_labels), 
                   "precision": 0.0, 
                   "recall": 0.0, 
                   "f1_score": 0.0}
        
        if not retrieved_set:
            return {"accuracy": accuracy_score(true_labels, pred_labels), 
                   "precision": 0.0, 
                   "recall": 0.0, 
                   "f1_score": 0.0}
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    @staticmethod
    def compute_rank_based_metrics(relevant_docs: List[str], retrieved_docs: List[str]) -> Dict[str, float]:
        """
        Compute rank-based retrieval metrics: MRR, MAP.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs in ranked order
            
        Returns:
            Dictionary with metric names and values
        """
        relevant_set = set(relevant_docs)
        
        if not relevant_set or not retrieved_docs:
            return {"mrr": 0.0, "map": 0.0}
        
        # Mean Reciprocal Rank (MRR)
        first_relevant_rank = None
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                first_relevant_rank = i
                break
        
        mrr = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        
        # Mean Average Precision (MAP)
        num_relevant = len(relevant_set)
        if num_relevant == 0:
            map_score = 0.0
        else:
            avg_precision = 0.0
            num_correct = 0
            
            for i, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_set:
                    num_correct += 1
                    avg_precision += num_correct / i
            
            map_score = avg_precision / num_relevant if num_relevant > 0 else 0.0
        
        return {
            "mrr": mrr,
            "map": map_score
        }
    
    @staticmethod
    def compute_latency(start_time: float, end_time: float) -> float:
        """
        Compute latency in seconds.
        
        Args:
            start_time: Start time in seconds (from time.time())
            end_time: End time in seconds (from time.time())
            
        Returns:
            Latency in seconds
        """
        return end_time - start_time


class GenerationMetrics:
    """Class for computing generation evaluation metrics."""
    
    @staticmethod
    def compute_similarity_metrics(generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Compute text similarity metrics (placeholder implementation).
        In a real implementation, this would use libraries like rouge, bert-score, etc.
        
        Args:
            generated_text: Generated text
            reference_text: Reference text
            
        Returns:
            Dictionary with metric names and values
        """
        # This is a placeholder implementation
        # In a real implementation, you would use libraries like:
        # - rouge for ROUGE metrics
        # - bert-score for BERT-based similarity
        # - sentence-transformers for embedding-based similarity
        
        # Simple overlap-based metrics for demonstration
        gen_words = set(generated_text.lower().split())
        ref_words = set(reference_text.lower().split())
        
        if not gen_words and not ref_words:
            return {"rouge_1": 1.0, "bleu": 1.0, "bert_score": 1.0}
        
        if not gen_words or not ref_words:
            return {"rouge_1": 0.0, "bleu": 0.0, "bert_score": 0.0}
        
        # ROUGE-1 (overlap of unigrams)
        overlap = len(gen_words.intersection(ref_words))
        rouge_1 = 2 * overlap / (len(gen_words) + len(ref_words)) if (len(gen_words) + len(ref_words)) > 0 else 0.0
        
        # BLEU-like precision
        bleu = overlap / len(gen_words) if len(gen_words) > 0 else 0.0
        
        # Placeholder for BERT score
        bert_score = rouge_1  # In real implementation, this would be actual BERT score
        
        return {
            "rouge_1": rouge_1,
            "bleu": bleu,
            "bert_score": bert_score
        }


class EvaluationAggregator:
    """Class for aggregating evaluation results across multiple queries."""
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple queries by computing mean.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary with averaged metrics
        """
        if not metrics_list:
            return {}
        
        # Get all metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        # Compute mean for each metric
        aggregated = {}
        for name in metric_names:
            values = [metrics.get(name, 0.0) for metrics in metrics_list]
            aggregated[name] = np.mean(values)
        
        return aggregated
