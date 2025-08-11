#!/usr/bin/env python3
"""
Simplified script to run the RAG system evaluation without database dependencies.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.chunking_eval import ChunkingEvaluator
from evaluation.metrics import EvaluationAggregator
from app.models.schemas import ChunkingStrategy


def main():
    """Run the simplified RAG system evaluation."""
    print("Running simplified RAG system evaluation...")
    
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
    with open("chunking_evaluation_results.json", "w") as f:
        json.dump(chunking_results, f, indent=2)
    
    print("\nDetailed results saved to chunking_evaluation_results.json")


if __name__ == "__main__":
    main()
