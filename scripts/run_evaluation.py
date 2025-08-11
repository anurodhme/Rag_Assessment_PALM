#!/usr/bin/env python3
"""
Script to run the RAG system evaluation.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.eval_runner import EvaluationRunner


def main():
    """Run the RAG system evaluation."""
    # Initialize the evaluation runner
    evaluator = EvaluationRunner()
    
    # Sample texts for chunking evaluation
    sample_texts = [
        """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine learning techniques are a core part of AI. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.""",
        
        """Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.""",
        
        """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, and that using a multilayered network with non-linear activation functions allows approximations of arbitrary functions."""
    ]
    
    # Sample queries and relevant document IDs for retrieval evaluation
    # In a real evaluation, these would come from a test dataset
    sample_queries = [
        ("What is artificial intelligence?", ["doc1"]),
        ("How do machine learning algorithms work?", ["doc2"]),
        ("What are deep neural networks?", ["doc3"]),
        ("Applications of machine learning in medicine", ["doc2"]),
        ("Difference between AI and machine learning", ["doc1", "doc2"])
    ]
    
    # Run the complete evaluation
    results = evaluator.run_complete_evaluation(
        texts=sample_texts,
        queries=sample_queries,
        chunk_size=800,
        overlap=200,
        top_k=10
    )
    
    # Print the evaluation report
    evaluator.print_evaluation_report(results)
    
    # Optionally, save results to a file
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
