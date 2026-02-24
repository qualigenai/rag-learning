"""
RAG Evaluation Pipeline using RAGAS
This script demonstrates how to evaluate RAG system performance.
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from simple_rag import SimpleRAG
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGEvaluator:
    """
    Evaluate RAG system performance using RAGAS metrics.
    
    RAGAS Metrics Explained:
    1. Faithfulness: Is the answer grounded in the retrieved context?
    2. Answer Relevancy: Does the answer address the question?
    3. Context Precision: Are the retrieved chunks relevant?
    4. Context Recall: Did we retrieve all relevant information?
    5. Answer Correctness: Compared to ground truth (if available)
    6. Answer Similarity: Semantic similarity to ground truth
    """
    
    def __init__(self, rag_system: SimpleRAG):
        """
        Initialize evaluator.
        
        Args:
            rag_system: Your RAG system to evaluate
        """
        self.rag = rag_system
        
        # Initialize models for RAGAS evaluation
        self.llm = Ollama(model="llama3.2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def prepare_evaluation_dataset(self, test_questions, ground_truths=None):
        """
        Prepare dataset for RAGAS evaluation.
        
        Args:
            test_questions: List of questions to test
            ground_truths: Optional list of expected answers
            
        Returns:
            Dataset ready for RAGAS evaluation
        """
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        print("Generating answers and collecting contexts...")
        for i, question in enumerate(test_questions):
            # Get answer from RAG system
            response = self.rag.query(question)
            
            # Extract contexts (retrieved chunks)
            contexts = [doc.page_content for doc in response['source_documents']]
            
            # Store data
            data["question"].append(question)
            data["answer"].append(response['result'])
            data["contexts"].append(contexts)
            
            # Ground truth (if provided)
            if ground_truths and i < len(ground_truths):
                data["ground_truth"].append(ground_truths[i])
            else:
                data["ground_truth"].append("No ground truth provided")
            
            print(f"  Processed question {i+1}/{len(test_questions)}")
        
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_dict(data)
        return dataset
    
    def evaluate_rag(self, dataset, metrics_to_use=None):
        """
        Run RAGAS evaluation.
        
        Args:
            dataset: Evaluation dataset
            metrics_to_use: List of metrics (None = use all)
            
        Returns:
            Evaluation results
        """
        if metrics_to_use is None:
            # Use metrics that work without ground truth
            metrics_to_use = [
                faithfulness,
                answer_relevancy,
                context_precision,
            ]
        
        print("\nRunning RAGAS evaluation...")
        print(f"Metrics: {[m.name for m in metrics_to_use]}")
        
        # Run evaluation
        results = evaluate(
            dataset,
            metrics=metrics_to_use,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return results
    
    def display_results(self, results):
        """Display evaluation results in a readable format."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Overall scores
        print("\nOverall Scores:")
        for metric, score in results.items():
            if not metric.startswith('question'):
                print(f"  {metric}: {score:.4f}")
        
        # Convert to DataFrame for detailed view
        df = results.to_pandas()
        
        print("\nDetailed Results by Question:")
        print(df.to_string(index=False))
        
        return df
    
    def save_results(self, results, filename="evaluation_results.csv"):
        """Save results to CSV file."""
        df = results.to_pandas()
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")


# Example test questions
SAMPLE_QUESTIONS = [
    "What is RAG and how does it work?",
    "What are the main components of a RAG system?",
    "How does vector search improve retrieval?",
    "What are common challenges in RAG systems?",
    "How do you evaluate RAG performance?"
]

# Optional: Ground truth answers (for more metrics)
SAMPLE_GROUND_TRUTHS = [
    "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation by first retrieving relevant documents, then using them as context for an LLM to generate answers.",
    "The main components are: document loader, text splitter, embeddings model, vector database, retriever, and language model.",
    "Vector search converts text to numerical representations (embeddings) that capture semantic meaning, enabling similarity-based search rather than just keyword matching.",
    "Common challenges include: chunking strategy, retrieval accuracy, handling contradictory information, latency, and hallucinations.",
    "RAG can be evaluated using metrics like faithfulness, answer relevancy, context precision, context recall, and answer correctness."
]


def run_basic_evaluation():
    """Run a basic evaluation example."""
    print("="*60)
    print("RAG EVALUATION EXAMPLE")
    print("="*60)
    
    # Load RAG system
    print("\n1. Loading RAG system...")
    rag = SimpleRAG(data_dir="./data", persist_dir="./chroma_db")
    
    if os.path.exists("./chroma_db"):
        rag.load_and_setup()
    else:
        print("ERROR: No RAG system found. Run simple_rag.py first!")
        return
    
    # Initialize evaluator
    print("\n2. Initializing evaluator...")
    evaluator = RAGEvaluator(rag)
    
    # Prepare evaluation dataset
    print("\n3. Preparing evaluation dataset...")
    eval_dataset = evaluator.prepare_evaluation_dataset(
        test_questions=SAMPLE_QUESTIONS,
        ground_truths=SAMPLE_GROUND_TRUTHS  # Optional
    )
    
    # Run evaluation
    print("\n4. Running evaluation...")
    results = evaluator.evaluate_rag(eval_dataset)
    
    # Display results
    print("\n5. Displaying results...")
    df = evaluator.display_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    # Interpretation guide
    print("\n" + "="*60)
    print("HOW TO INTERPRET SCORES")
    print("="*60)
    print("""
Scores range from 0.0 to 1.0 (higher is better):

Faithfulness (0-1):
  - High (>0.8): Answers are well-grounded in retrieved context
  - Low (<0.5): Answers include hallucinations or unsupported claims

Answer Relevancy (0-1):
  - High (>0.8): Answers directly address the questions
  - Low (<0.5): Answers are off-topic or incomplete

Context Precision (0-1):
  - High (>0.8): Retrieved chunks are highly relevant
  - Low (<0.5): System retrieves too much irrelevant information

Context Recall (0-1):
  - High (>0.8): All relevant information was retrieved
  - Low (<0.5): Important information was missed

What to do if scores are low:
- Adjust chunk size and overlap in text splitter
- Try different embedding models
- Tune retriever parameters (k value, similarity threshold)
- Improve document quality and organization
- Experiment with different LLMs
    """)


def custom_evaluation():
    """Run custom evaluation with your own questions."""
    print("\n" + "="*60)
    print("CUSTOM EVALUATION MODE")
    print("="*60)
    
    rag = SimpleRAG(data_dir="./data", persist_dir="./chroma_db")
    rag.load_and_setup()
    
    evaluator = RAGEvaluator(rag)
    
    print("\nEnter your test questions (type 'done' when finished):")
    questions = []
    i = 1
    while True:
        q = input(f"Question {i}: ")
        if q.lower() == 'done':
            break
        questions.append(q)
        i += 1
    
    if not questions:
        print("No questions provided!")
        return
    
    # Run evaluation
    eval_dataset = evaluator.prepare_evaluation_dataset(questions)
    results = evaluator.evaluate_rag(eval_dataset)
    evaluator.display_results(results)
    evaluator.save_results(results, f"custom_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "custom":
        custom_evaluation()
    else:
        run_basic_evaluation()
