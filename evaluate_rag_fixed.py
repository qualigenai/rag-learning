"""
RAG Evaluation Pipeline - FIXED for Ollama
This version handles local LLM compatibility issues.
"""

import os
import pandas as pd
from simple_rag import SimpleRAG


class SimpleRAGEvaluator:
    """
    Simplified RAG evaluator that works reliably with Ollama.
    Uses custom metrics that don't require strict JSON formatting.
    """
    
    def __init__(self, rag_system: SimpleRAG):
        """Initialize evaluator with your RAG system."""
        self.rag = rag_system
        
    def evaluate_answer_quality(self, question, answer, contexts):
        """
        Simple answer quality check without complex RAGAS metrics.
        
        Returns scores for:
        - Has answer: Did we get an answer?
        - Answer length: Is it detailed enough?
        - Uses context: Does it reference the retrieved info?
        """
        scores = {}
        
        # Check if we got an answer
        scores['has_answer'] = 1.0 if len(answer.strip()) > 20 else 0.0
        
        # Check answer length (200-1000 chars is good)
        length = len(answer)
        if 200 <= length <= 1000:
            scores['answer_length_score'] = 1.0
        elif length < 200:
            scores['answer_length_score'] = length / 200.0
        else:
            scores['answer_length_score'] = 0.8  # Penalize very long answers
        
        # Check if answer uses context (simple word overlap)
        answer_words = set(answer.lower().split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())
        
        if len(context_words) > 0:
            overlap = len(answer_words & context_words)
            scores['context_usage'] = min(overlap / 50.0, 1.0)  # Cap at 1.0
        else:
            scores['context_usage'] = 0.0
        
        # Check if answer seems relevant to question
        question_words = set(question.lower().split())
        relevance_overlap = len(answer_words & question_words)
        scores['question_relevance'] = min(relevance_overlap / 10.0, 1.0)
        
        return scores
    
    def evaluate_retrieval_quality(self, question, contexts):
        """
        Evaluate how good the retrieved contexts are.
        """
        scores = {}
        
        # Number of contexts retrieved
        scores['num_contexts'] = len(contexts)
        
        # Average context length
        if contexts:
            avg_length = sum(len(ctx) for ctx in contexts) / len(contexts)
            scores['avg_context_length'] = avg_length
        else:
            scores['avg_context_length'] = 0
        
        # Check if contexts seem relevant (simple keyword matching)
        question_words = set(question.lower().split())
        total_overlap = 0
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            total_overlap += len(question_words & ctx_words)
        
        if contexts:
            scores['context_relevance'] = min(total_overlap / (len(contexts) * 5.0), 1.0)
        else:
            scores['context_relevance'] = 0.0
        
        return scores
    
    def run_evaluation(self, test_questions):
        """
        Run complete evaluation on test questions.
        
        Returns DataFrame with all results.
        """
        results = []
        
        print("\n" + "="*60)
        print("RUNNING RAG EVALUATION")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[{i}/{len(test_questions)}] Evaluating: {question[:50]}...")
            
            # Get RAG response
            try:
                response = self.rag.query(question)
                answer = response['result']
                contexts = [doc.page_content for doc in response['source_documents']]
            except Exception as e:
                print(f"  ❌ Error getting response: {e}")
                continue
            
            # Evaluate answer quality
            answer_scores = self.evaluate_answer_quality(question, answer, contexts)
            
            # Evaluate retrieval quality
            retrieval_scores = self.evaluate_retrieval_quality(question, contexts)
            
            # Combine all scores
            result = {
                'question': question,
                'answer': answer,
                'num_contexts': len(contexts),
                **answer_scores,
                **retrieval_scores
            }
            
            results.append(result)
            
            # Show quick feedback
            overall = (answer_scores['has_answer'] + 
                      answer_scores['answer_length_score'] + 
                      answer_scores['context_usage'] +
                      answer_scores['question_relevance']) / 4.0
            print(f"  ✓ Overall Score: {overall:.2f}/1.00")
        
        return pd.DataFrame(results)
    
    def display_results(self, df):
        """Display evaluation results nicely."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Calculate overall statistics
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Questions Evaluated: {len(df)}")
        print(f"  Success Rate: {df['has_answer'].mean()*100:.1f}%")
        
        print("\n📝 ANSWER QUALITY:")
        print(f"  Average Length Score: {df['answer_length_score'].mean():.3f}/1.00")
        print(f"  Average Context Usage: {df['context_usage'].mean():.3f}/1.00")
        print(f"  Average Question Relevance: {df['question_relevance'].mean():.3f}/1.00")
        
        print("\n🔍 RETRIEVAL QUALITY:")
        print(f"  Average Contexts Retrieved: {df['num_contexts'].mean():.1f}")
        print(f"  Average Context Length: {df['avg_context_length'].mean():.0f} chars")
        print(f"  Average Context Relevance: {df['context_relevance'].mean():.3f}/1.00")
        
        # Show individual results
        print("\n📋 DETAILED RESULTS BY QUESTION:")
        print("-" * 60)
        
        for idx, row in df.iterrows():
            print(f"\nQ{idx+1}: {row['question'][:60]}...")
            print(f"  Answer Length: {len(row['answer'])} chars")
            print(f"  Contexts Used: {row['num_contexts']}")
            print(f"  Scores:")
            print(f"    - Answer Quality: {row['answer_length_score']:.2f}")
            print(f"    - Context Usage: {row['context_usage']:.2f}")
            print(f"    - Relevance: {row['question_relevance']:.2f}")
        
        print("\n" + "="*60)
        
    def save_results(self, df, filename="simple_evaluation_results.csv"):
        """Save results to CSV."""
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")


# Sample test questions
SAMPLE_QUESTIONS = [
    "What is RAG and how does it work?",
    "What are the main components of a RAG system?",
    "How does vector search improve retrieval?",
    "What are common challenges in RAG systems?",
    "How do you evaluate RAG performance?",
    "What is Oracle's approach to RAG?",
    "Explain the role of embeddings in RAG",
    "What is the purpose of chunking documents?"
]


def run_simple_evaluation():
    """Run simplified evaluation that works with Ollama."""
    print("="*60)
    print("RAG EVALUATION - SIMPLIFIED VERSION")
    print("(Works reliably with Ollama/local LLMs)")
    print("="*60)
    
    # Load RAG system
    print("\n1️⃣ Loading RAG system...")
    rag = SimpleRAG(data_dir="./data", persist_dir="./chroma_db")
    
    if os.path.exists("./chroma_db"):
        rag.load_and_setup()
        print("  ✓ RAG system loaded!")
    else:
        print("  ❌ ERROR: No RAG system found!")
        print("  Run 'python simple_rag.py' first to build the system.")
        return
    
    # Initialize evaluator
    print("\n2️⃣ Initializing evaluator...")
    evaluator = SimpleRAGEvaluator(rag)
    print("  ✓ Evaluator ready!")
    
    # Run evaluation
    print("\n3️⃣ Running evaluation on test questions...")
    results_df = evaluator.run_evaluation(SAMPLE_QUESTIONS)
    
    # Display results
    print("\n4️⃣ Displaying results...")
    evaluator.display_results(results_df)
    
    # Save results
    print("\n5️⃣ Saving results...")
    evaluator.save_results(results_df)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE! ✅")
    print("="*60)
    print("\n💡 TIP: To improve scores, try:")
    print("  - Adjusting chunk_size in simple_rag.py")
    print("  - Changing the number of contexts retrieved (k value)")
    print("  - Using different embedding models")
    print("  - Testing with different LLMs")
    print("\n" + "="*60)


def custom_evaluation():
    """Run evaluation with custom questions."""
    print("\n" + "="*60)
    print("CUSTOM EVALUATION MODE")
    print("="*60)
    
    rag = SimpleRAG(data_dir="./data", persist_dir="./chroma_db")
    
    if not os.path.exists("./chroma_db"):
        print("❌ ERROR: No RAG system found!")
        print("Run 'python simple_rag.py' first.")
        return
    
    rag.load_and_setup()
    evaluator = SimpleRAGEvaluator(rag)
    
    print("\nEnter your test questions (type 'done' when finished):")
    questions = []
    i = 1
    while True:
        q = input(f"Question {i}: ")
        if q.lower() in ['done', 'exit', 'quit']:
            break
        if q.strip():
            questions.append(q)
            i += 1
    
    if not questions:
        print("No questions provided!")
        return
    
    print("\nRunning evaluation...")
    results_df = evaluator.run_evaluation(questions)
    evaluator.display_results(results_df)
    
    # Save with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"custom_eval_{timestamp}.csv"
    evaluator.save_results(results_df, filename)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "custom":
        custom_evaluation()
    else:
        run_simple_evaluation()
