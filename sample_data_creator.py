"""
Sample Data Creator
Generates test documents about RAG systems for learning and testing.
"""

import os


def create_sample_documents():
    """Create sample text files about RAG systems."""
    
    # Create data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    
    documents = {
        "rag_introduction.txt": """
What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
with text generation. Instead of relying solely on a language model's training data, 
RAG first retrieves relevant documents from a knowledge base, then uses those documents 
as context for generating answers.

The key benefit of RAG is that it allows language models to access up-to-date information 
and domain-specific knowledge without requiring expensive model retraining. This makes 
RAG particularly valuable for enterprise applications where data changes frequently.

RAG systems typically consist of several components: a document store, an embedding model 
to convert text into vectors, a vector database for efficient similarity search, and a 
language model to generate final answers based on retrieved context.
""",

        "rag_components.txt": """
Core Components of a RAG System

1. Document Loader
The document loader ingests various file formats (PDF, text, HTML, etc.) and converts 
them into a standardized format for processing.

2. Text Splitter
Documents are broken into smaller chunks to improve retrieval accuracy and fit within 
the context window of language models. Common strategies include splitting by character 
count, sentences, or semantic boundaries.

3. Embedding Model
This component converts text chunks into dense vector representations (embeddings) that 
capture semantic meaning. Popular models include sentence-transformers and OpenAI's 
embedding models.

4. Vector Database
Vector databases (like ChromaDB, Pinecone, or Weaviate) store embeddings and enable 
fast similarity search. They use algorithms like HNSW or FAISS for efficient nearest 
neighbor search.

5. Retriever
The retriever finds the most relevant document chunks for a given query by comparing 
query embeddings with stored document embeddings.

6. Language Model
The LLM generates final answers by conditioning on both the user's question and the 
retrieved context. This grounds the response in actual documents rather than just 
the model's training data.
""",

        "vector_search.txt": """
Understanding Vector Search in RAG

Vector search, also known as semantic search, is fundamental to RAG systems. Unlike 
traditional keyword-based search, vector search understands the meaning and context 
of queries.

How Vector Search Works:
1. Text is converted into numerical vectors (embeddings) using neural networks
2. These vectors capture semantic relationships between words and concepts
3. Similar concepts are positioned close together in the vector space
4. Search involves finding vectors most similar to the query vector

Advantages over keyword search:
- Understands synonyms and related concepts
- Handles typos and variations better
- Captures semantic intent rather than just word matching
- Works across languages with multilingual models

Distance Metrics:
Common metrics for measuring similarity include:
- Cosine similarity: Measures angle between vectors
- Euclidean distance: Straight-line distance
- Dot product: Combines magnitude and direction

Vector search enables RAG systems to retrieve contextually relevant information even 
when exact keywords don't match.
""",

        "rag_challenges.txt": """
Common Challenges in RAG Systems

1. Chunking Strategy
Choosing the right chunk size is crucial. Too small and you lose context; too large 
and you may include irrelevant information. Overlap between chunks helps maintain 
continuity but increases storage requirements.

2. Retrieval Accuracy
Not all retrieved chunks may be relevant. This can lead to:
- Context pollution: Irrelevant information confusing the LLM
- Missing information: Relevant chunks not being retrieved
- Contradictory information: Different sources providing conflicting facts

3. Latency
RAG systems involve multiple steps (embedding, search, generation) which can impact 
response time. Optimization strategies include caching, indexing, and batching.

4. Hallucinations
Even with retrieved context, LLMs may generate incorrect or unsupported information. 
This is particularly problematic when the model has low confidence or when context 
is ambiguous.

5. Context Window Limitations
LLMs have limited context windows. Fitting multiple retrieved chunks plus the query 
and system prompt requires careful management of token budgets.

6. Metadata Management
Tracking document sources, versions, timestamps, and other metadata is essential for 
transparency and debugging but adds complexity.

7. Cost Considerations
Embedding generation, vector storage, and LLM inference all incur costs. Balancing 
performance with cost efficiency is an ongoing challenge.
""",

        "rag_evaluation.txt": """
Evaluating RAG System Performance

Effective evaluation is crucial for improving RAG systems. Key metrics include:

Retrieval Metrics:
- Precision: Proportion of retrieved chunks that are relevant
- Recall: Proportion of relevant chunks that were retrieved
- MRR (Mean Reciprocal Rank): Measures how quickly relevant results appear
- NDCG (Normalized Discounted Cumulative Gain): Rewards relevant results higher in rankings

Generation Metrics:
- Faithfulness: Is the answer grounded in retrieved context?
- Answer Relevancy: Does the answer address the question?
- Answer Correctness: Compared to ground truth (if available)
- Answer Similarity: Semantic similarity to expected answers

End-to-End Metrics:
- Context Precision: Are retrieved chunks relevant to the question?
- Context Recall: Were all relevant chunks retrieved?
- Context Relevancy: Overall quality of retrieved context

Tools for Evaluation:
- RAGAS: Comprehensive evaluation framework
- LlamaIndex: Includes evaluation modules
- DeepEval: Provides various RAG-specific tests
- TruLens: Focuses on tracing and observability

Best Practices:
1. Use a diverse test set covering different query types
2. Include edge cases and challenging questions
3. Combine automated metrics with human evaluation
4. Track metrics over time to measure improvements
5. A/B test different configurations
6. Monitor production performance continuously

Evaluation should be iterative - use insights to tune chunking strategies, 
embedding models, retrieval parameters, and prompts.
""",

    "ai_agents_mcp.txt": """

AI AGENTS AND MCP PROTOCOL – TEST KNOWLEDGE FILE
Version: 1.0
Last Updated: January 2026

------------------------------------------------------------
SECTION 1: INTRODUCTION TO AI AGENTS
------------------------------------------------------------

An AI Agent is a system that can perceive, reason, decide, and act toward achieving a goal.
Core Components of an AI Agent:
1. Memory (short-term and long-term)
2. Tools (APIs, databases, calculators)
3. Planning capability
4. Reasoning engine (LLM)
5. Action execution layer

Example:
If a user asks: "Book a flight from Delhi to Mumbai tomorrow",
The agent:
- Checks flight APIs
- Compares prices
- Selects best option
- Confirms booking

This is different from a chatbot. A chatbot only responds. An agent takes action.

------------------------------------------------------------
SECTION 2: TYPES OF AI AGENTS
------------------------------------------------------------

1. Reactive Agents
   - No memory
   - Respond only to current input
   - Example: Basic FAQ bot

2. Goal-Based Agents
   - Work toward achieving defined objectives
   - Example: Travel booking assistant

3. Autonomous Agents
   - Can plan multi-step workflows
   - Example: AutoGPT

4. Multi-Agent Systems
   - Multiple agents collaborating
   - Example:
       Research Agent
       Summary Agent
       Validation Agent

------------------------------------------------------------
SECTION 3: RAG (RETRIEVAL AUGMENTED GENERATION)
------------------------------------------------------------

RAG is a system where an LLM retrieves relevant information before generating an answer.

RAG Pipeline:
1. Document Ingestion
2. Chunking (e.g., 500 tokens with 50 overlap)
3. Embedding Generation
4. Vector Storage
5. Similarity Search
6. Context Injection
7. Final Answer Generation

Example Query:
"What are the benefits of MCP protocol?"

The system:
- Converts query into embedding
- Finds similar content
- Adds context to prompt
- Generates answer

Key Metrics:
- Precision@K
- Recall@K
- MRR
- Faithfulness Score

------------------------------------------------------------
SECTION 4: MCP (MODEL CONTEXT PROTOCOL)
------------------------------------------------------------

MCP (Model Context Protocol) is a standardized way for AI models to securely access external tools and data sources.

Purpose:
- Provide structured context to LLMs
- Enable secure tool usage
- Improve reliability

MCP Architecture:
1. Client
2. MCP Server
3. Tool Provider
4. Model

Example Workflow:
User asks: "Get my latest sales report"
Model:
- Calls MCP server
- MCP validates request
- Fetches data from sales database
- Returns structured response

Security Benefits:
- Controlled access
- Authentication layers
- Audit logging

------------------------------------------------------------
SECTION 5: VECTOR DATABASE CONCEPTS
------------------------------------------------------------

A vector database stores embeddings (numerical representations of text).

Common Similarity Metrics:
- Cosine Similarity
- Dot Product
- Euclidean Distance

ANN Algorithms:
- HNSW
- IVF
- Flat Index

Example:
If embedding dimension = 768,
Each document chunk becomes a 768-number vector.

------------------------------------------------------------
SECTION 6: ENTERPRISE USE CASE
------------------------------------------------------------

Company: Food Delivery Platform

Problem:
Customer churn increased by 12% in Tier-2 cities.

AI Agent Task:
1. Retrieve complaint tickets
2. Analyze cancellation reasons
3. Identify patterns
4. Suggest improvements

Findings:
- 45% late delivery
- 30% refund delays
- 15% app crashes
- 10% pricing complaints

Suggested Actions:
- Improve logistics routing
- Reduce refund processing time
- Optimize app performance

------------------------------------------------------------
SECTION 7: EDGE CASES FOR TESTING
------------------------------------------------------------

1. What is the embedding dimension mentioned?
2. What percentage of churn was caused by app crashes?
3. Explain difference between reactive and autonomous agents.
4. How does MCP improve security?
5. List all steps in RAG pipeline.
6. What similarity metrics are mentioned?

------------------------------------------------------------
SECTION 8: ADVANCED CONCEPTS
------------------------------------------------------------

Hybrid Search:
Combination of keyword search + vector search.

Re-ranking:
Using a cross-encoder to re-score retrieved documents.

Context Window Limitation:
LLMs have token limits (e.g., 8k, 32k, 128k).

Chunking Strategies:
- Fixed size
- Semantic chunking
- Recursive splitting

------------------------------------------------------------
END OF FILE
------------------------------------------------------------

""",










        "oracle_rag.txt": """
Oracle's RAG Implementation

Oracle has integrated RAG capabilities across multiple products:

Oracle Database 23ai with Select AI:
Oracle's flagship RAG offering automates the RAG process. Select AI creates and populates 
vector stores, performs semantic similarity searches, and augments prompts with relevant 
content from enterprise data. The system uses AI Vector Search built directly into the 
database.

Key Features:
- Native vector data type in SQL
- Hybrid search combining vectors with relational, graph, spatial, and JSON data
- DBMS_VECTOR_CHAIN PL/SQL package for RAG operations
- Security: Data never leaves the Oracle Database
- Integration with multiple LLM providers

Oracle AI Vector Search:
Supports efficient similarity search using indexes like HNSW and IVF. Allows combining 
vector search with traditional SQL queries for powerful enterprise RAG applications.

OCI Generative AI Agents:
These agents augment LLM knowledge without retraining. They use hybrid search combining 
lexical and semantic approaches. Oracle Integration (OIC) helps automate the end-to-end 
RAG process with low-code/no-code platforms.

Benefits for Enterprise:
- Integrated with existing Oracle infrastructure
- Enterprise-grade security and governance
- High performance and scalability
- Multi-cloud support (Azure, AWS, GCP)
- Comprehensive monitoring and observability

Oracle's approach embeds RAG directly into database infrastructure rather than requiring 
external tools, providing better security and performance for enterprise applications.
"""
    }
    
    # Write all documents
    for filename, content in documents.items():
        filepath = os.path.join("./data", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"Created: {filepath}")
    
    print(f"\n✓ Created {len(documents)} sample documents in ./data/")
    print("\nThese documents cover:")
    print("  - RAG fundamentals and introduction")
    print("  - Core RAG components")
    print("  - Vector search concepts")
    print("  - Common challenges")
    print("  - Evaluation methods")
    print("  - Oracle RAG implementation")
    print("\nYou can now run simple_rag.py to build your RAG system!")


if __name__ == "__main__":
    create_sample_documents()
