"""
Simple RAG System - Learning Implementation
This is a basic RAG system using free tools that demonstrates core concepts.
"""

import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

class SimpleRAG:
    """
    A simple RAG system for learning purposes.
    
    Components:
    1. Document Loader - Loads text files
    2. Text Splitter - Breaks documents into chunks
    3. Embeddings - Converts text to vectors
    4. Vector Store - Stores and searches embeddings (ChromaDB)
    5. LLM - Generates answers (Ollama)
    6. Retrieval Chain - Orchestrates everything
    """
    
    def __init__(self, data_dir: str = "./data", persist_dir: str = "./chroma_db"):
        """
        Initialize the RAG system.
        
        Args:
            data_dir: Directory containing your documents
            persist_dir: Directory to store vector database
        """
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        # Initialize embeddings model (runs locally, free)
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM (Ollama - runs locally, free)
        print("Connecting to Ollama...")
        self.llm = Ollama(model="llama3.2")
        
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self):
        """Load documents from the data directory."""
        print(f"Loading documents from {self.data_dir}...")
        
        # Load all .txt files from data directory
        loader = DirectoryLoader(
            self.data_dir, 
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents):
        """Split documents into smaller chunks."""
        print("Splitting documents into chunks...")
        
        # Split documents - important for good retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Characters per chunk
            chunk_overlap=50,  # Overlap between chunks
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create vector database from document chunks."""
        print("Creating vector database...")
        
        # Create ChromaDB vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        print("Vector database created!")
        
    def load_existing_vectorstore(self):
        """Load existing vector database if available."""
        print("Loading existing vector database...")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        print("Vector database loaded!")
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        print("Setting up QA chain...")
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple chain type
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Return top 3 relevant chunks
            ),
            return_source_documents=True  # Return sources for transparency
        )
        
        print("QA chain ready!")
    
    def query(self, question: str):
        """
        Query the RAG system.
        
        Args:
            question: Your question
            
        Returns:
            dict with 'result' and 'source_documents'
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Run setup_qa_chain() first.")
        
        print(f"\nQuestion: {question}")
        print("Searching and generating answer...")
        
        response = self.qa_chain.invoke({"query": question})
        
        return response
    
    def build_from_documents(self):
        """Complete pipeline: load docs → create vectors → setup QA."""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.setup_qa_chain()
        print("\n✓ RAG system ready!")
    
    def load_and_setup(self):
        """Quick setup using existing vector database."""
        self.load_existing_vectorstore()
        self.setup_qa_chain()
        print("\n✓ RAG system ready!")


# Example Usage
if __name__ == "__main__":
    # Create sample data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("Created ./data directory - add your .txt files there!")
        print("Run sample_data_creator.py to generate test documents.")
        exit()
    
    # Initialize RAG system
    rag = SimpleRAG(data_dir="./data", persist_dir="./chroma_db")
    
    # Build from scratch (first time)
    # Or load existing database (subsequent times)
    if not os.path.exists("./chroma_db"):
        print("Building RAG system from documents...")
        rag.build_from_documents()
    else:
        print("Loading existing RAG system...")
        rag.load_and_setup()
    
    # Test queries
    test_questions = [
        "What are the main topics in these documents?",
        "Summarize the key points",
    ]
    
    for question in test_questions:
        response = rag.query(question)
        print(f"\nAnswer: {response['result']}")
        print(f"\nSources used: {len(response['source_documents'])} chunks")
        
    print("\n" + "="*50)
    print("Try your own questions!")
    print("="*50)
    
    # Interactive mode
    while True:
        user_question = input("\nYour question (or 'quit' to exit): ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
            
        response = rag.query(user_question)
        print(f"\nAnswer: {response['result']}")
