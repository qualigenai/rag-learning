# Import 1:Import OS

This library is used by Python to interact with operating system.
This line tells Python : I want to use operating system features in my program
like FileSystem.

Oerating system is:

🪟 Windows
🍎 macOS
🐧 Linux

It manages :
Files
Folders
Environment variables
Paths
System Settings.

Scenarios: To Create folders, get folders, Check File exists, to read environemnt variables.

🔹 Real-Life Analogy

Imagine:
Python = A person
OS = The house

without os, the person cannot:
Open doors (folders)
Check rooms (directories)
See house settings (environment variables)
import os = Giving that person house access keys

In RAG we need them 
os.path.exists
os.makedirs("./data")

# Import 2: from typing import List
I want to clearly tell Python that something will be a list of specific type items.

List comes from typing module
It helps us specify:
List of strings → List[str]
List of integers → List[int]
List of floats → List[float]

from typing import List
def get_names() -> List[str]:
    return ["Ravi", "Sita"]
This means: This function returns a list of strings.

Why it is import in AI/RAG:

Here we deal with 
* List of documents
* List of embeddings
* List of strings
* List of tokens

# Import 3: from langchain_community.embeddings import HuggingFaceEmbeddings
HuggingFaceEmbeddings is wrapper (commonly from LangChain) that allows you to use
embedding models from Hugging Face.

It converts text into vector embeddings (numerical representation).
These Embeddings are used in RAG, Semantic Search, Vector databases,Similarity Comparison.

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

model_name="sentence-transformers/all-MiniLM-L6-v2"

This referees a model hosted on Hugging Face:
Organization: sentence-transformers
Model: all-MiniLM-L6-v2

all-MiniLM-L6-v2 : Produces 384-dimensional embeddings, Lightweight, Good for symanticsearch

It is part of the Sentence Transformers family built on BERT-style transformers.

Key Takeaway: Embeddings are the CORE of RAG. Without them, no semantic search possible!

3️⃣ What happens when this line runs?

When this code executes:

It downloads the model (first time only)

Loads it into memory

Prepares it to convert text → vector

Example:

text = "What is artificial intelligence?"
vector = self.embeddings.embed_query(text)


Now vector becomes something like:

[0.0123, -0.2345, 0.9876, ...]  # 384 numbers


That vector represents the meaning of the sentence.

4️⃣ Why is this important in RAG?
In a RAG pipeline:
Convert documents → embeddings
Store them in vector DB
Convert user query → embedding
Compare similarity
Retrieve most relevant chunks
This line is setting up the embedding engine.

In Simple Words, This line means: “Load a small, fast Hugging Face model that converts text into numerical vectors so I can do semantic search or RAG.”



# Import 4: from langchain_community.vectorstores import Chroma
I want use Chroma database to store my vectors and search embeddings. 

Vector Stores: A Vector store is special database that stores
Text converted into numbers(embeddings)
Allows similarity search

Chroma is a vector database can store, documents and their embedings and meta data

# 1. Convert text to embeddings
# 2. Store in Chroma
# 3. User asks question
# 4. Convert question to embedding
# 5. Chroma finds similar documents
# 6. Send them to LLM

Example

vectorstore= Chroma.from_documents (
documents=chunks,
embeddings=self.embeddings
persist_directory=self.persist_dir
)

# Import 5: from langchain.text_splitter import RecursiveCharacterTextSplitter

This means: 
“Import a smart tool that breaks big text into smaller chunks properly.”

What is RecursiveCharacterTextSplitter?
It is a smart text cutter ✂️ But not a dumb cutter.
It tries to:
* Split by paragraphs first
* If too big → split by sentences
* If still too big → split by words
* If needed → split by characters
* That’s why it's called recursive.
t
* ext_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
What does this mean?
chunk_size=500 → each piece max 500 characters
chunk_overlap=50 → next chunk repeats last 50 characters
* Full RAG flow:
1️⃣ Load document
2️⃣ Split using RecursiveCharacterTextSplitter
3️⃣ Convert chunks → embeddings
4️⃣ Store in Chroma
5️⃣ Query → retrieve similar chunks
6️⃣ Send to LLM

# Import 6: from langchain.chains import RetrievalQA

What is RetrievalQA :  This is a ready made chain from langchain library it helps:

1️⃣ Retrieve relevant document
2️⃣ send them to LLM
3️⃣ Generate an answer based on these documents.

So instead of LLM gussing it's answers from training data, it answers using own data

# Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple chain type
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Return top 3 relevant chunks
            ),
            return_source_documents=True  # Return sources for transparency
        )

Since you were already working with:
HuggingFaceEmbeddings
Chroma
TextLoader
RecursiveCharacterTextSplitter

👉 That means you are building a RAG pipeline, and RetrievalQA is the final step that connects everything.
RetrievalQA = Prebuilt RAG chain
It reduces manual coding
Good for production prototypes
Used widely in enterprise AI apps

# Import 7: from langchain_community.llms import Ollama
You are importing the Ollama LLM wrapper from LangChain’s community package so you can run a local Large Language Model.
Ollama is a tool that lets you:
Run LLMs locally on your computer
No API key needed
No internet required (after model download)
Works like a local AI server

OpenAI → Cloud AI (needs internet + API key)
Ollama → Local AI (runs inside your laptop)
