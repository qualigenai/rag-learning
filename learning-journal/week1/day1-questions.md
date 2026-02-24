SESSION 1: Understanding Imports

#### **Import 1: `import os`**

Q1: What is this library?
A: [This library is used by Python to interact with operating system like File system objects
    Files,Folders,Environment Variables,]

Q2: Why do we need it in RAG?
A: [In RAG we need to interact with file system like accessing files and folders paths , accessing environment variables]

Q3: What specific functions will we use?
A: [os.path.exists, os.makedirs]

Q4: Could we build RAG without it?
A: [No, we can't build RAG without Import OS as it is compulsory required]

#### **Import 2: `from typing import List`**

Q1: What is this library?
A: [I want to clearly tell Python that something will be a list of specific type items.]
    
Q2: Why do we need it in RAG?
A: [we need this library to work with List of items like list of documents, list of strings, list of tokens]]

Q3: What specific functions will we use?
A: [  def get_names() -> List[str]:
    return ["Ravi", "Sita"]
    This means: This function returns a list of strings.]

Q4: Could we build RAG without it?
A: [No- Python doesn't enforce types at runtime. But it's a best practice]

## **Import 3: `from langchain_community.embeddings import HuggingFaceEmbeddings`**
```
Q1: What is this library?
A: [HuggingFaceEmbeddings is a wrapper commonly from LangChain can be used for embedding models from Hugging Face  ]

Q2: Why do we need it in RAG?
A: [To load the embedding models which is used to convert the text into vectors which is indirectly useful for symantic search]

Q3: What specific functions will we use?
A: [ self.embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2" )
]

When this code executes:
It downloads the model (first time only)
Loads it into memory
Prepares it to convert text → vector

text = "What is artificial intelligence?"
vector = self.embeddings.embed_query(text)

Now vector becomes something like: [0.0123, -0.2345, 0.9876, ...]  # 384 numbers
That vector represents the meaning of the sentence.

Q4: Could we build RAG without it?
A: [No, we can't this core of the RAG]
Embeddings are the CORE of RAG. Without them, no semantic search possible!

#### **Import 4: `from langchain_community.vectorstores import Chroma`**
```
Q1: What is this library?
A: [This library is used to store the vectors in Chroma database]

Q2: Why do we need it in RAG?
A: [To store the data in vector stores and for semantic search]

Q3: What specific functions will we use?
A: [vectorstores= Chroma.from_documents(
documents=chunks,
embedding=self.embeddings,
persist_directory=persist_dir

)]

Q4: Could we build RAG without it?
A: [No, this is a core component to store the data]

# Import 5: from langchain.text_splitter import RecursiveCharacterTextSplitter

Q1: What is this library?
A: [This library is used to break big text into smaller chunks properly.]

Q2: Why do we need it in RAG?
A: [
 Split by paragraphs first
If too big → split by sentences
f still too big → split by words
If needed → split by characters
That’s why it's called recursive.
]

Q3: What specific functions will we use?
A: [
ext_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
]

# Import 6: from from langchain_community.llms import Ollama

Q1: What is this library?
A: [LLM wrapper from LangChain’s community package so you can run a local Large Language Model.]

Q2: Why do we need it in RAG?
A: [If we want to work with offline mode of LLM we need this]

Q3: What specific functions will we use?
A: [What ever the LLM functions are there we can use all of them ]

Q4: Could we build RAG without it?
A: [If I am treating Ollama as my local LLM and wanted to build RAG, without this we can't build RAG Because RAG must have a generation model.]