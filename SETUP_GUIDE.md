# RAG Evaluation Learning Environment - Windows Setup Guide

## Overview
This guide will help you set up a complete, free RAG evaluation environment on Windows for learning purposes.

## What You'll Build
- A working RAG system using free tools
- Multiple evaluation frameworks
- Sample datasets and test cases
- Transferable skills for Oracle (or any) RAG evaluation

## Prerequisites
- Windows 10/11
- At least 8GB RAM (16GB recommended)
- 10GB free disk space
- Internet connection

---

## Step 1: Install Python 3.10+

1. Download Python from: https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
```bash
python --version
```

---

## Step 2: Install Git (Optional but Recommended)

1. Download from: https://git-scm.com/download/win
2. Use default settings during installation
3. Verify:
```bash
git --version
```

---

## Step 3: Create Project Directory

Open Command Prompt or PowerShell:
```bash
mkdir C:\RAG-Learning
cd C:\RAG-Learning
```

---

## Step 4: Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your prompt now
```

---

## Step 5: Install Required Packages

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install core RAG components
pip install langchain langchain-community langchain-openai
pip install chromadb
pip install sentence-transformers
pip install ragas
pip install datasets
pip install pandas numpy

# Install optional but useful packages
pip install jupyter notebook
pip install python-dotenv
```

---

## Step 6: Choose Your LLM Option

### Option A: Ollama (Recommended - Completely Free, Runs Locally)

1. Download Ollama for Windows: https://ollama.ai/download
2. Install and run
3. Open new Command Prompt and pull a model:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Option B: OpenAI Free Tier (Requires API Key)

1. Sign up at: https://platform.openai.com/
2. Get API key from dashboard
3. Create `.env` file in project directory:
```
OPENAI_API_KEY=your_api_key_here
```

**For Learning: I recommend Ollama (Option A) - it's 100% free with no limits**

---

## Step 7: Verify Installation

Run this test script (save as `test_install.py`):

```python
import langchain
import chromadb
import ragas
from sentence_transformers import SentenceTransformer

print("✓ LangChain installed")
print("✓ ChromaDB installed")
print("✓ RAGAS installed")
print("✓ SentenceTransformers installed")
print("\nAll dependencies installed successfully!")
```

Run it:
```bash
python test_install.py
```

---

## Step 8: Project Structure

Your project should look like this:
```
C:\RAG-Learning\
├── venv\                    # Virtual environment
├── data\                    # Sample documents
├── evaluation\              # Evaluation scripts
├── simple_rag.py           # Basic RAG implementation
├── evaluate_rag.py         # Evaluation pipeline
├── requirements.txt        # Dependencies
└── .env                    # Environment variables (if using OpenAI)
```

---

## What's Next?

After setup, you'll learn:
1. Build a simple RAG system from scratch
2. Load and process documents
3. Create vector embeddings
4. Query the RAG system
5. Evaluate with RAGAS metrics
6. Understand what makes RAG systems good/bad

---

## Troubleshooting

### "Python not recognized"
- Reinstall Python and check "Add to PATH"
- Or manually add to PATH: Control Panel → System → Environment Variables

### "pip install fails"
- Run Command Prompt as Administrator
- Try: `python -m pip install package_name`

### "Module not found" errors
- Make sure virtual environment is activated: `venv\Scripts\activate`
- Reinstall the package

### Ollama not starting
- Check Windows Defender/Firewall
- Run Ollama as Administrator

---

## Resources for Learning

- LangChain Docs: https://python.langchain.com/
- ChromaDB Docs: https://docs.trychroma.com/
- RAGAS Docs: https://docs.ragas.io/
- Ollama Models: https://ollama.ai/library

---

Ready to proceed? Check the other files I've created for you:
- `simple_rag.py` - Build your first RAG system
- `evaluate_rag.py` - Evaluate RAG performance
- `sample_data_creator.py` - Generate test documents
