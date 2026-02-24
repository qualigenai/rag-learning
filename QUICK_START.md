# Quick Start Guide - RAG Evaluation Learning

## Complete Setup in 10 Minutes

Follow these steps exactly in order:

---

## Step 1: Download and Install Prerequisites (5 minutes)

### A. Install Python
1. Go to: https://www.python.org/downloads/
2. Download Python 3.10 or newer
3. **CRITICAL**: Check "Add Python to PATH" during installation
4. Click "Install Now"

### B. Install Ollama (Recommended)
1. Go to: https://ollama.ai/download
2. Download Ollama for Windows
3. Run installer
4. Open Command Prompt and run:
```bash
ollama pull llama3.2
```

This downloads a free local LLM (about 2GB). Wait for download to complete.

---

## Step 2: Set Up Project (2 minutes)

Open Command Prompt or PowerShell:

```bash
# Create project directory
mkdir C:\RAG-Learning
cd C:\RAG-Learning

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your prompt
```

---

## Step 3: Download Project Files (1 minute)

Copy all the files I created into `C:\RAG-Learning\`:
- SETUP_GUIDE.md
- simple_rag.py
- evaluate_rag.py
- sample_data_creator.py
- requirements.txt
- QUICK_START.md (this file)

---

## Step 4: Install Dependencies (2 minutes)

In the same Command Prompt (with venv activated):

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will take 1-2 minutes. Wait for it to complete.

---

## Step 5: Create Sample Data

```bash
python sample_data_creator.py
```

This creates 6 sample documents about RAG systems in the `./data` folder.

---

## Step 6: Build Your RAG System

```bash
python simple_rag.py
```

This will:
1. Load the sample documents
2. Split them into chunks
3. Create embeddings
4. Build a vector database (ChromaDB)
5. Set up the QA chain
6. Let you ask questions interactively

Try asking:
- "What is RAG?"
- "How does vector search work?"
- "What are RAG challenges?"

Type 'quit' when done.

---

## Step 7: Evaluate Your RAG System

```bash
python evaluate_rag.py
```

This will:
1. Load your RAG system
2. Run 5 test questions
3. Calculate RAGAS metrics (faithfulness, relevancy, etc.)
4. Display results
5. Save to `evaluation_results.csv`

---

## What You've Just Built

Congratulations! You now have:

✓ A working RAG system using free tools
✓ Sample documents to query
✓ Evaluation pipeline with RAGAS metrics
✓ Understanding of how RAG works

---

## Next Steps

### 1. Add Your Own Documents
Put any .txt files in the `./data` folder, then rebuild:
```bash
# Delete old database
rmdir /s chroma_db

# Rebuild with new docs
python simple_rag.py
```

### 2. Run Custom Evaluation
```bash
python evaluate_rag.py custom
```
This lets you enter your own test questions.

### 3. Experiment with Parameters

Edit `simple_rag.py` and try different settings:
- Change chunk_size (line 51): Try 300, 500, 1000
- Change chunk_overlap (line 52): Try 0, 50, 100
- Change k value (line 91): Try 2, 5, 10 (number of chunks retrieved)

Then rebuild and re-evaluate to see how metrics change!

### 4. Try Different Models

In `simple_rag.py`, change the model (line 32):
```python
self.llm = Ollama(model="llama3.2")  # Try: "mistral", "gemma", etc.
```

First pull the new model:
```bash
ollama pull mistral
```

### 5. Visualize Results

```bash
python
>>> import pandas as pd
>>> df = pd.read_csv('evaluation_results.csv')
>>> print(df)
>>> df[['faithfulness', 'answer_relevancy', 'context_precision']].mean()
```

---

## Troubleshooting

### "Python not recognized"
- Reinstall Python and check "Add to PATH"
- Restart Command Prompt

### "pip install fails"
- Run Command Prompt as Administrator
- Try: `python -m pip install <package>`

### "Ollama connection failed"
- Make sure Ollama is running (check system tray)
- Restart Ollama application

### "No documents found"
- Run `sample_data_creator.py` first
- Check that `./data` folder exists with .txt files

### Module import errors
- Make sure venv is activated: `venv\Scripts\activate`
- Reinstall: `pip install -r requirements.txt`

---

## Learning Resources

- **LangChain**: https://python.langchain.com/docs/
- **ChromaDB**: https://docs.trychroma.com/
- **RAGAS**: https://docs.ragas.io/
- **Ollama Models**: https://ollama.ai/library

---

## Daily Practice Routine

**Day 1**: Build the system (steps 1-7 above)
**Day 2**: Add your own documents and rebuild
**Day 3**: Experiment with chunk sizes and evaluate
**Day 4**: Try different embedding models
**Day 5**: Test different LLMs and compare results
**Day 6**: Build a custom evaluation dataset
**Day 7**: Read documentation and understand metrics deeply

By the end of week 1, you'll understand RAG better than most!

---

## How This Transfers to Oracle

Everything you learn here applies directly to Oracle RAG:

| Free Stack | Oracle Equivalent |
|------------|------------------|
| ChromaDB | Oracle AI Vector Search |
| LangChain | Oracle Integration (OIC) |
| Ollama | OCI Generative AI |
| RAGAS | Same evaluation framework! |

The concepts, metrics, and optimization strategies are identical. You're learning production-grade RAG skills!

---

## Questions?

If you get stuck:
1. Check SETUP_GUIDE.md for detailed explanations
2. Read the code comments in the Python files
3. Google the specific error message
4. The error messages usually tell you exactly what's wrong!

---

Ready to become a RAG expert? Start with Step 1!
