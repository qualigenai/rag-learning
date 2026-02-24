# RAG Evaluation Learning Environment

A complete, **100% free** learning environment for understanding and evaluating Retrieval-Augmented Generation (RAG) systems.

## 🎯 What You'll Learn

- Build a production-quality RAG system from scratch
- Understand core RAG components (embeddings, vector search, retrieval)
- Evaluate RAG performance using industry-standard metrics
- Optimize RAG systems through experimentation
- **Skills that transfer directly to Oracle RAG and other enterprise systems**

## 🆓 Completely Free

- No cloud credits needed
- No API keys required (optional)
- Runs entirely on your local machine
- Uses open-source tools only

## 📁 Project Files

```
RAG-Learning/
├── README.md                    # This file - overview
├── QUICK_START.md              # Fast setup guide (start here!)
├── SETUP_GUIDE.md              # Detailed setup instructions
├── requirements.txt            # Python dependencies
├── sample_data_creator.py      # Generate test documents
├── simple_rag.py               # Core RAG implementation
├── evaluate_rag.py             # Evaluation pipeline
├── data/                       # Sample documents (auto-created)
├── chroma_db/                  # Vector database (auto-created)
└── evaluation_results.csv      # Results (auto-created)
```

## 🚀 Quick Start (10 Minutes)

1. **Install Python 3.10+** and **Ollama**
2. **Create project folder** and virtual environment
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Create sample data**: `python sample_data_creator.py`
5. **Build RAG system**: `python simple_rag.py`
6. **Evaluate performance**: `python evaluate_rag.py`

See **QUICK_START.md** for detailed commands!

## 🏗️ What You're Building

### 1. Simple RAG System (`simple_rag.py`)
A complete RAG implementation with:
- Document loading and processing
- Text splitting and chunking
- Vector embeddings (sentence-transformers)
- Vector database (ChromaDB)
- Question answering with Ollama (local LLM)

### 2. Evaluation Pipeline (`evaluate_rag.py`)
Professional RAG evaluation using RAGAS metrics:
- **Faithfulness**: Are answers grounded in context?
- **Answer Relevancy**: Do answers address questions?
- **Context Precision**: Are retrieved chunks relevant?
- **Context Recall**: Was relevant info retrieved?

### 3. Sample Dataset (`sample_data_creator.py`)
6 documents covering:
- RAG fundamentals
- System components
- Vector search
- Common challenges
- Evaluation methods
- Oracle RAG implementation

## 📊 Evaluation Metrics Explained

### Faithfulness (0-1)
- **Measures**: Is the answer supported by retrieved context?
- **High score**: Answer is well-grounded in facts
- **Low score**: Answer contains hallucinations

### Answer Relevancy (0-1)
- **Measures**: Does answer address the question?
- **High score**: Direct, on-topic response
- **Low score**: Off-topic or incomplete

### Context Precision (0-1)
- **Measures**: Are retrieved chunks relevant?
- **High score**: Clean, relevant retrievals
- **Low score**: Too much noise in results

### Context Recall (0-1)
- **Measures**: Was all relevant info retrieved?
- **High score**: Nothing important missed
- **Low score**: Key information not found

## 🔧 Experimentation Ideas

Try changing these parameters and re-evaluating:

1. **Chunk Size** (`simple_rag.py`, line 51)
   - Try: 300, 500, 1000
   - Smaller = more precise, larger = more context

2. **Chunk Overlap** (`simple_rag.py`, line 52)
   - Try: 0, 50, 100
   - Overlap helps maintain continuity

3. **Retrieval Count (k)** (`simple_rag.py`, line 91)
   - Try: 2, 5, 10
   - More chunks = more context but potential noise

4. **Different LLMs**
   - Try: llama3.2, mistral, gemma
   - Each has different strengths

5. **Different Embeddings**
   - Try: all-MiniLM-L6-v2, all-mpnet-base-v2
   - Better embeddings = better retrieval

## 🏢 How This Relates to Oracle RAG

| Concept You Learn | Oracle Equivalent |
|------------------|-------------------|
| ChromaDB | Oracle AI Vector Search |
| HuggingFace Embeddings | Oracle Select AI |
| LangChain | Oracle Integration (OIC) |
| Ollama | OCI Generative AI |
| RAGAS Evaluation | Same framework! |
| Vector Search | DBMS_VECTOR_CHAIN |

**The fundamentals are identical!** Everything you learn here transfers directly to enterprise RAG systems.

## 🛠️ Tech Stack

- **Python**: 3.10+
- **LangChain**: RAG orchestration
- **ChromaDB**: Vector database
- **Ollama**: Local LLM (llama3.2)
- **Sentence-Transformers**: Embeddings
- **RAGAS**: Evaluation framework
- **Pandas**: Data analysis

## 📈 Success Metrics

After completing this learning environment, you should be able to:

✅ Explain how RAG works to a technical audience
✅ Build a RAG system from scratch
✅ Evaluate RAG performance scientifically
✅ Optimize retrieval and generation
✅ Debug common RAG issues
✅ Apply learnings to enterprise systems like Oracle

## 🐛 Troubleshooting

### Common Issues

**"Python not found"**
- Reinstall Python with "Add to PATH" checked

**"Module not found"**
- Activate virtual environment: `venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**"Ollama connection failed"**
- Start Ollama application
- Pull model: `ollama pull llama3.2`

**"No documents found"**
- Run: `python sample_data_creator.py`

See **SETUP_GUIDE.md** for detailed troubleshooting.

## 📚 Additional Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [RAGAS Docs](https://docs.ragas.io/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai/library)

### Learning Materials
- LangChain tutorials on YouTube
- RAGAS evaluation guides
- Vector database concepts
- LLM prompting techniques

### Oracle-Specific
- Oracle AI Vector Search documentation
- Oracle Select AI guides
- OCI Generative AI docs

## 🤝 Contributing Ideas

Ways to extend this project:
- Add more evaluation metrics
- Support PDF/Word documents
- Implement advanced retrieval (hybrid search)
- Add visualization dashboards
- Create comparison tools
- Build web interface

## 📄 License

This is a learning project - use it however you want!

## 🙋 Questions?

1. Read the code comments (they're detailed!)
2. Check SETUP_GUIDE.md
3. Review QUICK_START.md
4. Search the error message online
5. Experiment and learn by doing!

---

## 🎉 Ready to Start?

1. Open **QUICK_START.md**
2. Follow the 7 steps
3. Start building!

You're about to learn valuable, transferable RAG skills that are in high demand. Enjoy the journey! 🚀
