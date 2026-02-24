# DAY 2 QUICK REFERENCE CARD
## Document Loading & Chunking

---

## ⏰ SCHEDULE

| Session | Time | Focus |
|---------|------|-------|
| Session 1 | 45 min | load_documents() |
| Session 2 | 45 min | split_documents() |
| Session 3 | 30 min | 3 Experiments |
| Session 4 | 30 min | Document & Reflect |

---

## 🔑 KEY CONCEPTS

**Document Object:**
```
page_content = "actual text here"
metadata = {'source': 'filepath'}
```

**DirectoryLoader:**
```python
DirectoryLoader(
    path,           # WHERE to look
    glob="**/*.txt",# WHAT to find
    loader_cls=...  # HOW to load
)
```

**glob patterns:**
- `**` = all directories
- `*` = any filename
- `.txt` = file extension

**RecursiveCharacterTextSplitter:**
```python
RecursiveCharacterTextSplitter(
    chunk_size=500,    # ~500 chars per chunk
    chunk_overlap=50,  # 50 chars overlap
    length_function=len
)
```

---

## 📊 CHUNKING QUICK FACTS

| Parameter | Too Small | Just Right | Too Large |
|-----------|-----------|------------|-----------|
| chunk_size | Loses context | 300-800 | Too much noise |
| chunk_overlap | Splits sentences | 10-20% of size | Wastes storage |

**Documents → Chunks:**
```
7 documents → 50+ chunks
(1 file → many small pieces)
```

**Metadata preserved:** YES ✅
(Every chunk knows its source file!)

---

## 🧪 EXPERIMENTS

```bash
# Run all experiments
cd C:\RAG-Learning
python experiments/day2/explore_documents.py
python experiments/day2/chunking_comparison.py
python experiments/day2/metadata_test.py
```

---

## 🎯 QUESTIONS TO ANSWER

Without looking:
1. What does glob="**/*.txt" match?
2. What is a Document object?
3. Why do we use chunk_overlap?
4. How many chunks from 7 documents?
5. Is metadata preserved after chunking?

---

## 🔄 PIPELINE PROGRESS

```
Day 2 covers these steps:
↓
Files → [TextLoader] → Document objects
         ↓
Documents → [TextSplitter] → Chunks
         ↓
Chunks → [Day 3: Embeddings] → Vectors
         ↓
Vectors → [Day 4: ChromaDB] → Vector Store
```

---

## 💡 QA CONNECTIONS

| RAG Concept | QA Equivalent |
|-------------|---------------|
| glob pattern | Test filter |
| chunk_size | Test granularity |
| metadata | Traceability |
| silent_errors | Fault tolerance |
| overlap | Boundary testing |

---

## ⚠️ WATCH OUT FOR

1. Wrong directory path → FileNotFoundError
2. No .txt files → Empty documents list
3. chunk_size > document size → 1 chunk
4. chunk_overlap > chunk_size → Error!
5. Empty file → Empty document

---

## ✅ DELIVERABLES

- [ ] explore_documents.py (run it!)
- [ ] chunking_comparison.py (run it!)
- [ ] metadata_test.py (run it!)
- [ ] chunking_results.csv (check it!)
- [ ] day2-notes.md (fill it!)

---

**Your QA background = perfect for systematic experiments! 💪**
