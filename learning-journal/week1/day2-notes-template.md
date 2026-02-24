# Week 1, Day 2 - Active Learning Notes
## Document Loading & Chunking

**Date:** ___________
**Start Time:** ___________
**End Time:** ___________

---

## SESSION 1: load_documents() Method

### Method Signature
```python
def load_documents(self):
```

**Q: Why no extra parameters?**
Your answer: ___________

**Q: What does this return?**
Your answer: ___________

---

### DirectoryLoader Analysis

```python
loader = DirectoryLoader(
    self.data_dir,       # Where to look
    glob="**/*.txt",     # What to find
    loader_cls=TextLoader  # How to load
)
```

**Explain glob="**/*.txt" in your own words:**
Your answer: ___________

**What files would this MISS?**
Your answer: ___________

**What other loader_cls options exist?**
1. ___________
2. ___________
3. ___________

---

### Document Object Structure

After loading, a Document looks like:
```
page_content = "___________"
metadata = {___________}
```

**Q: Why is metadata important in RAG?**
Your answer: ___________

**Q: What happens if we lost metadata?**
Your answer: ___________

---

### QA Test Cases for load_documents():

Think like a QA engineer - what should you test?

| Test Case | Input | Expected | Actual |
|-----------|-------|----------|--------|
| Normal load | 7 .txt files | 7 Documents | |
| Empty directory | 0 files | 0 Documents | |
| Wrong directory | None | Error | |
| Mixed files | .txt + .pdf | Only .txt | |
| Empty .txt file | empty file | ? | |

Fill in "Actual" column from your experiments!

---

## SESSION 2: split_documents() Method

### The 3 Parameters - YOUR Understanding

**chunk_size=500:**
```
In my own words: ___________

Visual example:
[Draw a document being split into chunks]

Too small means: ___________
Too large means: ___________
500 is good because: ___________
```

**chunk_overlap=50:**
```
In my own words: ___________

Visual example:
Chunk 1: "...last 50 chars..."
Chunk 2: "last 50 chars...next content..."

Without overlap, problem is: ___________
With overlap, we solve: ___________
```

**length_function=len:**
```
In my own words: ___________
Alternative would be: ___________
Difference is: ___________
```

---

### The Recursive Strategy - MY Understanding

Priority order (fill in):
1. First splits at: ___________
2. Then splits at: ___________
3. Then splits at: ___________
4. Last resort: ___________

Why "recursive"?
Your answer: ___________

Why does order matter?
Your answer: ___________

---

### Documents vs Chunks

| | Documents | Chunks |
|--|-----------|--------|
| Count | 7 | ? |
| Size | ~2000 chars each | ~500 chars each |
| Metadata | Has source | _____ |
| Created by | load_documents() | _____ |
| Purpose | _____ | _____ |

---

### QA Test Cases for split_documents():

| Test Case | chunk_size | Expected Chunks | Actual |
|-----------|-----------|----------------|--------|
| Small chunks | 200 | Many | |
| Medium chunks | 500 | Normal | |
| Large chunks | 2000 | Few | |
| No overlap | 500/0 | ? | |
| Big overlap | 500/200 | ? | |

---

## SESSION 3: Experiment Results

### Experiment 1 - Document Objects

**Number of documents loaded:** ___
**Average document size:** ___ chars
**Smallest document:** ___ chars
**Largest document:** ___ chars

**Most interesting metadata field:**
___________

**Error with bad directory:**
Error type: ___________
What this tells me: ___________

---

### Experiment 2 - Chunking Comparison

| Configuration | Chunks Created | Avg Size | My Notes |
|--------------|---------------|----------|----------|
| Small/No-Overlap (200/0) | | | |
| Small/Overlap (200/50) | | | |
| Medium/No-Overlap (500/0) | | | |
| Medium/Overlap (500/50) | | | |
| Large/Overlap (1000/100) | | | |
| XLarge/Overlap (2000/200) | | | |

**Best configuration for RAG (my opinion):** ___________
**Reason:** ___________

**Overlap demonstration:**
End of Chunk 1: "...___________"
Start of Chunk 2: "___________..."
Overlap visible: Yes / No

---

### Experiment 3 - Metadata Preservation

**All chunks have metadata:** Yes / No
**Chunks per source file:**
- rag_introduction.txt: ___ chunks
- rag_components.txt: ___ chunks
- vector_search.txt: ___ chunks
- rag_challenges.txt: ___ chunks
- rag_evaluation.txt: ___ chunks
- oracle_rag.txt: ___ chunks
- ai_agents_mcp.txt: ___ chunks

**Total chunks:** ___

**Test result:** PASSED / FAILED
**What I learned:** ___________

---

## The Complete Pipeline So Far

Draw the flow in your own style:

```
Start: 7 files on disk
    ↓
Step 1 [load_documents()]:
___________
    ↓
Step 2 [split_documents()]:
___________
    ↓
Step 3 [Tomorrow - create_vectorstore()]:
___________
    ↓
End goal:
___________
```

---

## QA Connections I Made Today

1. **glob pattern = ___________**
   (like a test filter/selection pattern)

2. **chunk_size = ___________**
   (like test case granularity)

3. **metadata = ___________**
   (like test traceability)

4. **silent_errors = ___________**
   (like test fault tolerance)

5. **My own connection:**
   ___________

---

## Questions I Still Have

1. ___________
2. ___________
3. ___________

---

## Things to Research

1. [ ] How does PyPDFLoader work?
2. [ ] What is TokenTextSplitter?
3. [ ] Can I add custom metadata?
4. [ ] What's the max chunk_size?
5. [ ] ___________

---

## My "Teach It Back" Summary

**In plain English, here's what I learned today:**

Document loading is like ___________
Chunking is like ___________
The most important thing to understand is ___________
The thing that surprised me most was ___________
A QA analogy that helps me understand this is ___________

---

## Confidence Self-Assessment

| Topic | 1-10 |
|-------|------|
| load_documents() method | |
| DirectoryLoader parameters | |
| Document object structure | |
| split_documents() method | |
| chunk_size parameter | |
| chunk_overlap parameter | |
| RecursiveCharacterTextSplitter | |
| Metadata preservation | |
| **Overall Day 2** | |

---

## Tomorrow's Preparation

**Day 3 topic:** Embeddings & Vector Database

**Before tomorrow, think about:**
If you had 1 million text chunks and needed to find
the 3 most similar to a query in milliseconds,
how would YOU design the storage and search system?

My initial thoughts:
___________
___________

---

**End of Day 2 - Great Work! 🎉**
