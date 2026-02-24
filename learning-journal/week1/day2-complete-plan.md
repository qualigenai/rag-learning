# WEEK 1, DAY 2 - COMPLETE PLAN
## Document Loading & Chunking - The RAG Entry Point

**Date:** [Today's date]
**Time Started:** _____
**Time Ended:** _____
**Total Study Time:** _____

**QA Connection:** Document loading = Test data preparation!
Chunking = Test case breakdown into smaller test steps!

---

## ⏰ SCHEDULE

```
Session 1 (45 min): Deep dive into load_documents()
Session 2 (45 min): Deep dive into split_documents()
Session 3 (30 min): Hands-on Experiments
Session 4 (30 min): Documentation & Reflection
```

---

## 🔗 CONNECTING TO DAY 1

**Day 1 you learned:**
```
__init__ creates:
- self.embeddings (embedding model)
- self.llm (language model)
- self.vectorstore = None (to be filled)
- self.qa_chain = None (to be filled)
```

**Day 2 you'll learn:**
```
load_documents():
- Reads files from disk
- Returns Document objects

split_documents():
- Takes Documents
- Returns smaller Chunks

These fill the pipeline BEFORE embeddings!
```

**The Growing Picture:**
```
[DAY 2]         [DAY 3]       [DAY 4]
Files → Load → Split → Embed → Store → Retrieve → Answer
  ↑
You are here!
```

---

## SESSION 1: DEEP DIVE INTO load_documents() (45 min)

### THE CODE (Lines 41-55):

```python
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
```

---

### BLOCK 1: Method Signature

```python
def load_documents(self):
```

**Q1: Why does this method take only `self`?**
```
Answer: No additional parameters needed because:
- self.data_dir already stored in __init__
- Method is self-contained
- Uses object's own data

QA Connection: Like a test that uses pre-configured fixtures!
```

**Q2: What will this method return?**
```
Answer: A list of Document objects
Each Document has:
- page_content: The actual text
- metadata: {'source': 'filepath', ...}

Think: Returns TEST DATA for the RAG system!
```

**Q3: What's a LangChain Document object?**
```
It looks like this:
Document(
    page_content="The text of your document...",
    metadata={
        'source': './data/rag_introduction.txt',
        'row': 0
    }
)
```

---

### BLOCK 2: The Print Statement

```python
print(f"Loading documents from {self.data_dir}...")
```

**Q1: What is f-string?**
```
Answer: Python format string
f"..." allows variables inside {}
self.data_dir = "./data"
So output: "Loading documents from ./data..."

Practice: Open Python and try:
name = "RAG"
print(f"Hello {name}!")
```

**Q2: Why print progress?**
```
Answer:
- User knows something is happening
- Debugging aid
- Shows what directory is being used
- Professional UX practice

QA Connection: Like test execution logs!
```

---

### BLOCK 3: Creating the DirectoryLoader

```python
loader = DirectoryLoader(
    self.data_dir, 
    glob="**/*.txt",
    loader_cls=TextLoader
)
```

**Q1: What is DirectoryLoader?**
```
Answer: Loads ALL files from a directory
Recursively searches for matching files
Returns list of Document objects

How it works internally:
1. Scan directory (self.data_dir)
2. Find files matching glob pattern
3. Load each file using loader_cls
4. Return list of Documents
```

**Q2: Dissecting glob="**/*.txt"**
```
**    = Any directory (including subdirectories)
/     = Path separator
*     = Any filename
.txt  = Must end in .txt

So "**/*.txt" means:
"Find ALL .txt files in ANY subdirectory"

Examples that MATCH:
✓ data/rag_intro.txt
✓ data/subfolder/notes.txt
✓ data/a/b/c/deep.txt

Examples that DON'T MATCH:
✗ data/document.pdf
✗ data/spreadsheet.csv
✗ notes.txt (must be in data_dir)

QA Connection: Like a test filter/selection pattern!
```

**Q3: What is loader_cls=TextLoader?**
```
Answer: Specifies HOW to load each file
TextLoader = reads plain text files
Other options:
- PyPDFLoader (PDF files)
- CSVLoader (CSV files)
- JSONLoader (JSON files)
- UnstructuredHTMLLoader (HTML files)

QA Connection: Like selecting which test runner to use!
```

**Q4: What if a file fails to load?**
```
Default behavior: FAILS entire loading
Better for production: Add error handling

Try this later:
loader = DirectoryLoader(
    self.data_dir, 
    glob="**/*.txt",
    loader_cls=TextLoader,
    silent_errors=True  # Skip failed files
)

QA Mindset: What happens with corrupt files?
What about empty files?
What about very large files?
These are your test cases!
```

---

### BLOCK 4: Loading and Returning

```python
documents = loader.load()

print(f"Loaded {len(documents)} documents")
return documents
```

**Q1: What does .load() do?**
```
Answer: Actually reads all the files
Returns: List of Document objects
Each document = one loaded file
Slow for many large files
```

**Q2: What does len(documents) tell us?**
```
Answer: Number of files successfully loaded
If 7 files in data/ → 7 documents
But not 7 chunks yet! That's next step.

Important distinction:
Documents = Files (7)
Chunks = Pieces of documents (50+)
```

**Q3: Why return documents?**
```
Answer: Other methods need them
split_documents() will receive them
Could store as self.documents but
returning is cleaner for pipeline
```

---

### DOCUMENT OBJECT DEEP DIVE

**Open Python and explore:**

```python
from langchain_community.document_loaders import TextLoader

# Load one file
loader = TextLoader("./data/rag_introduction.txt")
docs = loader.load()

# Explore the result
print(type(docs))           # <class 'list'>
print(len(docs))            # 1 (one file = one doc)
print(type(docs[0]))        # <class 'langchain...Document'>
print(docs[0].page_content[:100])  # First 100 chars
print(docs[0].metadata)    # Source file info
```

**KEY LEARNING:**
- One file → One Document (by default)
- Document has: page_content + metadata
- Metadata tells you WHERE the text came from
- Critical for citations in RAG!

---

## SESSION 1 CHECKPOINT ✅

Can you answer these?
- [ ] What does DirectoryLoader do?
- [ ] What does glob="**/*.txt" match?
- [ ] What is a Document object?
- [ ] What's stored in metadata?
- [ ] What happens with non-.txt files?

---

## SESSION 2: DEEP DIVE INTO split_documents() (45 min)

### THE CODE (Lines 57-67):

```python
def split_documents(self, documents):
    """Split documents into smaller chunks."""
    print("Splitting documents into chunks...")
    
    # Split documents - important for good retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks
```

---

### BLOCK 1: Method Signature

```python
def split_documents(self, documents):
```

**Q1: Why does this take `documents` as parameter?**
```
Answer: Receives output from load_documents()
This creates a clean pipeline:
load_documents() → split_documents()

Could have been self.documents but
taking it as parameter = more flexible
Can split ANY list of documents, not just loaded ones

QA Connection: Like a function that processes test data
passed to it as input
```

---

### BLOCK 2: Creating the Text Splitter

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
```

**THE MOST IMPORTANT PARAMETERS IN RAG!**

---

#### Parameter 1: chunk_size=500

**What it means:**
```
Each chunk will be APPROXIMATELY 500 characters
The splitter tries to split AT boundaries
so actual size may vary slightly

500 characters ≈ 80-100 words ≈ 3-5 sentences

Visual example:
Original text (2000 chars):
"RAG stands for Retrieval-Augmented Generation...
[paragraph 1, ~500 chars]
[paragraph 2, ~500 chars]
[paragraph 3, ~500 chars]
[paragraph 4, ~500 chars]"

Result: 4 chunks of ~500 chars each
```

**Why 500?**
```
Too small (< 200):
- ❌ Loses context
- ❌ Fragments sentences
- ❌ Poor retrieval quality
- Example: "RAG stands for" (not useful alone)

Too large (> 2000):
- ❌ Loses precision
- ❌ Retrieves too much irrelevant text
- ❌ Fills LLM context window fast
- Example: Entire document in one chunk

Sweet spot (300-800):
- ✅ Preserves context
- ✅ Precise retrieval
- ✅ Efficient
- ✅ Works well in practice

QA Connection: Like breaking large test suites into
manageable test cases - not too small, not too large!
```

---

#### Parameter 2: chunk_overlap=50

**What it means:**
```
50 characters from END of chunk N
appear at START of chunk N+1

Visual example:
Document: "...context about RAG. The key components
are: embeddings, vector store..."

Chunk 1: "...context about RAG. The key components"
                                    ↑
                               Last 50 chars
Chunk 2: "The key components are: embeddings, vector store..."
         ↑
    First 50 chars (overlap!)
```

**Why overlap?**
```
WITHOUT overlap (overlap=0):
"...key components" | "are: embeddings..."
→ Sentence is SPLIT across chunks
→ Neither chunk makes complete sense
→ Bad retrieval!

WITH overlap (overlap=50):
"...key components are:" | "components are: embeddings..."
→ Key phrase preserved in BOTH chunks
→ Better retrieval!

Trade-off:
More overlap → Better continuity but more storage
Less overlap → Less storage but might miss context

QA Connection: Like having test data overlap between
test suites to catch boundary issues!
```

---

#### Parameter 3: length_function=len

**What it means:**
```
How to COUNT chunk size
len = count characters

Alternative: Count TOKENS instead of characters
Why matters:
- LLMs think in tokens, not characters
- 1 token ≈ 4 characters (roughly)
- 500 chars ≈ 125 tokens

For learning: len is fine
For production: Consider token counting
```

---

### BLOCK 3: How RecursiveCharacterTextSplitter WORKS

**The "Recursive" Strategy:**
```
Priority order for split points:
1. "\n\n"  (paragraph breaks) ← Try this first
2. "\n"    (line breaks)
3. " "     (spaces/words)
4. ""      (characters) ← Last resort

Example text:
"This is paragraph 1.\n\nThis is paragraph 2.\nWith a new line.\nAnd another."

Step 1: Try splitting at \n\n
→ "This is paragraph 1."
→ "This is paragraph 2.\nWith a new line.\nAnd another."

If either piece > chunk_size, split further:
Step 2: Try splitting at \n
→ "This is paragraph 2."
→ "With a new line."
→ "And another."

This preserves NATURAL language boundaries!
Sentences stay together whenever possible

QA Connection: Like trying coarse test groupings first,
then finer granularity if needed!
```

---

### BLOCK 4: The Actual Splitting

```python
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
return chunks
```

**Q1: What's the difference between documents and chunks?**
```
Documents (before split):
- 7 items (one per file)
- Each could be thousands of chars
- Metadata has source file

Chunks (after split):
- 50+ items (multiple per file)
- Each ~500 chars
- Metadata PRESERVED from original!

Critical: Chunks still know which file they came from!
```

**Q2: What does a chunk look like?**
```python
Document(
    page_content="RAG stands for Retrieval-Augmented 
    Generation. It combines information retrieval...",
    metadata={
        'source': './data/rag_introduction.txt'
        # Same metadata as original document!
    }
)
```

**Q3: Why preserve metadata in chunks?**
```
Answer: CITATIONS!
When user asks a question:
RAG finds relevant chunk
Returns answer
Shows SOURCE: "From rag_introduction.txt"

This is how RAG avoids hallucinations
and provides verifiable answers!

QA Connection: Like test results showing WHICH
test file and test case failed!
```

---

### CHUNKING STRATEGY COMPARISON

| Strategy | chunk_size | chunk_overlap | Best For |
|----------|-----------|---------------|----------|
| Micro | 200 | 20 | Very precise retrieval |
| Small | 300 | 50 | Technical docs |
| Medium | 500 | 50 | General use (current) |
| Large | 1000 | 100 | Narrative text |
| XLarge | 2000 | 200 | Long-form content |

---

## SESSION 2 CHECKPOINT ✅

Can you answer these?
- [ ] What does chunk_size=500 mean?
- [ ] Why do we use chunk_overlap?
- [ ] What's the "recursive" in RecursiveCharacterTextSplitter?
- [ ] Why does chunk count > document count?
- [ ] What metadata is preserved in chunks?

---

## SESSION 3: HANDS-ON EXPERIMENTS (30 min)

### Experiment 1: Explore Document Objects (10 min)

Create: `experiments/day2/explore_documents.py`

```python
"""
Experiment 1: Understanding Document Objects
"""
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader

print("="*60)
print("EXPERIMENT 1: DOCUMENT OBJECTS")
print("="*60)

# Test 1: Load a single file
print("\n--- Test 1: Single File Loading ---")
loader = TextLoader("./data/rag_introduction.txt")
docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Type: {type(docs[0])}")
print(f"\nPage Content (first 200 chars):")
print(docs[0].page_content[:200])
print(f"\nMetadata:")
print(docs[0].metadata)
print(f"\nTotal characters: {len(docs[0].page_content)}")

# Test 2: Load all files
print("\n--- Test 2: Directory Loading ---")
dir_loader = DirectoryLoader("./data", glob="**/*.txt", 
                              loader_cls=TextLoader)
all_docs = dir_loader.load()

print(f"Total documents loaded: {len(all_docs)}")
print(f"\nDocument sources:")
for doc in all_docs:
    print(f"  - {doc.metadata['source']} ({len(doc.page_content)} chars)")

# Test 3: Inspect metadata
print("\n--- Test 3: Metadata Analysis ---")
total_chars = sum(len(doc.page_content) for doc in all_docs)
print(f"Total characters across all docs: {total_chars:,}")
print(f"Average document size: {total_chars // len(all_docs):,} chars")

# Test 4: Try non-existent directory
print("\n--- Test 4: Error Handling ---")
try:
    bad_loader = DirectoryLoader("./nonexistent", glob="**/*.txt",
                                  loader_cls=TextLoader)
    bad_docs = bad_loader.load()
    print(f"Loaded {len(bad_docs)} docs")
except Exception as e:
    print(f"Error (expected): {type(e).__name__}: {e}")
    print("QA Insight: Need error handling for missing directories!")

print("\n" + "="*60)
print("EXPERIMENT 1 COMPLETE!")
print("="*60)
```

**Record in notes:**
- How many characters per document?
- What does metadata look like?
- What error occurred with bad directory?
- What does this tell you about error handling?

---

### Experiment 2: Chunking Comparison (15 min)

Create: `experiments/day2/chunking_comparison.py`

```python
"""
Experiment 2: Comparing Chunking Strategies
This is your first QA-style systematic comparison!
"""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

print("="*60)
print("EXPERIMENT 2: CHUNKING STRATEGIES COMPARISON")
print("="*60)

# Load a sample document
loader = TextLoader("./data/rag_introduction.txt")
docs = loader.load()
original_size = len(docs[0].page_content)
print(f"\nOriginal document: {original_size} characters")

# Test different configurations
configs = [
    {"chunk_size": 200, "chunk_overlap": 0,   "name": "Small/No-Overlap"},
    {"chunk_size": 200, "chunk_overlap": 50,  "name": "Small/With-Overlap"},
    {"chunk_size": 500, "chunk_overlap": 0,   "name": "Medium/No-Overlap"},
    {"chunk_size": 500, "chunk_overlap": 50,  "name": "Medium/With-Overlap (CURRENT)"},
    {"chunk_size": 1000, "chunk_overlap": 100, "name": "Large/With-Overlap"},
    {"chunk_size": 2000, "chunk_overlap": 200, "name": "XLarge/With-Overlap"},
]

results = []
print("\nTesting configurations...")
print("-"*60)

for config in configs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len
    )
    
    chunks = splitter.split_documents(docs)
    
    # Calculate stats
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    result = {
        "Config": config["name"],
        "Chunk Size": config["chunk_size"],
        "Overlap": config["chunk_overlap"],
        "Num Chunks": len(chunks),
        "Avg Actual Size": round(avg_size),
        "Min Size": min(chunk_sizes) if chunk_sizes else 0,
        "Max Size": max(chunk_sizes) if chunk_sizes else 0,
    }
    results.append(result)
    
    print(f"\n{config['name']}:")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {round(avg_size)} chars")
    print(f"  Size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Show actual chunk content for current config
print("\n" + "="*60)
print("SAMPLE: First 3 chunks with CURRENT config (500/50)")
print("="*60)
current_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=len
)
current_chunks = current_splitter.split_documents(docs)

for i, chunk in enumerate(current_chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content)
    print(f"Source: {chunk.metadata['source']}")

# Show overlap in action
print("\n" + "="*60)
print("OVERLAP DEMONSTRATION")
print("="*60)
if len(current_chunks) >= 2:
    chunk1_end = current_chunks[0].page_content[-60:]
    chunk2_start = current_chunks[1].page_content[:60:]
    print(f"\nEnd of Chunk 1: ...{chunk1_end}")
    print(f"Start of Chunk 2: {chunk2_start}...")
    
    # Find overlap
    common_words = set(chunk1_end.split()) & set(chunk2_start.split())
    print(f"\nCommon words (showing overlap): {common_words}")

print("\n" + "="*60)
print("KEY LEARNINGS:")
print("="*60)
print("1. Smaller chunks = More chunks = More precise retrieval")
print("2. Larger chunks = Fewer chunks = More context per chunk")
print("3. Overlap prevents splitting mid-sentence")
print("4. No single 'right' size - depends on content and use case")
print("\nQA Insight: Just like test case granularity!")
print("Too fine = too many small tests")
print("Too coarse = missing edge cases")
print("Balance is key!")

print("\n✅ Save your results to experiments/day2/chunking_results.csv")
df.to_csv("experiments/day2/chunking_results.csv", index=False)
print("Results saved!")
```
`
**Record in notes:**
- Which config creates most chunks? Fewest?
- Can you see the overlap between chunks?
- Which config would YOU choose and why?
- How does this relate to your QA experience?
`
---

### Experiment 3: Metadata Preservation Test (5 min)

Create: `experiments/day2/metadata_test.py`

```python
"""
Experiment 3: Metadata Preservation Through Pipeline
QA Focus: Does metadata survive chunking? (This is a REAL test!)
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("="*60)
print("EXPERIMENT 3: METADATA PRESERVATION")
print("="*60)

# Load documents
loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# VERIFY METADATA PRESERVED
print("\n✅ Testing metadata preservation...")
print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

# Check each chunk has metadata
chunks_with_metadata = [c for c in chunks if c.metadata]
chunks_without_metadata = [c for c in chunks if not c.metadata]

print(f"\nChunks WITH metadata: {len(chunks_with_metadata)}")
print(f"Chunks WITHOUT metadata: {len(chunks_without_metadata)}")

if len(chunks_without_metadata) == 0:
    print("✅ TEST PASSED: All chunks have metadata!")
else:
    print("❌ TEST FAILED: Some chunks missing metadata!")

# Show metadata variety
print("\nMetadata from first 5 chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"  Chunk {i+1}: {chunk.metadata}")

# Show which file each chunk came from
print("\nChunks per source file:")
source_counts = {}
for chunk in chunks:
    source = chunk.metadata.get('source', 'unknown')
    source_counts[source] = source_counts.get(source, 0) + 1

for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count} chunks")

print("\n" + "="*60)
print("QA INSIGHT:")
print("="*60)
print("Metadata = Traceability")
print("Every chunk knows its source file")
print("Like knowing WHICH test case found a bug!")
print("Critical for debugging retrieval issues!")
print("="*60)
```

---

## SESSION 4: DOCUMENTATION & REFLECTION (30 min)

### Task 1: Update Your Learning Journal

In `learning-journal/week1/day2-notes.md`:

```markdown
# Day 2 Notes - Document Loading & Chunking

## What I Learned

### load_documents():
1. [Key learning 1]
2. [Key learning 2]

### split_documents():
1. [Key learning 1]
2. [Key learning 2]

## Experiment Results

### Experiment 1 - Document Objects:
- Number of documents: ___
- Average document size: ___ chars
- Interesting observation: ___

### Experiment 2 - Chunking Comparison:
| Config | Chunks | Notes |
|--------|--------|-------|
| Small/No-Overlap | | |
| Medium/With-Overlap | | |
| Large/With-Overlap | | |

### Experiment 3 - Metadata:
- Metadata preserved: Yes/No
- Insight: ___

## My Best Understanding

### The Pipeline So Far:
```
Files (disk)
    ↓ [DirectoryLoader + TextLoader]
Document objects (page_content + metadata)
    ↓ [RecursiveCharacterTextSplitter]
Chunk objects (smaller pieces + same metadata)
    ↓ [Tomorrow: HuggingFaceEmbeddings]
Vectors...
```

## Questions I Still Have
1. ___
2. ___

## QA Connections I Made
1. ___
2. ___

## Confidence Level: ___ / 10
```

---

### Task 2: The "Teach It Back" Exercise (15 min)

**Explain in simple words to an imaginary friend:**

```
"Hey, imagine you have 7 text files about RAG.
When we 'load' them, what happens is... 
**Reads the content of each file and converts into doc objects**
**Stores : The actual text and Meta data**
**So loading = bringing raw text into memory in a structured format.**

Then when we 'split' them...
**The long text inside each document is broken into smaller chunks.**
**1 file = 5,000 words
After splitting = maybe 25 chunks of 200 words each**

The reason we split them is because...
**_LLMs have token limits – they cannot read very large documents at once.
Embeddings work better on smaller text pieces.
Retrieval becomes more accurate.**_

The tricky part is chunk_overlap, which works like...
**Without overlap, context can get cut off.**
The most important thing I learned today is...
**RAG is not just about storing documents in a vector database.
The quality of splitting directly affects:
    Retrieval quality
    Context accuracy
    Final LLM answer
Bad chunking = bad answers
Good chunking = smart AI system**

```

Write this in your notes! This is MORE valuable than copying code!

---

## ✅ DAY 2 DELIVERABLES CHECKLIST

### Files Created:
- [ ] `experiments/day2/explore_documents.py`
- [ ] `experiments/day2/chunking_comparison.py`
- [ ] `experiments/day2/metadata_test.py`
- [ ] `experiments/day2/chunking_results.csv`
- [ ] `learning-journal/week1/day2-notes.md`

### Knowledge Gained:
- [ ] Understand load_documents() completely
- [ ] Understand split_documents() completely
- [ ] Know what a Document object contains
- [ ] Understand chunk_size and chunk_overlap
- [ ] Know how metadata is preserved
- [ ] Can explain the chunking strategy

### QA Skills Applied:
- [ ] Ran systematic comparison (Experiment 2)
- [ ] Verified expected behavior (Experiment 3)
- [ ] Documented results in table format
- [ ] Identified edge cases (error handling)

---

## 🏆 SUCCESS CRITERIA

**You've succeeded if you can:**

1. **Draw the flow:**
   Files → Documents → Chunks (with details at each step)

2. **Answer without looking:**
   - What does glob="**/*.txt" match?
   - Why do we use chunk_overlap?
   - What's preserved in chunk metadata?
   - What happens to one file after splitting?

3. **Make a decision:**
   - For a technical manual with precise facts:
     Would you use chunk_size=200 or chunk_size=1000? Why?

4. **Identify a bug:**
   - If all chunks have metadata = {} (empty)
     What went wrong?

---

## 📅 TOMORROW PREVIEW: Day 3

Day 3 covers:
- `create_vectorstore()` - Converting chunks to vectors
- How embeddings are created
- How ChromaDB stores vectors
- The actual vector creation process

**Preparation for Day 3:**
Think about: How would YOU store millions of text vectors
and search them quickly?

---

## 🎯 END OF DAY REFLECTION

Answer these honestly:

```
1. What was hardest today? _______________

2. What surprised me most? _______________

3. My QA background helped me when: _______________

4. The concept I want to explore more: _______________

5. My confidence today (1-10): ___

6. One thing I'd tell my Day-1-self: _______________
```

---

## 🚀 WELL DONE!

**Today you mastered:**
- How documents enter the RAG system
- How they're processed into chunks
- Why chunking strategy matters enormously
- How metadata enables citations
- Systematic comparison (true QA mindset!)

**You're building REAL expertise, one layer at a time!**

---

**Tomorrow: Embeddings & Vector Database - where the magic happens! 🎯**
