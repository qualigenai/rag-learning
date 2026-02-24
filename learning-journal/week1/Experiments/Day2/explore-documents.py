import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
print("Current Working Directory:", os.getcwd())
print("="*60)
print("EXPERIMENT 1: DOCUMENT OBJECTS")
print("="*60)

# Test 1: Load a single file
print("\n--- Test 1: Single File Loading ---")
loader = TextLoader("../../../../data/rag_introduction.txt")
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
dir_loader = DirectoryLoader("../../../../data", glob="**/*.txt",
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