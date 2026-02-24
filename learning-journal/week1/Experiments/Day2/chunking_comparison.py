from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

print("=" * 60)
print("EXPERIMENT 2: CHUNKING STRATEGIES COMPARISON")
print("=" * 60)

# Load a sample document
loader = TextLoader("../../../../data/rag_introduction.txt")
docs = loader.load()
original_size = len(docs[0].page_content)
print(f"\nOriginal document: {original_size} characters")

# Test different configurations
configs = [
    {"chunk_size": 200, "chunk_overlap": 0, "name": "Small/No-Overlap"},
    {"chunk_size": 200, "chunk_overlap": 50, "name": "Small/With-Overlap"},
    {"chunk_size": 500, "chunk_overlap": 0, "name": "Medium/No-Overlap"},
    {"chunk_size": 500, "chunk_overlap": 50, "name": "Medium/With-Overlap (CURRENT)"},
    {"chunk_size": 1000, "chunk_overlap": 100, "name": "Large/With-Overlap"},
    {"chunk_size": 2000, "chunk_overlap": 200, "name": "XLarge/With-Overlap"},
]

results = []
print("\nTesting configurations...")
print("-" * 60)

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
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Show actual chunk content for current config
print("\n" + "=" * 60)
print("SAMPLE: First 3 chunks with CURRENT config (500/50)")
print("=" * 60)
current_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=len
)
current_chunks = current_splitter.split_documents(docs)

for i, chunk in enumerate(current_chunks[:3]):
    print(f"\n--- Chunk {i + 1} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content)
    print(f"Source: {chunk.metadata['source']}")

# Show overlap in action
print("\n" + "=" * 60)
print("OVERLAP DEMONSTRATION")
print("=" * 60)
if len(current_chunks) >= 2:
    chunk1_end = current_chunks[0].page_content[-60:]
    chunk2_start = current_chunks[1].page_content[:60:]
    print(f"\nEnd of Chunk 1: ...{chunk1_end}")
    print(f"Start of Chunk 2: {chunk2_start}...")

    # Find overlap
    common_words = set(chunk1_end.split()) & set(chunk2_start.split())
    print(f"\nCommon words (showing overlap): {common_words}")

print("\n" + "=" * 60)
print("KEY LEARNINGS:")
print("=" * 60)
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