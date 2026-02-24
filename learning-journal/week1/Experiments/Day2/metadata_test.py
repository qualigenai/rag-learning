from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("="*60)
print("EXPERIMENT 3: METADATA PRESERVATION")
print("="*60)

# Load documents
loader = DirectoryLoader("../../../../data", glob="**/*.txt", loader_cls=TextLoader)
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