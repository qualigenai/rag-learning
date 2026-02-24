from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=3
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")