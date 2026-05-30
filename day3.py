import pdfplumber
from sentence_transformers import SentenceTransformer

# Step 1: Extract text from PDF (same as day 2)
all_text = ""
with pdfplumber.open("notes.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

# Step 2: Create chunks
chunk_size = 500
chunks = []
for i in range(0, len(all_text), chunk_size):
    chunk = all_text[i:i+chunk_size]
    chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")

# Step 3: Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert chunks to vectors
print("Converting chunks to vectors...")
vectors = model.encode(chunks)

print(f"Each chunk is now a vector of {len(vectors[0])} numbers")
print(f"Total vectors created: {len(vectors)}")
print()
print("First vector (first 10 numbers):")
print(vectors[0][:10])