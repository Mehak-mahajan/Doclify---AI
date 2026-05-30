import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Extract text from PDF
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

# Step 3: Convert chunks to vectors
print("Loading model and creating vectors...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(chunks)

# Step 4: Store vectors in FAISS
print("Storing vectors in FAISS...")
dimension = len(vectors[0])  # 384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))
print(f"Total vectors stored in FAISS: {index.ntotal}")

# Step 5: Search function
def search(question, top_k=3):
    # Convert question to vector
    question_vector = model.encode([question])
    
    # Search FAISS for top 3 most similar chunks
    distances, indices = index.search(np.array(question_vector), top_k)
    
    # Return the matching chunks
    results = []
    for i in indices[0]:
        results.append(chunks[i])
    return results

# Step 6: Test it!
print("\nSearching...")
question = "What is hydrocarbon?"
results = search(question)

print(f"\nQuestion: {question}")
print("\nTop 3 relevant chunks found:")
for i, chunk in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(chunk)