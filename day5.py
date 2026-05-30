import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ---- STEP 1: Extract text from PDF ----
print("Reading PDF...")
all_text = ""
with pdfplumber.open("notes.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

# ---- STEP 2: Create chunks ----
chunk_size = 500
chunks = []
for i in range(0, len(all_text), chunk_size):
    chunk = all_text[i:i+chunk_size]
    chunks.append(chunk)
print(f"Total chunks: {len(chunks)}")

# ---- STEP 3: Convert chunks to vectors ----
print("Creating vectors...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedding_model.encode(chunks)

# ---- STEP 4: Store in FAISS ----
print("Storing in FAISS...")
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))
print(f"FAISS ready with {index.ntotal} vectors!")

# ---- STEP 5: Setup Groq ----

groq_client = Groq(api_key="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")
# ---groq_client = Groq(api_key="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")

# ---- STEP 6: The magic function ----
def ask_question(question):
    # Find relevant chunks using FAISS
    question_vector = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_vector), 3)
    
    # Join top 3 chunks into context
    context = ""
    for i in indices[0]:
        context += chunks[i] + "\n\n"
    
    # Send context + question to Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant. Answer the question using ONLY the context provided. If the answer is not in the context, say 'I could not find this in the provided notes.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    
    return response.choices[0].message.content

# ---- STEP 7: Test it! ----
print("\n" + "="*50)
print("STUDY BUDDY IS READY!")
print("="*50)

while True:
    question = input("\nAsk a question (or type 'quit' to exit): ")
    if question.lower() == "quit":
        break
    
    print("\nSearching your notes...")
    answer = ask_question(question)
    print(f"\nAnswer: {answer}")
