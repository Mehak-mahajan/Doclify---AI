import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ---- PAGE SETUP ----
st.set_page_config(
    page_title="Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Study Buddy")
st.subheader("Upload your notes and chat with them!")

# ---- LOAD MODELS ----
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ---- PDF PROCESSING ----
def process_pdf(uploaded_file):
    all_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    chunk_size = 500
    chunks = []
    for i in range(0, len(all_text), chunk_size):
        chunk = all_text[i:i+chunk_size]
        chunks.append(chunk)

    vectors = embedding_model.encode(chunks, show_progress_bar=False, batch_size=32)

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))

    return chunks, index

# ---- ASK QUESTION WITH STREAMING ----
def ask_question(question, chunks, index):
    question_vector = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_vector), 3)

    context = ""
    for i in indices[0]:
        context += chunks[i] + "\n\n"

    stream = groq_client.chat.completions.create(
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
        ],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

# ---- MAIN UI ----
uploaded_file = st.file_uploader(
    "Upload your PDF notes",
    type="pdf"
)

if uploaded_file is not None:
    with st.spinner("Reading and processing your PDF..."):
        chunks, index = process_pdf(uploaded_file)

    st.success(f"PDF processed! Created {len(chunks)} chunks. Ready to answer questions!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask anything about your notes...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            answer = st.write_stream(ask_question(question, chunks, index))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("👆 Please upload a PDF to get started!")