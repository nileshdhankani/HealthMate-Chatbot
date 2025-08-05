# RAG-style Chatbot for SDG 3 using Gemini API + ChromaDB + Streamlit

# ----------------------
# Step 1: Install Required Packages
# pip install streamlit google-generativeai chromadb sentence-transformers python-dotenv
# ----------------------

import os
import pickle
import streamlit as st
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ----------------------
# Step 2: Load API Key
# ----------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please add GOOGLE_API_KEY to your .env file")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------
# Step 3: Prepare SDG 3 Knowledge Base (if not already embedded)
# ----------------------
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sdg3_docs"
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH))

if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
    with open("sdg3_docs.txt", "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")

    collection = client.create_collection(name=COLLECTION_NAME)
    embeddings = model.encode(docs)
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        collection.add(documents=[doc], ids=[f"doc_{i}"], embeddings=[emb.tolist()])
else:
    collection = client.get_collection(COLLECTION_NAME)

# ----------------------
# Step 4: Define RAG Function
# ----------------------
gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def query_sdg3_chatbot(user_query):
    query_embedding = model.encode([user_query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    relevant_passages = results["documents"][0]
    context = "\n\n".join(relevant_passages)

    prompt = f"""
You are a helpful assistant with expertise in SDG 3: Good Health and Well-being.
Only use the following context to answer the question:

{context}

Question: {user_query}
Answer:
"""
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------
# Step 5: Streamlit UI
# ----------------------
st.set_page_config(page_title="SDG 3 RAG Chatbot", page_icon="🧬")
st.title("🧬 HealthMate RAG - SDG 3 Expert Bot")

st.markdown("""
Ask any question about SDG 3 - Good Health and Well-being.
Examples:
- What is Ayushman Bharat?
- How can we reduce child mortality?
- What are India's health schemes?
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something about SDG 3...")
if user_input:
    with st.spinner("Thinking..."):
        reply = query_sdg3_chatbot(user_input)
    st.session_state.chat_history.append((user_input, reply))

for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

# ----------------------
# Step 6: Deployment Guide
# ----------------------
# 1. Save this file as `rag_chatbot.py`
# 2. Upload `sdg3_docs.txt` (your knowledge base)
# 3. Create a `.env` file with:
#    GOOGLE_API_KEY=your_key_here
# 4. Run locally: `streamlit run rag_chatbot.py`
# 5. To deploy:
#    - Push to GitHub
#    - Deploy on Streamlit Cloud
#    - Add `GOOGLE_API_KEY` under Secrets
