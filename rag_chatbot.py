# RAG-style Chatbot for SDG 3 using Gemini API + FAISS + Streamlit

# ----------------------
# Step 1: Install Required Packages
# pip install streamlit google-generativeai faiss-cpu sentence-transformers python-dotenv
# ----------------------

import os
import faiss
import pickle
import numpy as np
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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
def prepare_embeddings():
    with open("sdg3_docs.txt", "r") as f:
        docs = f.read().split("\n\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    with open("sdg3_knowledge.pkl", "wb") as f:
        pickle.dump((docs, index, model), f)

if not os.path.exists("sdg3_knowledge.pkl"):
    prepare_embeddings()

# ----------------------
# Step 4: Load Embeddings
# ----------------------
with open("sdg3_knowledge.pkl", "rb") as f:
    docs, index, embed_model = pickle.load(f)

# ----------------------
# Step 5: Define RAG Function
# ----------------------
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def query_sdg3_chatbot(user_query):
    query_vec = embed_model.encode([user_query])
    D, I = index.search(np.array(query_vec), k=3)
    relevant_passages = [docs[i] for i in I[0]]
    context = "\n\n".join(relevant_passages)

    prompt = f"""
You are a helpful assistant with expertise in SDG 3: Good Health and Well-being.
Only use the following context to answer the question:

{context}

Question: {user_query}
Answer:
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------
# Step 6: Streamlit UI
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
# Step 7: Deployment
# ----------------------
# 1. Save this as `rag_sdg3_chatbot.py`
# 2. Create `.env` with your Google Gemini key:
#    GOOGLE_API_KEY=your_key_here
# 3. Ensure `sdg3_docs.txt` is in the same folder.
# 4. Run: `streamlit run rag_sdg3_chatbot.py`
# 5. Or deploy on Streamlit Cloud with secret set.
