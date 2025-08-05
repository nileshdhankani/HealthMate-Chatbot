# ---------------------- Required Packages ----------------------
# pip install streamlit google-generativeai sentence-transformers faiss-cpu python-dotenv
# ---------------------------------------------------------------

import os
import pickle
import numpy as np
import streamlit as st
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ---------------------- Load API Key ----------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment!")
    st.stop()
genai.configure(api_key=api_key)

# ---------------------- Embedding Setup ----------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists("sdg3_index.pkl"):
    with open("sdg3_docs.txt", "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")
    embeddings = embed_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    with open("sdg3_index.pkl", "wb") as f:
        pickle.dump((docs, index, embed_model), f)
else:
    with open("sdg3_index.pkl", "rb") as f:
        docs, index, embed_model = pickle.load(f)

# ---------------------- Gemini Setup ----------------------
model = genai.GenerativeModel("gemini-1.5-flash")

def get_response(query):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)

    # Gather the top context passages
    retrieved_passages = [docs[i] for i in I[0] if i < len(docs)]
    context = "\n\n".join(retrieved_passages).strip()

    # Fallback if no context is found or too short
    if not context or len(context.split()) < 20:
        prompt = f"""
You are a helpful assistant with expertise in general health and SDG 3 (Good Health and Well-being).
Answer the question below using your own knowledge:

Question: {query}
Answer:"""
    else:
        prompt = f"""
You are a helpful assistant with knowledge about SDG 3 (Health and Well-being).
Only use the context below to answer the question. If the context is not relevant, you may use your own knowledge.

Context:
{context}

Question: {query}
Answer:"""

    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="SDG 3 Chatbot", page_icon="💊")
st.title("💊 HealthMate - SDG 3 Chatbot")
st.markdown("Ask anything related to **Good Health and Well-being (SDG 3)**.")

if "chat" not in st.session_state:
    st.session_state.chat = []

q = st.chat_input("Ask about health...")
if q:
    with st.spinner("Thinking..."):
        a = get_response(q)
    st.session_state.chat.append((q, a))

for user, bot in st.session_state.chat:
    st.chat_message("user").write(user)
    st.chat_message("assistant").write(bot)
