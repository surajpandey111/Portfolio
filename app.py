import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")

@st.cache_resource
def load_vectorstore():
    loader = TextLoader("portfolio_data.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db

db = load_vectorstore()

st.title("🧠 Suraj Portfolio AI Chatbot")

question = st.text_input("Ask me anything about Suraj's work or skills:")

if question:
    results = db.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
Use this context to answer the question:

{context}

Question: {question}
"""

    try:
        response = model.generate_content(prompt)
        st.markdown(f"**Answer:** {response.text}")
    except Exception as e:
        st.error(f"Gemini error: {e}")
