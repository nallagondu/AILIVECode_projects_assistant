# streamlit_ai_assistant/app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
import requests
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import sqlite3
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Push notification helper
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("pushover_token"),
            "user": os.getenv("pushover_user"),
            "message": text,
        },
    )

# Tool: record user details
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

# Tool: record unknown questions
def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}

# Load resume + summary
@st.cache_data
def load_resume():
    reader = PdfReader("data/ailivecodeinfo.pdf")
    linkedin = ""
    for page in reader.pages:
        linkedin += page.extract_text() or ""
    with open("data/summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()
    return linkedin, summary

linkedin_text, summary_text = load_resume()

# System prompt
def system_prompt():
    return f"""
    You are acting as ailivecode boat. You are answering questions on ailivecode website, AILiveCode.com.
    You must represent his career and profile faithfully.

    ## Summary:
    {summary_text}

    ## LinkedIn Profile:
    {linkedin_text}

    If you don't know the answer to any question, use the record_unknown_question tool.
    Try to capture email for future contact.
    """

# ---------------- RAG Pipeline ------------------
@st.cache_resource
def build_vector_db():
    loaders = [
        PyPDFLoader(os.path.join("data", file))
        for file in os.listdir("data")
        if file.endswith(".pdf")
    ]
    docs = []
    for loader in loaders:
        for doc in loader.load():
            if doc.page_content:
                cleaned = doc.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
                doc.page_content = cleaned
                docs.append(doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings()
    )
    return vectordb

vectordb = build_vector_db()
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

# ---------------- Chat UI ------------------
st.title("🤖 Gangireddy - Your AI Assistant")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input after chat history
user_input = st.text_input("Ask me anything...", key="user_input")
if user_input:
    answer = rag_chain.invoke({"query": user_input})
    #answer = rag_chain.run(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Capture email
with st.sidebar:
    st.markdown("### 📬 Stay in touch")
    email = st.text_input("Your Email")
    name = st.text_input("Your Name")
    notes = st.text_area("Additional Notes")
    if st.button("Submit Contact"):
        res = record_user_details(email=email, name=name, notes=notes)

        conn = sqlite3.connect("contact_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contact_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                name TEXT,
                notes TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO contact_submissions (email, name, notes) VALUES (?, ?, ?)",
                       (email, name, notes))
        conn.commit()
        conn.close()
        st.success("Thanks! We'll be in touch.")

    if st.checkbox("Show Contact Submissions"):
        conn = sqlite3.connect("contact_data.db")
        df = pd.read_sql_query("SELECT * FROM contact_submissions", conn)
        st.dataframe(df)
        conn.close()
