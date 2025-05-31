
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import tempfile

# ✅ Securely access your API key from .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Keystone Brain", page_icon="🧠")
st.title("🧠 Keystone Brain")
st.write("Upload your internal documents (PDFs or Word) and ask anything.")

DB_DIR = "db"
embedding = OpenAIEmbeddings()

# Load persistent DB if exists
if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
else:
    db = None
    qa = None

uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file: {uploaded_file.name}")
            continue

        docs = loader.load_and_split()
        all_docs.extend(docs)

    if all_docs:
        db = Chroma.from_documents(all_docs, embedding, persist_directory=DB_DIR)
        db.persist()
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

if qa:
    question = st.text_input("Ask Keystone Brain a question:")
    if question:
        answer = qa.run(question)
        st.write("📣 Answer:", answer)
