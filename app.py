import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR = "chroma_db"
DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# ─── Streamlit UI Setup ───────────────────────────────────────────────────────

st.set_page_config(page_title="Keystone Brain", layout="wide")
st.title("🧠 Keystone Brain")

# ─── API Key Check ────────────────────────────────────────────────────────────

api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
if not api_key:
    st.error("❌ OPENAI_API_KEY is missing. Please check your secrets.")
    st.stop()

# ─── Initialize Embeddings & Vectorstore ──────────────────────────────────────

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# ─── File Upload ──────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader("Upload PDF or Word documents", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()
        file_path = os.path.join(DOCS_DIR, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file.getbuffer())
                loader = Docx2txtLoader(tmp.name)
        else:
            st.warning(f"⚠️ Unsupported file type: {file.name}")
            continue

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        for doc in splits:
            doc.metadata["source"] = file.name

        vectorstore.add_documents(splits)
        vectorstore.persist()
        st.success(f"✅ {file.name} uploaded and indexed.")

# ─── Question Answering ───────────────────────────────────────────────────────

question = st.text_input("Ask a question based on all uploaded documents")
if question:
    docs = vectorstore.similarity_search(question, k=4)
    chain = load_qa_chain(ChatOpenAI(model_name="gpt-4", temperature=0), chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)
    st.markdown("### 💬 Answer")
    st.write(response)

# ─── Document Deletion ────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🧹 Document Management")
if st.checkbox("Delete a document from memory"):
    existing_files = os.listdir(DOCS_DIR)
    if existing_files:
        file_to_delete = st.selectbox("Choose a document to delete", existing_files)
        if st.button("Delete Document"):
            try:
                os.remove(os.path.join(DOCS_DIR, file_to_delete))
                vectorstore.delete(filter={"source": file_to_delete})
                vectorstore.persist()
                st.success(f"🗑️ {file_to_delete} removed from memory.")
            except Exception as e:
                st.error(f"❌ Failed to delete: {e}")
    else:
        st.info("No documents to delete.")

