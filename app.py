import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import tempfile
import shutil

# Set OpenAI key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ChromaDB directory for persistent storage
CHROMA_DIR = "chroma_db"
DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="Keystone Brain", layout="wide")
st.title("🧠 Keystone Brain")

# Upload widget
uploaded_files = st.file_uploader("Upload PDF or Word documents", type=["pdf", "docx"], accept_multiple_files=True)

# Initialize embeddings & vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Load new documents if uploaded
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
                tmp_path = tmp.name
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        for doc in splits:
            doc.metadata["source"] = file.name

        vectorstore.add_documents(splits)
        vectorstore.persist()

        st.success(f"{file.name} uploaded and indexed.")

# Ask a question
question = st.text_input("Ask a question based on all uploaded documents")
if question:
    docs = vectorstore.similarity_search(question, k=4)
    chain = load_qa_chain(ChatOpenAI(model_name="gpt-4", temperature=0), chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)
    st.markdown("### 💬 Answer")
    st.write(response)

# Document deletion (optional)
st.markdown("---")
st.subheader("🧹 Document Management")
if st.checkbox("Delete a document from memory"):
    existing_files = os.listdir(DOCS_DIR)
    file_to_delete = st.selectbox("Choose a document to delete", existing_files)
    if st.button("Delete Document"):
        try:
            os.remove(os.path.join(DOCS_DIR, file_to_delete))
            vectorstore.delete(filter={"source": file_to_delete})
            vectorstore.persist()
            st.success(f"{file_to_delete} removed from memory.")
        except Exception as e:
            st.error(f"Failed to delete: {e}")
