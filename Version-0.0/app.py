import os
import uuid
import tempfile
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#Page Configuration & Environment Setup
st.set_page_config(
    page_title="Production RAG - Version 0.0",
    page_icon = "🗒️",
    layout = "wide"
)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

#----------------------------------------------------------
# Cached Resources (Intialized once per session)
#----------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model (bge-small-en-v1.5)")
def load_embedding():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner="Connecting to Groq LPU inference engine...")
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
        api_key=api_key
    )

#-------------------------------------------------------------
# Pipeline Functions
#-------------------------------------------------------------
def process_pdf(file_path: str, embeddings):
    """
    V0.0 Pipeline:
    1. Parse with PyPDFLoader
    2. Chunk with RecursiveTextSplitter (1000 chars, 200  overlap)
    2. Index in-memory or persistent Chroma vectorstore
    """
    #1. PARSING
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    #2. CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n","\n"," ",""]
    )
    chunks = text_splitter.split_documents(documents)

    #New: Safety check to prevent Chroma from crashing on empty chunks
    if not chunks:
        st.error("⚠️ No text could be extracted. The PDF might be image-based (scanned) or corrupted.")
        st.stop()

    #3. EMBEDDING & INDEXING
    #Generate a unique ID to ensure a completely fresh database
    unique_collection_name = f"collection_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=unique_collection_name
    )

    #4. RETRIEVER SETUP
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs={"k":4}
    )
    return retriever, len(documents), len(chunks)

def build_rag_chain(retriever, llm):
    #5. PROMPT ORCHESTRATION
    template = """Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Answer: """
    prompt = PromptTemplate.from_template(template)

#6RUNNABLE CHAIN
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

#----------------------------------------------------------------
# Streamlit Interface
#----------------------------------------------------------------
st.title("🛡️ Production-Grade RAG: Version 0.0")
st.caption("Baseline Architecture: PyPDFLoader | Character Chunking | Groq LPU")

#Verify API Key
if not api_key:
    st.error("`GROQ_API_KEY` not detected in '.env'. Please check your configuration.")
    st.stop()

embeddings = load_embedding()
llm = load_llm()

#Intialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

#Sidebar: Document Management
with st.sidebar:
    st.header("📂 Document Selection")

    source_docs_dir = Path("../Source-Documents")
    available_files = []
    if source_docs_dir.exists():
        available_files = [f.name for f in source_docs_dir.glob("*.pdf")]

    doc_source = st.radio(
        "Choose document source:",
        options = ["Select existing document", "Upload new pdf"]
    )

    selected_pdf_path = None

    if doc_source == "Select existing document":
        if available_files:
            chosen_file = st.selectbox("Available in `Source-Documents/`:",available_files)
            selected_pdf_path = str(source_docs_dir / chosen_file)
            display_name = chosen_file
        else:
            st.info("No PDFs found in `../Source-Documents/`. You can upload one below.")
    else:
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded_file is not None:
            #Bypass tempfile entirely to avoid Windows file-locking quirks
            display_name = uploaded_file.name
            selected_pdf_path = "temp_uploaded.pdf"

            #Use getbuffer() which safely reads memory without pointer issues
            with open(selected_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    #Ingestion Trigger Button
    if st.button("Ingest & Build Index", type="primary", use_container_width=True):
        if not selected_pdf_path:
            st.warning("Please select or upload a PDF first.")
        else:
            with st.spinner("Processing document through Version 0.0 pipeline..."):
                retriever, num_pages, num_chunks = process_pdf(selected_pdf_path, embeddings)
                st.session_state.rag_chain = build_rag_chain(retriever, llm)
                st.session_state.active_doc = display_name
                st.session_state.messages = [] #Reset chat on new document
                st.success(f"Indexed **{num_pages} pages** into **{num_chunks} chunks**!")


    #Information Box
    st.markdown("---")
    st.markdown("### ⚠️ V0.0 Baseline Note")
    st.info(
        "Version 0.0 parses via character slicing. "
        "Complex tabular grids (e.g., PFMEAs) may suffer structural data loss."
    )

#--------------------------------------------------------------------
# Chat Interface
#--------------------------------------------------------------------
if st.session_state.rag_chain is None:
    st.info("👈 Select or upload a PDF from the sidebar and click **'Ingest & Build Index'** to begin.")
else:
    st.write(f"**Active Document:** `{st.session_state.active_doc}`")

    #Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    #Chat Input
    if user_prompt := st.chat_input("Ask a question about the document..."):
        #Display user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        #Run inference through the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                response = st.session_state.rag_chain.invoke(user_prompt)
                st.markdown(response)

        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})