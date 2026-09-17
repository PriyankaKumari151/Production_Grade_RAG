import os
import markdown
import uuid
import pandas as pd
import nest_asyncio
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app_helper import process_pfmea_to_chunks

#Apply asyncio patch (Required for LlamaParse to run inside streamlit)
nest_asyncio.apply()

#-------------------------------------------------------
#Page Configuration & Environment Setup
#--------------------------------------------------------
st.set_page_config(
    page_title="Production RAG - Version 1.0",
    page_icon="👁️",
    layout="wide"
)

#------------------------------------------------
# Streamlit Interface 
#----------------------------------------------------
st.title("👁️ Production-Grade RAG: Version 1.0")
st.caption("Architecture: LlamaParse (Vision) | Semantic Markdown Chunking | ChromaDB | Groq")


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

#-----------------------------------------------------
# Cached Resources
#----------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model (bge-small-en-v1.5)...")
def load_embedding():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner="Connecting to Groq LPU inference engine...")
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        api_key=groq_api_key
    )

#----------------------------------------------------
# Pipeline Function (v1.0 Upgrades)
#----------------------------------------------------
def process_pdf_v1(file_path: str, embeddings):
    """
    v1.0 Pipeline:
    1. Vision Parsing: LlamaParse extracts documents as structured Markdown (preserves tables)
    2. Semantic Chunking: Split by Markdown headers first, then recursively if needed.
    3. Indexing: Store in Chroma with a unique collection UUID.
    """

    #1. VISION PARSING (LlamaParse)
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        premium_mode=True,
        parsing_instruction="Extract all tables precisely as they appear. Do not merge cells visually. If a cell spans multiple rows in the document, leave the cell blank in the subsequent rows.",
        verbose=True
    )

    #Extract data (Returns LlamaIndex Document objects)
    llama_docs = parser.load_data(file_path)

    #Combine the parsed pages into a single Markdown string
    raw_markdown = "\n\n".join([doc.text for doc in llama_docs])

    # ---------------------------------------------------------
    # THE V1.1 FIX: Repair the merged cells before chunking!
    # ---------------------------------------------------------
    full_structured_text = process_pfmea_to_chunks(raw_markdown)

    #NEW DEBUGGING BLOCK: Display the raw Markdown in the UI
    with st.expander("👀 View Raw LlamaParse Markdown Output", expanded=False):
        st.code(full_structured_text, language="markdown")

    #2. SEMANTIC CHUNKING
    text_splitter = CharacterTextSplitter(
        separator="===CHUNK_BOUNDARY===",
        chunk_size=8000, # Set high enough to safely hold the largest possible chunk
        chunk_overlap=0, # Overlap is now 0 because each block is 100% self-contained
        is_separator_regex=False
    )
    
    # LangChain splitters expect a list of documents or strings
    chunks = text_splitter.create_documents([full_structured_text])

    # Remove empty chunks created by trailing separators
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        st.error("⚠️ No text could be extracted. Please check the PDF.")
        st.stop()
    
    #3. EMBEDDING & INDEXING (Isolated Collection)
    unique_collection_name = f"collection_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=unique_collection_name
    )

    #4. RETRIEVER SETUP
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )

    return retriever, len(llama_docs), len(chunks)

def build_rag_chain(retriever, llm):
    # 5. PROMPT ORCHESTRATION
    template = """You are an expert manufacturing engineering assistant analyzing PFMEA documents.
    
     Use the following retrieved context to answer the user's question. 
    
     CRITICAL INSTRUCTIONS:
     1. EXHAUSTIVE EXTRACTION: If a Failure Mode spans multiple rows, or has multiple Causes/Controls, you MUST list ALL of them. Do not stop at the first one you find. Scan the entire retrieved context.
     2. Read the columns carefully: Do not confuse a "Cause" with an "Effect" or a "Failure Mode".
     3. Only use the provided context. If the document does not mention it, do not guess or say "in practice".
    
     Context: {context}

     Question: {question}

     Answer:"""""
    
    prompt = PromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever, "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

#--------------------------------------------------------
#Streamlit Interface
#---------------------------------------------------------

#Verify API Keys
if not groq_api_key or not llama_api_key:
    st.error("⚠️ `GROQ_API_KEY` OR `LLAMA_CLOUD_API_KEY` missing from `.env`.")
    st.stop()

embeddings = load_embedding()
llm = load_llm()

#Intialize Session States
if "messages" not in st.session_state:
    st.session_state.messages =[]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

#Sidebar: Document Management
with st.sidebar:
    st.header("📂 Document Selection")

    uploaded_file = st.file_uploader("Upload an Engineering PDF (PFMA, MAUDE, etc.)", type=["pdf"])
    selected_pdf_path = None
    display_name = None

    if uploaded_file is not None:
        display_name = uploaded_file.name
        selected_pdf_path = "temp_uploaded_v1.pdf"

        #Safely read memory without OS pointer issues
        with open(selected_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    #Ingestion Trigger Button
    if st.button("Extract Tables & Build Index", type="primary", use_container_width=True):
        if not selected_pdf_path:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("LlamaParse Vision API is reading the document... This may take a minute."):
                retriever, num_pages, num_chunks = process_pdf_v1(selected_pdf_path, embeddings)
                st.session_state.rag_chain = build_rag_chain(retriever, llm)
                st.session_state.active_doc = display_name
                st.session_state.messages = []
                st.success(f"Successfully processed **{num_pages} pages** into **{num_chunks} semantic chunks**!")
    
    st.markdown("---")
    st.markdown("### ✨ v1.0 Upgrades")
    st.success("✔️ **Vision Parsing:** Tables extracted as perfect Markdown grids.\n\n✔️ **Semantic Chunking:** Text is split logically by document header, keeping rows together.")

#---------------------------------------------------------
# Chat Interface
#-----------------------------------------------------------
if st.session_state.rag_chain is None:
    st.info("👈 Upload a PDF containing complex tables and click **'Extract Tables  & Build Index'**.")
else:
    st.write(f"**Active Document:** `{st.session_state.active_doc}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if user_prompt := st.chat_input("Ask a question about the document data..."):
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving semantic chunks and generating answer..."):
                response = st.session_state.rag_chain.invoke(user_prompt)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

