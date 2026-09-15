import os
import uuid
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Apply asyncio patch (Required for LlamaParse to run inside streamlit)
nest_asyncio.apply()

#-------------------------------------------------------
#Page Configuration & Environment Setup
#--------------------------------------------------------
st.set_page_config(
    page_title="Production RAG - Version 1.0",
    page_icon="👁️"
    layout="wide"
)

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
        model="openai/gpt-oss-20b",
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
        verbose=True
    )

    #Extract data (Returns LlamaIndex Document objects)
    llama_docs = parser.load_data(file_path)

    #Combine the parsed pages into a single Markdown string
    full_markdown_text = "\n\n".join([doc.text for doc in llama_docs])

    #2. SEMANTIC CHUNKING
    # Step A: Split logically by Markdown headers (keeps sections/tables together)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=header_to_split_on,
        strip_headers=False
    )
    md_header_splits = markdown.splitter.split_text(full_markdown_text)

    #Step B: Fallback split for massive sections that still exceed context limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_Splitter.split_documents(ms_header_splits)

    if not chunks:
        st.error("⚠️ No text could be extracted. Please check the PDF.")
        st.stop()
    
    #3. EMBEDDING & INDEXING (Isolated Collection)
    unique_collection_name = f"collection_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embeddings=embeddings,
        collection_name=unique_collection_name
    )

    #4. RETRIEVER SETUP
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return retriever, len(llama_docs), len(chunks)

def build_rag_chain(retriever, llm):
    #5. PROMPT ORCHESTRATION
    template = """Use the following pieces of retrieved context to answer the question.
    If the context contains tables, extract the data carefully across the rows.
    Tf you don't know the answer, just say you don't know.

    Context: {context}

    Qustion: {question}

    Answer:"""
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
if not groq_Api_key or not llama_api_key:
    st.error("⚠️ `GROQ_API_KEY` OR `LLAMA_CLOUD_API_KEY` missing from `.env`.")
    st.stop()

embeddings = load_embeddings()
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

    if uplaoded_file is not None:
        display_name = uploaded_file.name
        selected_pdf_path = "temp_uploaded_v1.pdf"

        #Safely read memory without OS pointer issues
        with open(selected_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    #Ingestion Trigger Button
    if st.button("Extract Tables & Build Index", type="primary", use_conatiner_width=True):
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
    st.success("✔️ **Vision Parsing:** Tables extracted as perfect Markdown grids. \n\n✔️ **Semantic Chunking:** Text is split logically by document header, keeping rows together.")

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
            
