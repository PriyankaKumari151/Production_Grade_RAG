import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

print("Loading document...\n")

#1 PARSING: Extract raw text from the PDF
# Using a relative path with forward slashes (best practice)
loader = PyPDFLoader("PFMEA-DFMEA-Documents/PFMEA_Rev_01.pdf")
documents = loader.load()
# print(documents[0].metadata)


#2 CHUNKING: Split text blindly by character count
print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n","\n"," ",""]
)

chunks = text_splitter.split_documents(documents)


#3. EMBEDDING & INDEXING: Convert chunks to vectors locally and store in Chroma
print("Embedding and Indexing...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory="./chroma_db"
)

#4. RETRIEVAL SETUP: Configure database to return the top 4 closest chunks
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

#5 PROMPT ORCHESTRATION: Create a basic template for the LLM
template = """Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

print(type(prompt))

#6. GENERATION: Define the free Groq LLM and build the chain
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0
)

rag_chain = (
    {
        "context": retriever, "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

#7. EXECUTE QUERY
user_query = "What is the recommended action associated with the process Enclosure Tray Stamping (Al 5083-H111)"
print(f"\nQuerying: {user_query}")
response = rag_chain.invoke(user_query)

print("\nResponse:")
print(response)
