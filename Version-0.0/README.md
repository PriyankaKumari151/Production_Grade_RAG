# Production-Grade RAG Systems: Version 0.0 (The Foundation)

Welcome to Version 0.0 of the Production-Grade RAG series. This repository establishes the baseline architecture for a Retrieval-Augmented Generation (RAG) pipeline designed for automotive manufacturing documents (PFMEAs and Control Plans).

Version 0.0 implements a **Naive RAG** approach. The primary goal of this version is to stand up the core infrastructure—getting documents loaded, vectorized, stored, and queried—using an entirely free, open-source stack. By starting with naive parsing, this version explicitly demonstrates the critical data-loss issues that occur when complex engineering tables are processed with basic text splitters.

---

## Architecture Diagram

![Version 0.0 Architecture Diagram](../assets/RAG-V0-architecture.png)
*(Note: Replace the path above with the link to your generated architecture diagram)*

The Version 0.0 pipeline follows a linear, single-pass ingestion and retrieval flow:

1. **Document Loading:** Ingests raw PDFs via standard text extraction.
2. **Text Chunking:** Slices text at fixed character intervals.
3. **Embedding:** Translates chunks into 384-dimensional mathematical vectors.
4. **Vector Storage:** Indexes the vectors and text payloads in a local database.
5. **Prompt Orchestration:** Injects retrieved context and the user query into a static template.
6. **Generation:** Streams the prompt payload to a cloud LLM for inference.

---

## Tech Stack & Tools Used

This version prioritizes speed, zero-cost tooling, and local data privacy for the embeddings.

- **Orchestration Framework:** `LangChain` (v0.2+)[cite: 2]
- **Document Parsing:** `PyPDFLoader` (Extracts raw, unformatted text coordinates)[cite: 2]
- **Chunking Strategy:** `RecursiveCharacterTextSplitter` (Blind character counting)[cite: 2]
- **Embedding Model:** `HuggingFaceEmbeddings` (`BAAI/bge-small-en-v1.5`) - Runs 100% locally on CPU to protect proprietary engineering data[cite: 2].
- **Vector Database:** `Chroma DB` - Runs locally, storing embeddings in a persistent directory without requiring cloud infrastructure[cite: 2].
- **LLM Inference:** `Groq` (`openai/gpt-oss-20b`) - Delivers lightning-fast, free cloud inference via custom LPU chips[cite: 2].

---


# Setup Instructions

## 1. Clone the Repository

Clone the repository and navigate to the project root:

```bash
git clone <your-repository-url>
cd Production-grade-RAG
```

---

## 2. Create the Virtual Environment

Create a dedicated Python virtual environment:

### Windows

```bash
python -m venv rag-pipeline-venv
```

### macOS / Linux

```bash
python3 -m venv rag-pipeline-venv
```

---

## 3. Activate the Virtual Environment

### Windows

```bash
rag-pipeline-venv\Scripts\activate
```

### macOS / Linux

```bash
source rag-pipeline-venv/bin/activate
```

After activation, your terminal should display something similar to:

```text
(rag-pipeline-venv)
```

---

## 4. Install Dependencies

Navigate to the project root and install the dependencies listed in the Version 0.0 requirements file:

```bash
pip install -r Version-0.0/requirements.txt
```

Alternatively, from inside the `Version-0.0` directory:

```bash
pip install -r requirements.txt
```

### Verify the Environment

You can verify that Python is using the intended virtual environment:

```bash
where python
```

On Windows, the output should point to something similar to:

```text
...\Production-grade-RAG\rag-pipeline-venv\Scripts\python.exe
```

---

## 5. Configure Environment Variables

Create a `.env` file at the **project root**:

```text
Production-grade-RAG/
├── .env
├── Source-Documents/
├── Version-0.0/
│   ├── app1.py
│   └── requirements.txt
└── assets/
    └── RAG-V0-architecture.png
```

Add your Groq API key:

```env
GROQ_API_KEY="your_free_groq_api_key_here"
```

> **Important:** Never commit your `.env` file or API keys to Git.

Add `.env` to your `.gitignore`:

```gitignore
.env
```

---

## 6. Add Source Documents

Place your target PDF documents inside the root-level `Source-Documents` directory.

For example:

```text
Source-Documents/
└── PFMEA_Rev_01.pdf
```

The documents can include automotive manufacturing artifacts such as:

- PFMEAs
- Control Plans
- Process documentation
- Manufacturing quality documents

---

## 7. Run the Pipeline

Navigate to the Version 0.0 directory:

```bash
cd Version-0.0
```

Run the application:

```bash
python app1.py
```

---

# Project Structure

The expected project structure is:

```text
Production-grade-RAG/
│
├── .env
├── .gitignore
├── README.md
│
├── Source-Documents/
│   └── PFMEA_Rev_01.pdf
│
├── assets/
│   └── RAG-V0-architecture.png
│
├── Version-0.0/
│   ├── app1.py
│   └── requirements.txt
│
└── rag-pipeline-venv/
```

> The virtual environment directory should generally **not be committed to Git**.

Add it to `.gitignore`:

```gitignore
rag-pipeline-venv/
__pycache__/
.env
*.pyc
```

---
