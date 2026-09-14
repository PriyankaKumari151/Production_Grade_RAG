# Production-Grade RAG Systems: Version 0.0 (The Foundation)

Welcome to Version 0.0 of the Production-Grade RAG series. This repository establishes the baseline architecture for a Retrieval-Augmented Generation (RAG) pipeline designed for automotive manufacturing documents (PFMEAs and Control Plans).

Version 0.0 implements a **Naive RAG** approach. The primary goal of this version is to stand up the core infrastructure—getting documents loaded, vectorized, stored, and queried—using an entirely free, open-source stack. By starting with naive parsing, this version explicitly demonstrates the critical data-loss issues that occur when complex engineering tables are processed with basic text splitters.

---

## Architecture Diagram

![Version 0.0 Architecture Diagram](../assets/RAG-V0-architecture.png)
_(Note: Replace the path above with the link to your generated architecture diagram)_

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

# Pros

## Zero Infrastructure Cost

The pipeline utilizes:

- Open-source embedding models
- Local vector storage
- Free-tier cloud inference

This makes Version 0.0 suitable for experimentation and prototyping without requiring expensive infrastructure.

---

## Rapid Prototyping

A naive RAG architecture can be deployed quickly with relatively little code.

This makes it useful for:

- Proof-of-concept development
- RAG experimentation
- Learning retrieval pipelines
- Establishing a baseline for future versions

---

## Partial Data Privacy

Embedding generation happens locally using Hugging Face.

Therefore, proprietary engineering documents do **not need to be sent to a third-party embedding API** during the indexing process.

> However, retrieved document context is eventually sent to the cloud LLM for generation, so this should not be considered a fully private or air-gapped architecture.

---

# Limitations

Version 0.0 is intentionally naive.

Its purpose is not to provide production-grade accuracy, but rather to establish a baseline and expose the problems that need to be solved in later versions.

## 1. Destruction of Tabular Structure

`PyPDFLoader` extracts text from PDFs without understanding the semantic structure of engineering tables.

For example, a PFMEA table may contain relationships such as:

```text
Process Step
     │
     ├── Failure Mode
     │       │
     │       ├── Effect
     │       ├── Severity
     │       └── Cause
     │
     └── Detection Control
```

A naive text extractor may instead produce a flattened sequence of text:

```text
Process Step
Failure Mode
Effect
Severity
Cause
Detection Control
```

The relationships between these fields can be lost.

This can cause the system to incorrectly associate a **Failure Mode** with the wrong **Severity**, **Cause**, or **Detection Control**.

---

## 2. Semantic Severing

`RecursiveCharacterTextSplitter` divides text according to character-based chunk boundaries.

For example:

```text
Chunk 1
────────────────────────
Process Step: Welding
Failure Mode: Incomplete
...
```

```text
Chunk 2
────────────────────────
weld penetration
Severity: 8
Detection Control: Visual
...
```

A critical engineering record can therefore be split across multiple chunks.

The retriever may retrieve only one half of the information, resulting in incomplete context.

---

## 3. High Hallucination Risk

Because the LLM receives flattened and potentially fragmented text rather than coherent engineering records, it can struggle to correctly establish relationships between:

- Process steps
- Failure modes
- Effects
- Causes
- Severity ratings
- Occurrence ratings
- Detection ratings
- Detection controls

This increases the risk of incorrect or hallucinated answers.

---

## 4. Lack of Provenance

Version 0.0 does not retain precise document coordinates such as:

- Page numbers
- Bounding boxes
- Table coordinates
- Cell coordinates
- Original row/column relationships

Consequently, the system cannot reliably provide visual citations pointing back to the exact location of an answer in the original PDF.

---

# Why Version 0.0 Matters

The purpose of Version 0.0 is to establish a **baseline**.

A simple RAG pipeline can appear to work extremely well when tested against clean text documents.

However, automotive manufacturing documents are often highly structured and table-heavy.

For example:

```text
                    PFMEA
                      │
          ┌───────────┴───────────┐
          │                       │
    Process Information       Risk Analysis
          │                       │
     Process Step             Severity
     Function                  Occurrence
     Requirement               Detection
          │                       │
          └───────────┬───────────┘
                      │
                Control Plan
```

A naive text-based pipeline can destroy these relationships during PDF extraction and chunking.

Therefore:

> **Retrieval quality is ultimately constrained by the quality and structure of the data being retrieved.**

Version 0.0 makes these limitations visible so that future versions can address them systematically.

---

# Future Scope — Version 1.0

Version 0.0 establishes the baseline.

Version 1.0 will transition from **Naive RAG** toward a more modular and production-oriented architecture.

The primary focus areas are:

## 1. Vision-Based Parsing

Replace basic PDF text extraction with a more structure-aware parsing approach such as:

```text
LlamaParse
```

The goal is to preserve:

- Table structure
- Rows
- Columns
- Headers
- Cell relationships
- Page-level information

---

## 2. Structured JSON Chunking

Instead of splitting documents arbitrarily based on character count, documents will be transformed into structured representations.

For example:

```json
{
  "process_step": "Welding",
  "failure_mode": "Incomplete weld penetration",
  "effect": "Structural weakness",
  "severity": 8,
  "cause": "Incorrect welding parameters",
  "detection_control": "Visual inspection"
}
```

This allows retrieval to operate on **semantic engineering records** rather than arbitrary text fragments.

---

## 3. Cross-Encoder Reranking

Version 1.0 will introduce a second-stage retrieval process:

```text
User Query
    │
    ▼
Vector Retrieval
    │
    ▼
Top-K Candidates
    │
    ▼
Cross-Encoder Reranker
    │
    ▼
High-Relevance Context
    │
    ▼
LLM
```

The reranker will evaluate the relationship between the query and retrieved documents more precisely, filtering out low-relevance chunks before they reach the LLM.

---

# Version Roadmap

| Version    | Architecture   | Primary Goal                                                     |
| ---------- | -------------- | ---------------------------------------------------------------- |
| **V0.0**   | Naive RAG      | Establish baseline pipeline                                      |
| **V1.0**   | Modular RAG    | Preserve structure and improve retrieval                         |
| **Future** | Production RAG | Accuracy, provenance, evaluation, observability, and scalability |

---

# Key Takeaway

Version 0.0 intentionally demonstrates a fundamental problem with applying generic RAG tutorials to complex engineering documents:

> **The biggest problem is often not the LLM — it is the loss of document structure before retrieval even begins.**

By establishing a naive baseline first, subsequent versions can measure how much each architectural improvement contributes to retrieval accuracy and answer quality.

---

## License

Add your project license here, for example:

```text
MIT License
```

if applicable.
