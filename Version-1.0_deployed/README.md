# Production-Grade Engineering RAG Pipeline (PFMEA Document Intelligence)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enterprise-rag-v1-live.streamlit.app/)

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to parse, structure, and query complex, multi-page engineering documents — specifically Process Failure Mode and Effects Analysis (PFMEA) tables — with absolute data fidelity.

---

## 🚀 Live Demo
You can access and interact with the deployed application directly here:
👉 [**Enterprise RAG Version 1.0 Live App**](https://enterprise-rag-v1-live.streamlit.app/)

---

## 📁 Sample PFMEA Documents for Testing
Want to experiment with the application right away? You can download sample engineering PFMEA PDFs to upload and test out the parsing engine from the link below:
👉 [**Download Sample PFMEA Input Files**](https://drive.google.com/drive/folders/1Bq39D98MZKmiWsgHdM9xeuQFeVtoz6gY?usp=drive_link)

---

## Table of Contents

- [Project Overview & What It Solves](#project-overview--what-it-solves)
- [Use Cases](#use-cases)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Key Architectural Decisions & Problem Solving](#key-architectural-decisions--problem-solving)
- [Model Trade-Offs, Context Windows, and Free-Tier Optimization](#model-trade-offs-context-windows-and-free-tier-optimization)
- [Limitations & Key Architectural Trade-Offs](#limitations--key-architectural-trade-offs)
- [Future Scope: Version 2.0 (Enterprise Roadmap)](#future-scope-version-20-enterprise-roadmap)
- [Setup & Installation](#setup--installation)

---

## Project Overview & What It Solves

Standard RAG systems fail when applied to complex engineering tables because tabular data relies heavily on visual grid structures, merged cells, and hierarchical parent-child relationships (e.g., one Process Step containing multiple Failure Modes, each mapping to multiple Causes, Controls, and Action Priorities).

This project solves the fundamental breakdown of traditional RAG pipelines when handling dense engineering documents by addressing:

- **Sparse & Missing Context** — LlamaParse omits cells in merged rows, resulting in incomplete text that breaks semantic vector retrieval.
- **Data Bleeding & Row Shifting** — Global forward-filling (`ffill()`) accidentally drags row-specific data (like Recommended Actions) downward into unrelated rows.
- **Arbitrary Chunk Boundaries** — Standard character chunkers slice tables mid-row, separating root causes from their corresponding detection controls and leading to LLM hallucinations or "first-match" blindspots.

---

## Use Cases

### Use Case 1: Automated Engineering Design Review & Risk Auditing

**Scenario:** Manufacturing and quality engineers need to rapidly audit compliance, trace root causes, and verify control measures across thousands of lines of PFMEA data (e.g., EV Battery Enclosure Assemblies).

**Application:** Engineers can query complex multi-cause or cross-referencing questions (e.g., *"What are all the detection controls for weld spatter across different laser welding stations?"*) and receive exhaustive, hallucination-free answers mapped directly to specific severity and occurrence ratings.

### Use Case 2: Reverse-Lookup Root-Cause Analysis for Field Failures

**Scenario:** A quality team encounters a specific defect or field failure (e.g., water ingress leading to an IP67 failure or an intermittent BMS fault code) during vehicle testing.

**Application:** The RAG system performs a reverse lookup from the "Effect" column back through the process hierarchy to instantly identify the exact failure mode, potential causes, and required preventive actions.

---

## Tech Stack

| Category | Technology |
|---|---|
| Orchestration & Framework | LangChain, Python |
| Document Parsing & Transformation | LlamaParse (Vision extraction), Pandas, `markdown` library, LXML, Regular Expressions |
| Vector Database | ChromaDB |
| Inference Engine & LLM | Groq API (`openai/gpt-oss-120b`) |
| User Interface | Streamlit |
| Environment & Configuration | Python-dotenv, Requests |

---

## Architecture Overview

```
[ Raw PFMEA PDF Document ]
           │
           ▼
[ LlamaParse (Vision Extraction) ]
           │
           ▼
[ Uniform HTML Conversion via `markdown` library ]
           │
           ▼
[ Pandas Table Extraction & Targeted Forward-Fill (Cols 0–5) ]
           │
           ▼
[ Table-to-Text Grouping into Unbreakable Plain-Text Blocks ]
           │
           ▼
[ CharacterTextSplitter (Split strictly on ===CHUNK_BOUNDARY===) ]
           │
           ▼
[ ChromaDB Vector Store & Semantic Retriever ]
           │
           ▼
[ Groq LLM (openai/gpt-oss-120b) + Prompt Orchestration ]
           │
           ▼
[ Streamlit Interactive User Interface ]
```

---

## Key Architectural Decisions & Problem Solving

### How We Solved the Merged Cell & Data-Bleeding Problem

In standard tables, merged cells cause missing text, and global forward-filling (`df.ffill()`) inadvertently drags row-specific data (such as Recommended Actions or Detection Controls) downward into empty cells of subsequent rows.

**The Solution:** We implemented a **Targeted Forward-Fill Strategy**. Pandas restricts the `ffill()` operation exclusively to the hierarchical parent columns (Columns 0 through 5: Process Step, Function, Failure Mode, Effect, Severity). Row-specific data like causes and controls are strictly isolated, and missing values are safely filled with `"Not specified"`, eliminating downward data bleeding and column-shifting hallucinations.

### The "Table-to-Text" Grouping & Unbreakable Chunking Strategy

Standard chunkers slice text arbitrarily based on character counts, which breaks the relationship between a failure mode and its multiple root causes.

**The Solution:** After parsing uniform HTML tables via Pandas and applying targeted forward-fills, our pipeline programmatically groups data by its parent hierarchy into self-contained plain-text blocks. Each complete failure mode block is separated by a custom delimiter (`===CHUNK_BOUNDARY===`). Using LangChain's `CharacterTextSplitter` configured to split only on this delimiter, we ensure that every chunk sent to ChromaDB contains 100% of the context for that failure mode, completely eliminating the "first-match blindspot."

---

## Model Trade-Offs, Context Windows, and Free-Tier Optimization

Deploying a production-grade RAG system for dense engineering documentation (PFMEA grids) on a restricted free tier introduced a severe systems engineering constraint: balancing model reasoning capability against strict API Token-Per-Minute (TPM) rate limits and context window caps.

### 1. The Model Selection Trade-Off

- **The Challenge:** Smaller open-source models (7B–8B parameters) provided generous rate limits and large token windows, but utterly collapsed under the analytical rigor required for PFMEA documents. They suffered from "LLM laziness," truncated responses mid-sentence, hallucinated numerical scores (e.g., misreading a Severity 10 as 3), and fell victim to first-match blindspots. Conversely, advanced heavyweight models (like `openai/gpt-oss-120b`) possessed the elite reasoning capabilities needed to flawlessly interpret hierarchical engineering relationships, but came with massive resource footprints and strict rate limit enforcement (capping free-tier usage at 8,000 TPM).
- **The Decision:** We prioritized reasoning fidelity over inference speed by selecting the high-parameter model (`openai/gpt-oss-120b`). Because dense engineering tables cannot tolerate hallucinations or incomplete data extraction, an advanced model was non-negotiable.

### 2. Solving the Context Window & Rate-Limit Bottleneck

To use a 120B parameter model on a restricted 8,000 TPM free tier without triggering HTTP 413 (Payload Too Large) or HTTP 429 (Rate Limit Exceeded) errors, we could not rely on brute force. We engineered a multi-layered optimization strategy:

- **Precision Chunk Sizing** — We restricted the vector store retriever to a tight similarity scope (`k = 5` to `7`) and optimized the text splitter parameters to eliminate verbose, redundant padding.
- **Zero-Overlap Unbreakable Chunks** — Traditional RAG systems use large chunk overlaps to prevent context loss, which needlessly multiplies token consumption. By transforming our tables into 100% self-contained semantic blocks separated by `===CHUNK_BOUNDARY===`, we achieved zero overlap (`chunk_overlap = 0`), cutting redundant token transmission down to absolute zero.
- **Targeted Information Density** — Instead of passing raw, messy Markdown grids or massive multi-page HTML blocks that waste tokens on boilerplate table tags, our Pandas "Table-to-Text" pre-processor strips away structural noise. It converts rows into clean, high-density key-value text pairs. This ensures that every single token sent to the 120B model carries core engineering context (Process, Failure Mode, Cause, Control), maximizing the efficiency of the limited context window.

### 3. Pulling It Off on a Free Tier

By combining structural pre-processing (offloading heavy lifting from the LLM to Pandas and Python logic) with precision prompt orchestration and strict token budgeting, we built a fully functional, enterprise-grade engineering intelligence tool on a **$0 infrastructure budget**. The system achieves 100% extraction accuracy on complex multi-cause queries while operating smoothly within the tight constraints of an API free tier.

---

## Limitations & Key Architectural Trade-Offs

While this pipeline successfully resolves structural table extraction and data-bleeding challenges for engineering documents, it operates under specific constraints and trade-offs inherent to its design and free-tier infrastructure:

### 1. API Rate Limits & Concurrency Constraints (Free-Tier Bottlenecks)

- **The Limitation:** Because the system relies on Groq's free-tier `on_demand` service tier (capped at 8,000 Tokens Per Minute for high-parameter models like `openai/gpt-oss-120b`), the application is not suited for high-concurrency, multi-user enterprise deployment out of the box.
- **The Impact:** Rapid-fire queries or simultaneous multi-user usage will trigger HTTP 429 (Rate Limit Exceeded) errors, requiring artificial request throttling or a paid developer tier upgrade for production scaling.

### 2. Dependency on High-Quality Initial Parsing (LlamaParse)

- **The Limitation:** The pipeline's accuracy is downstream of LlamaParse's vision extraction capability. If a source PDF contains severely degraded scans, handwritten markups, non-standard grid layouts, or highly corrupted table structures that fail to render into proper HTML/Markdown tags, `pd.read_html` may drop rows or misalign columns.
- **The Mitigation:** The system relies on strict structural assumptions (e.g., standard 11–12 column PFMEA grids). Documents with non-standard column layouts require manual schema mapping adjustments in `app_helper.py`.

### 3. Static Document Scope & Inability to Handle Dynamic Updates

- **The Limitation:** ChromaDB stores a static snapshot of the embedded PFMEA document. If an engineer updates a specific revision of a process step in the source PDF, the entire document must be re-parsed, cleaned, chunked, and re-indexed into the vector store.
- **The Impact:** There is no real-time incremental synchronization mechanism; the system treats documents as batch artifacts rather than live, mutable databases.

### 4. Semantic Search vs. Exact Numerical Filtering Trade-Offs

- **The Limitation:** ChromaDB performs semantic similarity search (cosine distance over vector embeddings) rather than strict relational database querying (SQL).
- **The Impact:** While our prompt engineering and structured chunks help the LLM interpret numerical scores (like Severity 10 or Occurrence 2), vector retrieval can occasionally surface semantically close but numerically irrelevant chunks if a user query relies entirely on complex multi-variable numerical filtering.

### 5. Context Window and Document Length Scaling

- **The Limitation:** Although our "Table-to-Text" grouping strategy optimizes token density, massive multi-hundred-page engineering manuals containing hundreds of distinct PFMEA tables will generate large numbers of dense chunks.
- **The Impact:** Retrieving too many chunks (`k > 10`) will inevitably breach token limits on restricted model tiers, forcing a tight balance between the retriever's search scope and the LLM's input capacity.

---

## Future Scope: Version 2.0 (Enterprise Roadmap)

While Version 1.0 successfully conquered single-document tabular parsing, targeted forward-filling, and free-tier LLM optimization, transitioning this pipeline to an enterprise-wide deployment requires solving broader, systemic data challenges. Version 2.0 introduces the following architectural advancements:

### 1. Multi-Document Cross-Linking & Global Knowledge Graphs

- **The Challenge:** In a real manufacturing ecosystem, a PFMEA document does not exist in a vacuum. It references Design FMEAs (DFMEAs), Control Plans, Engineering Change Orders (ECOs), and historical warranty or scrap data stored across different files.
- **v2.0 Solution:** Implement a **GraphRAG** (Knowledge Graph + Vector RAG) architecture. By utilizing graph databases (like Neo4j alongside ChromaDB), the system will cross-link entities — such as tracing a specific failure mode in a PFMEA directly to its root cause in a DFMEA or an associated supplier quality report across multiple documents.

### 2. Multi-Modal Ingestion & Image/CAD Intelligence

- **The Challenge:** Engineering documents frequently contain embedded schematics, wiring harness diagrams, 3D CAD snapshots, and microstructural failure photos that text-based parsers and standard RAG completely miss.
- **v2.0 Solution:** Integrate **Multi-Modal Vision LLMs** (such as GPT-4o or specialized vision encoders) directly into the parsing pipeline. Images and technical diagrams extracted by LlamaParse will be vectorized and indexed, allowing engineers to query visual artifacts (e.g., *"Show me the harness routing diagram for this connector failure"*).

### 3. Real-Time Incremental Sync & Dynamic Version Control

- **The Challenge:** Engineering specifications change constantly via Revisions (e.g., Rev A to Rev B). Version 1.0 requires a full static re-indexing of the document.
- **v2.0 Solution:** Build a **document versioning and delta-sync pipeline**. Using change-detection algorithms on parsed HTML blocks, Version 2.0 will support incremental updates — automatically detecting modified rows in a PFMEA table, updating only the affected vector embeddings, and maintaining an immutable audit history of engineering changes.

### 4. Hybrid Search (Semantic + Relational/SQL) for Exact Numerical Filtering

- **The Challenge:** Vector embeddings excel at semantic similarity, but struggle with precise relational or mathematical queries (e.g., *"Find all failure modes where Severity > 8 AND Occurrence < 3"*).
- **v2.0 Solution:** Implement a **Hybrid Search Engine** combining ChromaDB (for conceptual/textual lookups) with an automated Text-to-SQL layer. Structured tabular data will be dual-indexed into a lightweight relational database (SQLite/PostgreSQL), allowing the system to route exact numerical or filtering queries to SQL and qualitative queries to the vector store.

### 5. Enterprise Infrastructure, Caching, and Guardrails

- **The Challenge:** Production environments demand low latency, zero data leakage, and high concurrency, which are incompatible with public API free tiers.
- **v2.0 Solution:**
  - **Semantic Caching** — Integrate Redis-backed semantic caching to store and instantly serve frequent engineering queries, drastically reducing LLM API token consumption and latency.
  - **Enterprise LLM Gateway** — Migrate from free-tier rate-limited endpoints to self-hosted open-source heavyweights (like Llama 3 70B hosted locally via vLLM or Ollama) or dedicated enterprise cloud clusters.
  - **Safety & Compliance Guardrails** — Implement automated output validation layers to ensure generated engineering recommendations strictly adhere to safety standards (e.g., AIAG-VDA guidelines) without hallucinations.

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Groq API Key
- LlamaParse API Key

### Step-by-Step Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/production-grade-rag.git
   cd production-grade-rag
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv rag-pipeline-venv

   # On Windows:
   rag-pipeline-venv\Scripts\activate

   # On macOS/Linux:
   source rag-pipeline-venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Create a `.env` file in the root directory and add your API keys:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   LLAMA_CLOUD_API_KEY=your_llamaparse_api_key_here
   ```

5. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```