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

- **Orchestration Framework:** `LangChain` (v0.2+)
- **Document Parsing:** `PyPDFLoader` (Extracts raw, unformatted text coordinates)
- **Chunking Strategy:** `RecursiveCharacterTextSplitter` (Blind character counting)
- **Embedding Model:** `HuggingFaceEmbeddings` (`BAAI/bge-small-en-v1.5`) - Runs 100% locally on CPU to protect proprietary engineering data.
- **Vector Database:** `Chroma DB` - Runs locally, storing embeddings in a persistent directory without requiring cloud infrastructure.
- **LLM Inference:** `Groq` (`openai/gpt-oss-20b`) - Delivers lightning-fast, free cloud inference via custom LPU chips.

---

## Pros and Limitations

### Pros

- **Zero Infrastructure Cost:** Utilizes free-tier APIs and local models.
- **Rapid Prototyping:** Extremely fast to deploy; requires less than 50 lines of code to stand up a functional Q&A bot.
- **Data Privacy (Partial):** By embedding locally with HuggingFace and Chroma, proprietary documents are not sent to third-party APIs during the indexing phase.

### Limitations (The Real-World Challenges)

- **Destruction of Tabular Structure:** `PyPDFLoader` reads left-to-right, ignoring table gridlines. In a PFMEA, this separates the "Process Step" from its "Failure Mode."
- **Semantic Severing:** `RecursiveCharacterTextSplitter` cuts text arbitrarily every 1,000 characters. It frequently splits critical data rows directly in half, leaving orphaned text segments.
- **High Hallucination Risk:** Because the LLM receives mashed, unstructured strings rather than coherent parent-child relationships, it struggles to accurately link a specific manufacturing defect to its correct severity rating or detection control.
- **Lack of Provenance:** The chunks do not retain exact bounding-box coordinates, meaning the system cannot visually cite its sources on the original PDF.

---

## Future Scope (Version 1.0)

Version 0.0 proves that standard tutorials fail in enterprise manufacturing contexts. To achieve production-grade accuracy, Version 1.0 will transition from Naive RAG to **Modular RAG**, tackling these limitations directly:

1.  **Vision-Based Parsing:** Replacing `PyPDFLoader` with `LlamaParse` to extract table gridlines and preserve row integrity.
2.  **Structured JSON Chunking:** Segmenting documents by semantic table rows rather than arbitrary character limits.
3.  **Cross-Encoder Reranking:** Adding a secondary retrieval layer to aggressively filter out low-relevance chunks before prompt assembly.
