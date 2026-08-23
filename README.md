# University Academic AI & PYQ Intelligence Platform

An end-to-end multi-workspace RAG (Retrieval-Augmented Generation) system and universal intelligence platform designed for processing university exam Previous Year Question (PYQ) papers, extracting structured questions, generating grounded answers, and providing recurrence analysis and study prioritization.

---

## Overview

University exam preparation requires analyzing historical question papers, identifying high-yield topics, and studying grounded, accurate solutions. However, raw academic PDF question papers present significant structural variations—differing layouts, complex subquestion hierarchies, cross-page continuations, and low-resolution scans.

This system solves these challenges by providing:
- **Universal Academic PDF Processing**: Hybrid text extraction combining native PDF parsing, OCR (Tesseract), and geometrical layout analysis to reliably extract questions across diverse university schemes.
- **Source-Grounded RAG Engine**: Vector-based semantic search and grounded answer generation strictly tied to ingested course materials and question papers.
- **PYQ Intelligence Engine**: Automated recurrence classification categorizing questions into exact repeats, semantic/paraphrased repeats, and related topic clusters.
- **Study Prioritization**: Dynamic ranking of topics based on historical paper frequency, weightage distribution, and recurrence trends.
- **Multi-Workspace Isolation**: Complete data and context segregation per academic workspace (e.g., subject, branch, and semester).

---

## Architecture

```text
               +----------------------------------+
               |        Academic PDF Papers       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |    Hybrid Document Extraction    |
               |  (Native Text / OCR / Layout)    |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |     Question Reconciliation      |
               |  (Hierarchies & Cross-Page)      |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   Canonical Question Database    |
               +----------------------------------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
+--------------------------+          +--------------------------+
|  Embeddings & VectorDB   |          | PYQ Intelligence Engine  |
|       (ChromaDB)         |          | (Exact / Semantic /      |
+--------------------------+          |  Related Recurrence)     |
             |                        +--------------------------+
             v                                     |
+--------------------------+                       v
|  RAG Answer Engine       |          +--------------------------+
|  (Source Grounded LLM)   |          | Study Priority Engine    |
+--------------------------+          | (Topic & Question Rank)  |
                                      +--------------------------+
```

---

## Key Features

### 1. Universal Academic PDF Processing
- **Multi-Modal Fallback**: Automatically selects native text extraction, switching to OCR layout analysis for scanned PDFs.
- **Subquestion Preservation**: Preserves question numbering hierarchies (e.g., `Q1(a)(i)`), marks, and section headers.
- **Cross-Page Continuation**: Merges questions split across page boundaries without dropping sub-parts.

### 2. Source-Grounded RAG Pipeline
- **Workspace-Scoped Search**: Hybrid vector retrieval strictly filtered by `workspace_id`.
- **Grounded Verification**: Answers are synthesized with explicit citations referencing the source paper, year, and question number.
- **Multi-Provider Fallback**: Resilient LLM client supporting OpenAI, Gemini, Groq, and OpenRouter with automatic provider failover.

### 3. PYQ Intelligence & Recurrence Analysis
- **Exact Repeats**: Identifies questions with identical wording across different exam years.
- **Semantic Repeats**: Detects paraphrased questions testing identical concepts using semantic signature alignment.
- **Related Topics**: Groups questions belonging to shared subject domains without incorrectly marking them as repeats.

### 4. Study Prioritization
- **Weightage Distribution**: Computes topic importance based on frequency, mark allocation, and multi-year paper coverage.
- **High-Priority Recommendations**: Ranks topics to guide student exam revision strategy.

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Backend Framework** | Python 3.13, FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide React |
| **Vector Store** | ChromaDB |
| **Embeddings & NLP** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **LLM Provider Integration** | OpenAI (`gpt-4o-mini`), Google Gemini, Groq, OpenRouter |
| **PDF Processing & OCR** | PyMuPDF (fitz), pdf2image, pytesseract (Tesseract OCR) |
| **Testing** | Pytest, FastAPI TestClient |

---

## Project Structure

```text
pyqrag/
├── rag/                             # Core Backend Engine
│   ├── answer_engine.py             # Grounded RAG answer synthesis
│   ├── api.py                       # FastAPI application & endpoints
│   ├── config.py                    # Environment & configuration defaults
│   ├── dynamic_ingest.py            # PDF ingestion pipeline
│   ├── evidence_fusion.py           # Evidence weighting & retrieval fusion
│   ├── hybrid_question_extraction.py # Universal PDF extraction engine
│   ├── ocr_layout.py                # Geometrical layout & OCR analysis
│   ├── pyq_intelligence.py          # Recurrence classification engine
│   ├── question_structure.py        # Question hierarchy models
│   ├── vector_store.py              # ChromaDB vector store wrapper
│   └── workspace_db.py              # Workspace DB persistence
├── frontend/                        # React Frontend Application
│   ├── src/
│   │   ├── api/academicApi.js       # Backend API client
│   │   ├── components/              # Sidebar, Header, Modals
│   │   ├── pages/                   # Dashboard, PYQ Intelligence, Study Priority, Ask UI
│   │   └── App.jsx                  # Main application router
│   ├── package.json
│   └── vite.config.js
├── data/                            # Persistent Uploads & PYQ Papers
├── tests/                           # Regression & Integration Test Suite
│   ├── test_workspace_single_source_of_truth.py
│   ├── test_pyq_intelligence_rebuilt.py
│   ├── test_study_priority_engine.py
│   └── test_rag_e2e.py
├── .env.example                     # Environment template
├── .gitignore                       # Professional gitignore rules
├── requirements.txt                 # Backend Python dependencies
└── README.md                        # Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Tesseract OCR (Optional, required for scanned PDF ingestion)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/university-academic-ai.git
cd university-academic-ai
```

### 2. Backend Setup
```bash
# Create Python virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate
# Activate environment (Linux/macOS)
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## Environment Variables

Copy `.env.example` to `.env` in the root directory:

```bash
cp .env.example .env
```

Configure your LLM provider credentials in `.env`:

```env
LLM_PROVIDER=openai
LLM_FALLBACK_PROVIDERS=gemini,groq,openrouter
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2

OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-flash

GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant

OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

*Note: If no LLM API keys are provided, the system operates in deterministic extraction mode, running OCR and PYQ analysis locally.*

---

## Running the Application

### Start Backend API Server
```bash
python -m uvicorn rag.api:app --host 127.0.0.1 --port 8000 --reload
```
The FastAPI backend will run at `http://127.0.0.1:8000`. API documentation is available at `http://127.0.0.1:8000/docs`.

### Start Frontend Development Server
```bash
cd frontend
npm run dev
```
The React frontend UI will run at `http://localhost:5173`.

---

## Production API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check system, LLM provider, and vector store status |
| `GET` | `/workspaces` | List all academic workspaces |
| `POST` | `/workspaces` | Create a new academic workspace |
| `DELETE` | `/workspaces/{workspace_id}` | Delete a workspace and purge its stored vectors |
| `POST` | `/workspaces/{workspace_id}/ingest` | Upload and ingest a syllabus or PYQ PDF |
| `POST` | `/search` | Perform workspace-scoped hybrid vector retrieval |
| `POST` | `/ask` | Generate a grounded answer with source citations |
| `POST` | `/workspaces/{workspace_id}/analyze-pyq` | Run PYQ intelligence & repeat classification |
| `GET` | `/workspaces/{workspace_id}/pyq-questions` | Fetch extracted canonical questions |
| `GET` | `/workspaces/{workspace_id}/pyq-patterns` | Retrieve within-paper & multi-year topic patterns |
| `POST` | `/workspaces/{workspace_id}/study-priority` | Rank exam revision topics by weightage & priority |

---

## PYQ Intelligence Recurrence Logic

The system distinguishes between three levels of question relationship:

1. **Exact Repeat**: Questions sharing identical normalized text and structure across different paper years.
2. **Semantic Repeat**: Questions testing the exact same core concept using paraphrased phrasing, differing variable names, or inverted subquestion clauses.
3. **Related Topic**: Questions sharing a broader subject domain (e.g., *Routing Protocols*) but asking for different mechanisms (e.g., *Distance Vector* vs *Link State*). Related topics are grouped together for study focus but are **never** classified as repeat questions.

---

## Data & Workspace Isolation

All ingested documents, canonical question structures, and vector embeddings are indexed strictly with a `workspace_id`.
- **Query Scoping**: Vector queries enforce a hard filter `{"workspace_id": {"$eq": target_workspace_id}}`.
- **Zero Cross-Leakage**: Questions from Workspace A are never returned during searches or PYQ analysis in Workspace B.

---

## Testing Strategy

Run the automated backend test suite using `pytest`:

```bash
python -m pytest tests/test_workspace_single_source_of_truth.py tests/test_pyq_intelligence_rebuilt.py tests/test_study_priority_engine.py
```

Run frontend production build verification:
```bash
cd frontend
npm run build
```

---

## Security & Guidelines

- **Secrets Management**: Real credentials or `.env` files must never be committed. Always use environment variables or `.env.example`.
- **Deterministic Extraction**: Extraction logic avoids hardcoded university/subject heuristics to maintain universal subject-agnostic performance.
- **Source Traceability**: Every extracted question retains metadata linking back to its original PDF file, year, and page.

---

## License

License: Not specified.
