# Multilingual Clinical & Academic Document Retrieval Assistant
> **Course Project:** AI Essentials: Theory and Applications / Natural Language Processing  
> **Architecture:** Hybrid RAG (pgvector + multilingual-e5) + Custom Medical BioBERT NER + OCR Screen Capture + DeepSeek / Ollama / Gemini Reasoning

---

## 📌 Executive Summary

Modern clinical and research workflows require rapid synthesis of information across dense, heterogeneous documents (PDFs, clinical trials, academic guidelines, DOCX). Standard keyword search fails to capture medical synonyms and conceptual nuance, while pure cloud LLMs frequently hallucinate and lack grounded verification.

This project delivers an end-to-end **Multilingual Document Intelligence & Retrieval Assistant** combining:
1. **Semantic Vector Search**: High-dimensional embeddings (`intfloat/multilingual-e5-base`) with PostgreSQL + `pgvector`.
2. **Specialized Information Extraction**: Custom fine-tuned **BioBERT** for clinical Named Entity Recognition (NER) trained on the HUMADEX medical corpus (BIOES tagging: `PROBLEM`, `TREATMENT`, `TEST`), with cross-lingual fallback (Kazakh & Russian XLM-RoBERTa).
3. **Dual Interactive Interfaces**:
   - **Streamlit Web Platform (`app.py`)**: For full document corpus ingestion, vectorization, and multi-turn conversational RAG.
   - **PySide6 Desktop Assistant (`desktop_assistant/`)**: Global hotkey-triggered (`Ctrl+Alt+Space`) screen capture with Tesseract OCR, real-time query distillation via **DeepSeek Flash**, instant document retrieval, and contextual floating chat.
4. **Hybrid LLM Pipeline**: Cloud reasoning via **DeepSeek Flash** / **Gemini** and fully offline, privacy-preserving inference via **Ollama Dolphin 8B**.

---

## 🏗️ System Architecture

```text
       [ User Desktop Screen / PDFs / DOCX / HTML ]
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
┌───────────────────────┐             ┌───────────────────────┐
│ Desktop Screen Capture│             │ Multi-format Ingestion│
│ (PySide6 + mss + OCR) │             │ (PyMuPDF, docx, HTML) │
└───────────┬───────────┘             └───────────┬───────────┘
            │ OCR Text                            │ Chunking (400 tok, 50 ovlp)
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│ DeepSeek Flash Query  │             │ Multilingual NER      │
│ Distillation API      │             │ (BioBERT / XLM-R)     │
└───────────┬───────────┘             └───────────┬───────────┘
            │ Short Query                         │ Entities & Metadata
            ▼                                     ▼
    ┌───────────────────────────────────────────────────┐
    │     Embedding Model (multilingual-e5-base)        │
    └─────────────────────────┬─────────────────────────┘
                              ▼
    ┌───────────────────────────────────────────────────┐
    │       PostgreSQL Vector Store (pgvector)          │
    │   Cosine Distance Top-K Semantic Similarity Search│
    └─────────────────────────┬─────────────────────────┘
                              ▼
    ┌───────────────────────────────────────────────────┐
    │           Retrieved Context & Source Docs         │
    └─────────────────────────┬─────────────────────────┘
                              ▼
    ┌───────────────────────────────────────────────────┐
    │        Grounded LLM Reasoning & Synthesis         │
    │    • DeepSeek Flash (Cloud API)                   │
    │    • Ollama Dolphin 8B (Local Privacy Mode)       │
    │    • Gemini Pro (Cloud Fallback)                  │
    └─────────────────────────┬─────────────────────────┘
                              ▼
    ┌───────────────────────────────────────────────────┐
    │             Interactive User Interfaces           │
    │   • Streamlit Web UI  • PySide6 Floating Overlay  │
    └───────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

- **Domain-Specific Sequence Labeling**: Fine-tuned BioBERT achieves **0.888 F1 score** on medical entities, outperforming off-the-shelf general NER baselines (0.642 F1).
- **Multilingual Support**: Supports English clinical texts, Kazakh (`xlm-roberta-large-ner-kazakh`), and Russian (`xlm-roberta-base-ner-hrl`).
- **Instant Desktop Workflow**: Press `Ctrl+Alt+Space` over any lecture slide, clinical note, or paper to instantly pull matching local reference literature without switching windows.
- **Privacy & Grounding**: LLM answers cite verified local document chunks and file paths to completely mitigate hallucinations. Local Ollama mode ensures zero data egress when handling sensitive clinical notes.

---

## 📂 Project Structure

```
.
├── app.py                         # Streamlit web application interface
├── main.py                        # Core RAG pipeline, LLM orchestration & CLI flow
├── document_load_pipe.py          # Document chunking and embedding ingestion pipeline
├── deepseek_client.py             # DeepSeek Flash API client (root re-export)
├── requirements.txt               # Pinned dependencies
├── documentation.md               # Original course documentation
│
├── desktop_assistant/             # PySide6 Desktop Assistant Module
│   ├── main.py                    # Qt runtime, event loop, and overlay triggers
│   ├── deepseek_client.py         # DeepSeek Flash client (OpenAI-compatible)
│   ├── gemini_client.py           # Gemini client (cloud alternative)
│   ├── hotkey_service.py          # Global hotkey listener (Ctrl+Alt+Space) & OCR pipe
│   ├── ocr_service.py             # Tesseract OCR extraction service
│   ├── overlay.py                 # Transparent floating results & chat GUI
│   ├── retrieval.py               # Vector search interface for desktop client
│   ├── screenshot.py              # High-performance screen capture (mss)
│   └── config.py                  # Runtime configuration and environment parsing
│
├── models/                        # ML/DL Model Weights & Wrappers
│   ├── encoder.py                 # multilingual-e5 transformer embedding wrapper
│   ├── ner_med.py                 # Fine-tuned BioBERT medical NER pipeline
│   ├── ner.py                     # Multilingual NER router (EN, KK, RU)
│   ├── ner.ipynb                  # BioBERT fine-tuning, training & evaluation notebook
│   └── biobert-humadex-ner-output/ # Fine-tuned BioBERT model weights & config
│
├── database/                      # Database & pgvector Operations
│   ├── database_setup.py          # PostgreSQL schema initialization & table creation
│   └── database_search.py         # Cosine similarity vector search queries
│
├── parsers/                       # Multi-format Document Extractors
│   ├── dispatcher.py              # File format router
│   ├── parser_pdf.py              # PyMuPDF extractor with clean formatting
│   ├── parser_docx.py             # Microsoft Word extractor
│   ├── parser_html.py             # HTML extractor
│   ├── parser_txt_md.py           # Plain text / Markdown extractor
│   └── directory_analyzer.py      # Recursive folder scanner & metadata analyzer
│
├── test/                          # Evaluation and sample PDF corpus
│
└── documentation/                 # Full Defense & Project Documentation (100 pts)
    ├── 01_problem_and_dataset.md  # Problem formulation, dataset, EDA & splits
    ├── 02_methods_and_architecture.md # ML/DL methods, embeddings, pgvector, LLMs
    ├── 03_experiments_and_metrics.md # Experimental results, comparisons & error analysis
    ├── 04_engineering_and_reproducibility.md # Engineering quality, setup & reproduction
    ├── 05_responsible_ai_and_limitations.md # Ethics, safety, bias & boundaries
    ├── 06_team_contributions_and_defense_qa.md # Team roles matrix & defense Q&A guide
    └── presentation.md            # Complete 12-slide presentation text & speech script
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python**: Version 3.10 to 3.12 (Python 3.12 recommended)
- **PostgreSQL**: With `pgvector` extension enabled
- **Tesseract OCR**: Installed for screen capture text extraction
- *(Optional)* **Ollama**: If using offline local LLMs (`ollama pull dolphin-llama3:8b`)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/fulaura/Relevant-File-Retrieval.git
cd Relevant-File-Retrieval

# Create and activate virtual environment
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Initialization
Ensure PostgreSQL is running and execute:
```powershell
py -3.12 -c "from database import database_setup; database_setup.init_database()"
```

### 4. Configure Environment Variables (Optional)
Defaults are pre-configured, but you can customize via PowerShell:
```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/document_retrieval"
$env:DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 🖥️ Running the Applications

### Option A: Web Application (Streamlit)
Ideal for batch document loading, corpus administration, and web-based research chat:
```powershell
py -3.12 -m streamlit run app.py
```
- Open `http://localhost:8501`
- Ingest documents from the sidebar (point to `test/` directory)
- Chat with the assistant and inspect retrieved context and medical tags.

### Option B: Desktop Assistant (PySide6 Overlay)
Ideal for real-time workflow assistance across any desktop window:
```powershell
py -3.12 -m desktop_assistant.main
```
1. Focus any text on your screen (e.g., lecture slide, medical journal, note).
2. Press **`Ctrl + Alt + Space`**.
3. A floating window displays:
   - Extracted OCR text snippet.
   - Top-K matched documents with similarity scores.
   - Interactive chat dialogue powered by **DeepSeek Flash**.

### Option C: CLI Mode
```powershell
py -3.12 main.py
```

---

## 📊 Summary of Results

| Model / Approach | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **General Baseline (BERT-NER)** | 0.651 | 0.634 | **0.642** | Fails on medical jargon & clinical abbreviations |
| **Few-Shot Prompting (Ollama 8B)** | 0.628 | 0.652 | **0.640** | High latency; boundary instability |
| **Few-Shot Prompting (Gemini Pro)** | 0.702 | 0.718 | **0.710** | Good understanding, but inconsistent span boundaries |
| **Fine-Tuned BioBERT (Ours)** | **0.894** | **0.882** | **0.888** | **Best performance; sharp clinical span accuracy** |

*Detailed metrics, confusion analysis, and ablation studies are documented in [`documentation/03_experiments_and_metrics.md`](documentation/03_experiments_and_metrics.md).*

---

## 📚 Complete Project Documentation

All academic defense deliverables are organized in the [`documentation/`](documentation/) directory:

| Document | Topic | Rubric Alignment |
| :--- | :--- | :---: |
| [**01_problem_and_dataset.md**](documentation/01_problem_and_dataset.md) | Problem formulation, HUMADEX dataset, EDA, splits | Criteria 1 & 2 (22 pts) |
| [**02_methods_and_architecture.md**](documentation/02_methods_and_architecture.md) | BioBERT NER, e5 embeddings, pgvector, DeepSeek/Ollama | Criterion 3 (15 pts) |
| [**03_experiments_and_metrics.md**](documentation/03_experiments_and_metrics.md) | Metrics, evaluation tables, error analysis (20 cases) | Criterion 4 (18 pts) |
| [**04_engineering_and_reproducibility.md**](documentation/04_engineering_and_reproducibility.md) | Engineering design, verification, reproduction guide | Criteria 5 & 6 (20 pts) |
| [**05_responsible_ai_and_limitations.md**](documentation/05_responsible_ai_and_limitations.md) | AI safety, privacy, medical disclaimer, hallucination | Criterion 7 (7 pts) |
| [**06_team_contributions_and_defense_qa.md**](documentation/06_team_contributions_and_defense_qa.md) | Contribution breakdown, anticipated committee Q&A | Criteria 8 & 9 (18 pts) |
| [**presentation.md**](documentation/presentation.md) | Complete 12-slide presentation text & speech script | Criteria 8 & 9 (Defense) |

---

## 👥 Authors & Academic Context
- **Course**: AI Essentials: Theory and Applications / NLP (Final Defense)
- **Institution**: Astana IT University (AITU)
- **License**: MIT
