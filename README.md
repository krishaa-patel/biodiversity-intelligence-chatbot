# 🌿 EcoSignal AI — Evidence-Grounded Environmental Decision Intelligence

The system combines scientific retrieval, multi-metric environmental reasoning, conversation memory, evidence-backed recommendations, and scientific grounding verification.

---

## Project Objective

Environmental problems depend on multiple connected variables such as soil health, rainfall, land use, climate, biodiversity, and human environmental pressures.

This project is designed to behave more like an AI environmental scientist than a generic chatbot.

The system can:

- Understand environmental conditions from natural-language input
- Build a structured environmental profile
- Ask follow-up questions when important information is missing
- Remember information across multiple conversation turns
- Retrieve relevant scientific evidence
- Perform multi-metric environmental reasoning
- Generate evidence-backed recommendations
- Cite scientific source documents and page numbers
- Verify scientific claims against retrieved evidence

---

## System Workflow

User Input  
↓  
Environmental Information Extraction  
↓  
Structured Environmental Profile  
↓  
Conversation Memory  
↓  
Missing Information Check  
↓  
Clarifying Questions if Needed  
↓  
Scientific RAG Retrieval  
↓  
Chroma Vector Database  
↓  
Relevant Scientific Evidence  
↓  
Multi-Metric Environmental Reasoning  
↓  
Evidence-Backed Recommendations  
↓  
Scientific Grounding Verification  
↓  
Final Environmental Decision Support

---

## Environmental Variables Supported

The environmental profile supports:

- Soil organic carbon
- Soil pH
- Soil moisture
- Rainfall
- Temperature
- Land use
- Crop type
- Biodiversity trend
- Species richness
- Habitat diversity
- Pollution
- Deforestation
- Region / climate type

---

## Multi-Turn Conversation Example

### Message 1

Biodiversity is declining on my wheat farm.

The system extracts:

- Land use: Agriculture
- Crop: Wheat
- Biodiversity trend: Declining

### Message 2

Soil organic carbon is 0.3% and rainfall is low.

The system adds:

- Soil organic carbon: 0.3%
- Rainfall: Low

### Message 3

The farm is in a semi-arid region.

The completed profile contains information such as:

- Soil organic carbon: 0.3%
- Rainfall: Low
- Land use: Agriculture
- Crop: Wheat
- Biodiversity trend: Declining
- Region: Semi-arid

Once enough information is available, the scientific assessment begins.

---

## Scientific Knowledge Base

The project uses a **Retrieval-Augmented Generation (RAG)** architecture.

Scientific PDF documents are stored inside:

`knowledge_base/`

The knowledge base contains scientific environmental material related to:

- Soil organic carbon
- Soil health
- Soil biodiversity
- Agroforestry
- Climate change
- Biodiversity
- Sustainable agriculture
- Land management
- Ecosystem restoration

The exact PDF filenames may vary.

The application uses metadata stored with the vector database so retrieved evidence can still display the actual source filename and page number.

---

## RAG Pipeline

Scientific PDF Documents  
↓  
PDF Text Extraction  
↓  
Text Chunking  
↓  
Sentence Transformer Embeddings  
↓  
Chroma Vector Database  
↓  
Semantic Similarity Search  
↓  
Relevant Scientific Evidence  
↓  
Environmental Reasoning

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

### Vector Database

`ChromaDB`

The vector database is stored locally inside:

`chroma_db/`

---

## Evidence Filtering

Scientific documents can contain bibliography-heavy or reference-heavy sections.

The retrieval pipeline filters reference-heavy chunks before passing evidence to the reasoning model.

This helps improve:

- Retrieval quality
- Scientific relevance
- Token efficiency
- Grounding quality

---

## Multi-Metric Environmental Reasoning

The chatbot reasons across multiple environmental variables together rather than treating each variable independently.

For example, it may analyze interactions between:

- Soil organic carbon
- Soil biodiversity
- Rainfall
- Agricultural land use
- Crop type
- Climate
- Habitat
- Biodiversity trend

When reasoning goes beyond direct scientific evidence, the system labels it as:

`Inference:`

---

## Evidence-Backed Recommendations

The current system generates three environmental intervention categories:

1. Agroforestry
2. Cover Cropping / Crop Rotation
3. Conservation / Reduced Tillage

Each recommendation includes:

- Direct Evidence
- Relevance to the Site
- Targeted Metrics
- Time Horizon
- Confidence
- Scientific source filename and page
- Priority Order

---

## Scientific Grounding Verification

The system does not automatically trust its own generated recommendations.

A separate verification stage compares generated scientific claims with the retrieved evidence.

Possible statuses include:

- Grounded
- Partially Grounded
- Not Grounded
- Verification Failed

This helps reduce hallucination and makes the environmental reasoning more transparent.

---

## Reliability Features

The application includes:

- Scientific RAG retrieval
- Source and page-level references
- Reference-heavy chunk filtering
- Deterministic environmental field extraction
- Multi-turn profile memory
- Missing-information detection
- Explicit inference labelling
- Conservative recommendation prompts
- Separate scientific grounding verification
- Incomplete recommendation detection
- Automatic retry handling for temporary Groq rate limits
- Preservation of user-provided units

---

## Language Model

The application uses:

`openai/gpt-oss-20b`

through the Groq API.

The LLM is mainly used for:

- Multi-metric environmental reasoning
- Recommendation generation
- Scientific grounding verification

Simple environmental information extraction is handled locally where possible to improve reliability and reduce token usage.

---

## User Interface

The application uses **Streamlit**.

The interface includes:

- Conversational chat
- Environmental profile sidebar
- Clarifying questions
- Environmental assessment
- Evidence-backed recommendations
- Scientific grounding status
- Verification notes
- Conversation reset functionality

---

## Project Structure

Biodiversity Chatbot/

- app/
  - app.py
  - backend.py
- knowledge_base/
  - scientific PDF documents
- chroma_db/
- darukaa_biodiversity.ipynb
- README.md
- requirements.txt
- .gitignore
- .env

The exact scientific PDF filenames inside `knowledge_base/` may differ depending on the source documents used during indexing.

---

## Installation

Install the required Python packages:

`pip install -r requirements.txt`

---

## Environment Variable

Create a `.env` file in the main project folder containing:

`GROQ_API_KEY=your_groq_api_key_here`

Do not upload the `.env` file to GitHub.

---

## Run the Application

Run:

`streamlit run app/app.py`

For the Anaconda environment used during development on Windows:

`& "C:\Users\krish\anaconda3\python.exe" -m streamlit run app/app.py`

---

## Recommended .gitignore

The `.gitignore` file should contain:

.env  
__pycache__/  
*.pyc  
.ipynb_checkpoints/  
.DS_Store

---

## Main Demo Scenario

Use the following messages one by one:

1. Biodiversity is declining on my wheat farm.
2. Soil organic carbon is 0.3% and rainfall is low.
3. The farm is in a semi-arid region.

This demonstrates:

- Natural-language understanding
- Structured environmental profile extraction
- Clarifying questions
- Multi-turn memory
- Multi-variable environmental reasoning
- Scientific RAG retrieval
- Evidence-backed recommendations
- Source citations
- Grounding verification

---

## Limitations

- The chatbot is a decision-support tool and is not a replacement for field measurements or professional environmental assessment.
- Environmental outcomes can vary by geography and local conditions.
- Planning time horizons are estimates unless explicitly supported by retrieved evidence.
- Scientific grounding depends on the quality and coverage of the documents in the knowledge base.
- Geographic coordinate analysis is not currently implemented.

---

## Technology Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers
- Groq API
- Pydantic
- PyPDF
- Retrieval-Augmented Generation
- Scientific grounding verification

---

## Project Goal

The goal is to provide:

**Retrievable, explainable, evidence-backed environmental intelligence.**

---

