# NexaAI Self-RAG Assistant

A Self-RAG (Retrieval-Augmented Generation) chatbot built using LangGraph + Ollama + Streamlit.
It intelligently decides when to retrieve company documents and ensures answers are grounded, useful, and verifiable.

<img src="https://github.com/NeelContractor/NexaAI_SelfRAG_APP/blob/main/public/demo.png" width="70%" height="70%">

## Features
- Smart Retrieval Decision
    - Uses keyword rules + LLM to decide whether to retrieve docs or answer directly
- Document RAG Pipeline
    - Loads PDFs → chunks → embeddings → FAISS vector store
- Self-RAG Loops
    - Relevance filtering (keeps only useful chunks)
    - IsSUP → verifies grounding (with revise loop)
    - IsUSE → checks usefulness (with query rewrite loop)
- Query Rewriting
    - Automatically improves retrieval queries when answers are weak
- Trace Debug Panel (Streamlit)
    - Step-by-step reasoning
    - Grounding + usefulness badges
    - Sources + evidence

## Tech Stack
- LangGraph → pipeline orchestration
- Ollama → local LLM + embeddings
- FAISS → vector database
- Streamlit → frontend UI

## Project Structure
```
.
├── backend.py          # Self-RAG pipeline (LangGraph)
├── app.py              # Streamlit frontend
├── documents/          # PDF knowledge base
│   ├── Company_Policies.pdf
│   ├── Company_Profile.pdf
│   └── Product_and_pricing.pdf
├── .env
└── README.md
```
