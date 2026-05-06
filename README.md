# RAG-PROJECT

# RAG-Based Q&A System – Attention Is All You Need

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions from the research paper “Attention Is All You Need” using semantic search and LLM-based generation.

# Overview

The system processes the PDF document and enables intelligent question-answering by combining:

- Vector Search (FAISS) → Retrieves relevant content
- Embeddings (Sentence Transformers) → Converts text into vectors
- LLM (Groq API) → Generates final answers

# Project Architecture
  
# 1. Ingestion Pipeline (ingest.py)

Builds the knowledge base:

- Loads PDF documents from /data
- Splits text into chunks (size: 500, overlap: 50)
- Converts chunks into embeddings using all-MiniLM-L6-v2
  
- Stores:
   - FAISS index → /store/faiss_index
   - Document chunks → /store/chunks
  
# 2. Retriever (retriever.py)
- Loads FAISS index and chunk data
- Converts user query into embedding
- Performs similarity search
- Returns top-k (default=5) relevant chunks
  
# 3. RAG Pipeline (rag_chain.py)
- Retrieves context using retriever
- Builds prompt with context + question
- Uses Groq LLM (openai/gpt-oss-120b) to generate answer
- Ensures answers are grounded in document context
  
# 4. CLI Interface (main.py)
Simple command-line interface

Users can:
- Ask questions
- Type exit or quit to stop

# Key Features
- Efficient semantic search using FAISS
- Context-aware answers (reduces hallucination)
- Modular and scalable design
- Works with any PDF-based knowledge source

# Note

If the answer is not found in the document, the system responds with:

"I don't know based on the provided document."
