# RAG-Based Mutual Fund Chatbot

A Retrieval-Augmented Generation (RAG) system designed to analyze AMC mutual fund factsheets and answer queries about fund performance, allocations, and financial metrics.

## Overview

This project implements a production-grade RAG chatbot that processes PDF factsheets from Asset Management Companies (AMCs) and provides accurate answers to queries about fund returns, NAV allocations, expense ratios, fund managers, and other financial metrics.

## Architecture

### Core Components

- **LLM Generation**: Llama 3.1-8B (4-bit quantized) for response generation
- **Embeddings**: BAAI/bge-small-en-v1.5 for semantic search
- **Vector Database**: Qdrant for vector storage and retrieval
- **Caching Layer**: Redis for response caching
- **PDF Parser**: LlamaParse for high-accuracy financial document parsing
- **Frontend**: Gradio interface for user interaction

### Infrastructure

- **Deployment**: Modal Labs
- **Compute**: L40S GPUs
- **API Integration**: LlamaParse API for document processing

## Key Features

### Advanced PDF Processing

- Financial keyword detection for improved relevance
- Table-aware chunking with 3x larger chunk sizes to preserve table integrity
- Extracts complex financial tables and structured data
- Handles multi-column layouts and nested data structures

### Hybrid Search

- Semantic vector search using embeddings
- Query-aware reranking for improved relevance
- Context-preserving retrieval with larger chunk sizes (400-7200 characters)

### Performance Optimizations

- 4-bit quantization for efficient GPU memory usage
- Response caching via Redis
- Optimized chunk sizes for financial data

## Technical Implementation

### Document Processing Pipeline

1. PDF ingestion via LlamaParse API
2. Financial keyword detection and filtering
3. Table-aware chunking with overlap
4. Embedding generation using BGE-small
5. Vector storage in Qdrant

### Query Processing

1. Query embedding generation
2. Hybrid vector search with semantic matching
3. Context reranking based on relevance
4. LLM generation with retrieved context
5. Response caching for repeated queries

## Deployment

### Prerequisites

- Modal Labs account
- LlamaParse API key
- Python 3.8+

### Environment Variables

```
LLAMA_PARSE_API_KEY=your_api_key_here
```

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
modal serve app.py
```

### Deploying to Production

```bash
modal deploy app.py
```

## Performance Metrics

- Successfully extracts 140+ tables per factsheet
- Generates 800+ searchable chunks per document
- Accurately answers queries about:
  - Fund returns and performance metrics
  - NAV allocations and percentages
  - Financial ratios and indicators
  - Fund management information

## Test Query Examples

- "What is the 1-year return of the fund?"
- "What percentage of NAV is allocated to HDFC Bank?"
- "What is the expense ratio?"
- "Who is the fund manager?"
- "Define Jensen's Alpha"

## Challenges Addressed

### Initial Approach

- Local PDF parsing tools (PyMuPDF, Camelot, Tabula) failed to extract table data
- Small chunk sizes (25-73 characters) fragmented context
- Poor retrieval accuracy for financial queries

### Solution

- Migrated to LlamaParse for specialized financial document parsing
- Implemented table-aware chunking with 3x larger chunks
- Added financial keyword detection for relevance filtering
- Resolved deployment issues including nested event loops and API authentication

## Frontend

A simple Gradio interface provides:

- PDF upload functionality
- Chat interface for querying fund information
- Real-time response generation
- Clean, professional UI without unnecessary complexity

## Project Context

This project was developed as part of the Bajaj Finserv AI Engineer evaluation, demonstrating practical implementation of RAG systems for financial document analysis.

## License

This project is provided as-is for evaluation purposes.

## Notes

The system prioritizes accuracy and simplicity over complex features, focusing on reliable information extraction from financial documents and straightforward deployment architecture.
