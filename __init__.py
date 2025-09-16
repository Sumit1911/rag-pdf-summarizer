"""
RAG-based PDF Summarizer Utilities

This package contains utility modules for:
- PDF text extraction and preprocessing
- Retrieval-augmented generation (RAG) with FAISS
- Document summarization using transformer models
"""

__version__ = "1.0.0"
__author__ = "RAG PDF Summarizer"

# Import main classes for easy access
from .text_extraction import extract_text_from_pdf, chunk_text, extract_metadata
from .retrieval import RAGRetriever
from .summarization import DocumentSummarizer

__all__ = [
    'extract_text_from_pdf',
    'chunk_text', 
    'extract_metadata',
    'RAGRetriever',
    'DocumentSummarizer'
]