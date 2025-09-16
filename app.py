import streamlit as st
import os
import tempfile
import io
import re
from datetime import datetime
from typing import Optional, List

# Import required libraries
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG PDF Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #d4edda;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Text extraction functions
def clean_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace and formatting issues."""
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (simple patterns)
    text = re.sub(r'\b\d+\b(?=\s*$)', '', text, flags=re.MULTILINE)
    
    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Fix common spacing issues
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Add space before capital letters
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """Extract text from PDF file using pdfplumber."""
    try:
        extracted_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            for page_num in range(pages_to_process):
                page = pdf.pages[page_num]
                
                # Extract text from the page
                page_text = page.extract_text()
                
                if page_text:
                    # Clean the page text
                    cleaned_page_text = clean_text(page_text)
                    extracted_text += cleaned_page_text + " "
                
                # Try to extract text from tables if regular text extraction fails
                if not page_text and page.extract_tables():
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " ".join([cell for cell in row if cell])
                                extracted_text += clean_text(row_text) + " "
        
        # Final cleaning of the entire document
        final_text = clean_text(extracted_text)
        
        if not final_text:
            raise Exception("No text could be extracted from the PDF file")
            
        return final_text
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks for processing."""
    if not text:
        return []
    
    # Simple word-based chunking (approximating tokens)
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    i = 0
    while i < len(words):
        # Create chunk
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        i += chunk_size - overlap
        
        # Break if we're at the end
        if i >= len(words):
            break
    
    return chunks

# RAG Retriever class
class RAGRetriever:
    """Retrieval-Augmented Generation retriever using FAISS for vector similarity search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, top_k: int = 3):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # Load the sentence transformer model
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.encoder = SentenceTransformer(model_name)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def process_document(self, document_text: str, overlap: int = 50) -> None:
        """Process a document by chunking and creating embeddings."""
        try:
            logger.info("Starting document processing...")
            
            # Chunk the document
            logger.info(f"Chunking document with chunk_size={self.chunk_size}, overlap={overlap}")
            self.chunks = chunk_text(document_text, chunk_size=self.chunk_size, overlap=overlap)
            
            if not self.chunks:
                raise Exception("No chunks were created from the document")
            
            logger.info(f"Created {len(self.chunks)} chunks")
            
            # Create embeddings for all chunks
            logger.info("Generating embeddings for chunks...")
            self.embeddings = self.encoder.encode(
                self.chunks,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Build FAISS index
            logger.info("Building FAISS index...")
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized embeddings
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Successfully processed document with {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve relevant chunks for a given query."""
        if self.index is None or not self.chunks:
            raise Exception("Document must be processed before retrieval")
        
        try:
            # Use instance default if top_k not specified
            k = top_k if top_k is not None else self.top_k
            k = min(k, len(self.chunks))  # Don't retrieve more chunks than available
            
            logger.info(f"Retrieving top {k} chunks for query: '{query[:50]}...'")
            
            # Encode the query
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search for similar chunks
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Get the relevant chunks
            relevant_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    relevant_chunks.append(chunk)
                    logger.debug(f"Retrieved chunk {i+1} (score: {score:.4f}): {chunk[:100]}...")
            
            logger.info(f"Successfully retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise Exception(f"Failed to retrieve relevant chunks: {str(e)}")

# Document Summarizer class
class DocumentSummarizer:
    """Document summarization using pre-trained transformer models."""
    
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.summarizer = None
        
        # Length settings for different summary types
        self.length_settings = {
            "short": {"min_length": 30, "max_length": 80},
            "medium": {"min_length": 80, "max_length": 150},
            "detailed": {"min_length": 150, "max_length": 300}
        }
        
        try:
            logger.info(f"Loading summarization model: {model_name}")
            
            # Create pipeline
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                framework="pt",
                device=-1  # CPU only for lightweight deployment
            )
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise Exception(f"Failed to load summarization model: {str(e)}")
    
    def summarize(self, text: str, length: str = "medium", query: Optional[str] = None) -> str:
        """Summarize text with specified length."""
        try:
            if not text.strip():
                return "No text provided for summarization."
            
            # Get length settings
            length_config = self.length_settings.get(length, self.length_settings["medium"])
            
            # Prepare input text
            input_text = text[:1000]  # Limit input length
            if query:
                input_text = f"Question: {query}\nContext: {text[:800]}"
            
            # Generate summary
            result = self.summarizer(
                input_text,
                min_length=length_config["min_length"],
                max_length=length_config["max_length"],
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )
            
            return result[0]['summary_text'].strip()
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def extract_key_points(self, text: str) -> str:
        """Extract key points from text."""
        try:
            prompt_text = f"Extract the main key points from this text:\n\n{text[:1000]}"
            
            summary = self.summarizer(
                prompt_text,
                min_length=50,
                max_length=200,
                do_sample=False,
                num_beams=4
            )
            
            key_points_text = summary[0]['summary_text']
            
            # Format as bullet points
            sentences = [s.strip() for s in key_points_text.split('.') if s.strip()]
            
            formatted_points = []
            for i, sentence in enumerate(sentences[:5]):
                if sentence:
                    formatted_points.append(f"• {sentence.capitalize()}")
            
            if formatted_points:
                return "\n".join(formatted_points)
            else:
                return "• " + key_points_text
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return f"Error extracting key points: {str(e)}"

# Session state initialization
def initialize_session_state():
    """Initialize session state variables"""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'summary_history' not in st.session_state:
        st.session_state.summary_history = []

def main():
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>📄 RAG-Based PDF Summarizer</h1>
        <p>Upload a PDF, ask questions, and get intelligent summaries using Retrieval-Augmented Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading status
        if st.session_state.summarizer is None:
            with st.spinner("Loading AI models..."):
                st.session_state.summarizer = DocumentSummarizer()
            st.success("✅ Models loaded successfully!")
        
        st.markdown("---")
        
        # Summary length setting
        summary_length = st.selectbox(
            "📏 Summary Length",
            ["short", "medium", "detailed"],
            index=1,
            help="Choose how detailed you want the summary to be"
        )
        
        # Chunk size setting
        chunk_size = st.slider(
            "📝 Chunk Size (tokens)",
            min_value=200,
            max_value=1000,
            value=500,
            step=100,
            help="Size of text chunks for processing"
        )
        
        # Top-k retrieval setting
        top_k = st.slider(
            "🔍 Retrieval Chunks",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant chunks to retrieve"
        )
        
        st.markdown("---")
        st.markdown("### 📊 App Info")
        st.info("**Tech Stack:**\n- Embeddings: SentenceTransformers\n- Vector Search: FAISS\n- Summarization: DistilBART\n- Framework: Streamlit")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload PDF Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze and summarize"
        )
        
        if uploaded_file is not None:
            # Process PDF
            with st.spinner("📄 Extracting text from PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    document_text = extract_text_from_pdf(tmp_file_path)
                    st.session_state.document_text = document_text
                    
                    if document_text.strip():
                        st.success(f"✅ Successfully extracted {len(document_text)} characters")
                        
                        # Show document preview
                        with st.expander("📖 Document Preview", expanded=False):
                            st.text_area(
                                "First 1000 characters:",
                                value=document_text[:1000] + "..." if len(document_text) > 1000 else document_text,
                                height=200,
                                disabled=True
                            )
                        
                        # Initialize RAG components
                        with st.spinner("🧠 Building knowledge base..."):
                            st.session_state.retriever = RAGRetriever(
                                chunk_size=chunk_size,
                                top_k=top_k
                            )
                            st.session_state.retriever.process_document(document_text)
                            st.session_state.pdf_processed = True
                            
                        st.success("🎉 PDF processed successfully! Ready for queries.")
                        
                    else:
                        st.error("❌ Could not extract text from PDF. Please try a different file.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
    
    with col2:
        st.header("❓ Ask Questions")
        
        if st.session_state.pdf_processed:
            # Query input
            query = st.text_input(
                "Enter your question about the document:",
                placeholder="e.g., What are the main points discussed in this document?",
                help="Ask any question about the uploaded PDF content"
            )
            
            # Generate summary button
            if st.button("🔍 Generate Summary", type="primary", disabled=not query.strip()):
                if query.strip():
                    with st.spinner("🤖 Generating intelligent summary..."):
                        try:
                            # Retrieve relevant chunks
                            relevant_chunks = st.session_state.retriever.retrieve(query)
                            
                            if relevant_chunks:
                                # Combine chunks for summarization
                                combined_text = " ".join(relevant_chunks)
                                
                                # Generate summary
                                summary = st.session_state.summarizer.summarize(
                                    combined_text,
                                    length=summary_length,
                                    query=query
                                )
                                
                                # Display summary
                                st.markdown("""
                                <div class="summary-box">
                                    <h4>📋 Generated Summary</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.write(summary)
                                
                                # Add to history
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.summary_history.append({
                                    'query': query,
                                    'summary': summary,
                                    'timestamp': timestamp,
                                    'length': summary_length
                                })
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Summary",
                                    data=f"Query: {query}\n\nSummary:\n{summary}\n\nGenerated on: {timestamp}",
                                    file_name=f"summary_{timestamp.replace(':', '-').replace(' ', '_')}.txt",
                                    mime="text/plain"
                                )
                                
                            else:
                                st.warning("⚠️ No relevant content found for your query. Try rephrasing your question.")
                                
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {str(e)}")
            
            # Quick summary options
            st.markdown("### 🚀 Quick Actions")
            col_quick1, col_quick2 = st.columns(2)
            
            with col_quick1:
                if st.button("📝 Full Document Summary"):
                    with st.spinner("Generating full document summary..."):
                        try:
                            # Use entire document for summary
                            summary = st.session_state.summarizer.summarize(
                                st.session_state.document_text[:2000],  # Limit for model constraints
                                length=summary_length
                            )
                            st.success("✅ Full document summary generated!")
                            st.write(summary)
                            
                            # Add to history
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.summary_history.append({
                                'query': 'Full Document Summary',
                                'summary': summary,
                                'timestamp': timestamp,
                                'length': summary_length
                            })
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with col_quick2:
                if st.button("🔑 Key Points Extraction"):
                    with st.spinner("Extracting key points..."):
                        try:
                            key_points = st.session_state.summarizer.extract_key_points(
                                st.session_state.document_text[:2000]
                            )
                            st.success("✅ Key points extracted!")
                            st.write(key_points)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            
        else:
            st.markdown("""
            <div class="info-box">
                <p>👆 Please upload a PDF document first to start asking questions and generating summaries.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary History Section
    if st.session_state.summary_history:
        st.markdown("---")
        st.header("📚 Summary History")
        
        for i, item in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"🕒 {item['timestamp']} - {item['query'][:50]}..."):
                st.write(f"**Query:** {item['query']}")
                st.write(f"**Length:** {item['length']}")
                st.write(f"**Summary:**")
                st.write(item['summary'])
                
                # Download individual summary
                st.download_button(
                    label="📥 Download This Summary",
                    data=f"Query: {item['query']}\n\nSummary:\n{item['summary']}\n\nGenerated on: {item['timestamp']}",
                    file_name=f"summary_{i+1}.txt",
                    mime="text/plain",
                    key=f"download_{i}"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit, SentenceTransformers, FAISS, and DistilBART</p>
        <p>🚀 Optimized for deployment on Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
