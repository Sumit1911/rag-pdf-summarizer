import streamlit as st
import os
import tempfile
import io
from datetime import datetime
from utils.text_extraction import extract_text_from_pdf
from utils.retrieval import RAGRetriever
from utils.summarization import DocumentSummarizer

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
                                st.session_state.document_text[:4000],  # Limit for model constraints
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
                                st.session_state.document_text[:4000]
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
        <p>🚀 Optimized for deployment on Hugging Face Spaces & Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()