# 📄 RAG-Based PDF Summarizer

A powerful **Retrieval-Augmented Generation (RAG)** web application that allows users to upload PDF documents, ask questions, and receive intelligent summaries based on the most relevant content.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **📤 PDF Upload**: Easy drag-and-drop PDF upload with text extraction
- **🧠 Intelligent Retrieval**: Uses SentenceTransformers and FAISS for semantic search
- **📝 Smart Summarization**: Powered by DistilBART for accurate, concise summaries
- **❓ Query-Based**: Ask specific questions and get focused answers
- **📏 Customizable Length**: Choose between short, medium, and detailed summaries
- **📚 Summary History**: Keep track of all generated summaries
- **📥 Export Options**: Download summaries as text files
- **⚙️ Configurable**: Adjust chunk sizes, retrieval parameters, and more
- **🚀 Deployment Ready**: Optimized for free hosting platforms

## 🏗️ Architecture

```
User Query → Text Extraction → Chunking → Embeddings → FAISS Index → Retrieval → Summarization → Response
```

### Tech Stack

- **Backend**: Python with Streamlit
- **Text Extraction**: pdfplumber
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS (CPU-optimized)
- **Summarization**: DistilBART (`sshleifer/distilbart-cnn-12-6`)
- **Frontend**: Streamlit's built-in components

## 📁 Project Structure

```
app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── utils/
    ├── __init__.py            # Package initialization
    ├── text_extraction.py     # PDF text extraction utilities
    ├── retrieval.py           # RAG retrieval system
    └── summarization.py       # Document summarization
```

## 🚀 Quick Start

### Local Development

1. **Clone or download the project files**

2. **Create a virtual environment**:
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create the utils directory and add __init__.py**:
   ```bash
   mkdir utils
   touch utils/__init__.py
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Fork/Upload to GitHub**: Upload all files to a GitHub repository
2. **Connect to Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Connect your GitHub repo and deploy
4. **Access**: Your app will be available at `https://username-app-name.streamlit.app`

### Option 2: Hugging Face Spaces

1. **Create a Space**: Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Choose Streamlit**: Select Streamlit as your SDK
3. **Upload Files**: Upload all project files
4. **Auto-Deploy**: The space will automatically build and deploy

### Option 3: Render

1. **Connect GitHub**: Link your GitHub repository to Render
2. **Create Web Service**: Choose "Web Service" and connect your repo
3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. **Deploy**: Render will handle the rest

## 💡 Usage Guide

### Basic Workflow

1. **Upload PDF**: Click "Choose a PDF file" and select your document
2. **Wait for Processing**: The app will extract text and build the knowledge base
3. **Ask Questions**: Enter your question in the query box
4. **Get Summary**: Click "Generate Summary" to receive an intelligent response
5. **Customize**: Adjust summary length and retrieval parameters in the sidebar
6. **Download**: Use the download button to save summaries as text files

### Advanced Features

- **Full Document Summary**: Get an overview of the entire document
- **Key Points Extraction**: Extract main points from the document
- **Summary History**: Review all previous queries and summaries
- **Configurable Parameters**: Adjust chunk size, retrieval count, etc.

## 📊 Model Information

### Embedding Model: `all-MiniLM-L6-v2`
- **Size**: ~80MB
- **Speed**: Fast inference
- **Quality**: Good semantic understanding
- **Dimensions**: 384

### Summarization Model: `sshleifer/distilbart-cnn-12-6`
- **Size**: ~300MB
- **Type**: Encoder-decoder transformer
- **Optimized**: For news summarization
- **Performance**: Balanced speed/quality

## ⚙️ Configuration Options

### In the Sidebar:
- **Summary Length**: Short (30-80 words), Medium (80-150 words), Detailed (150-300 words)
- **Chunk Size**: 200-1000 tokens per chunk
- **Retrieval Chunks**: 1-10 most relevant chunks to use

### Advanced Settings (in code):
```python
# Text extraction
max_pages = None  # Process all pages
overlap = 50      # Token overlap between chunks

# Embeddings
normalize_embeddings = True  # Better similarity search
show_progress_bar = True     # Visual feedback

# Summarization
temperature = 0.7           # Creativity level
repetition_penalty = 1.2    # Reduce repetition
num_beams = 4              # Beam search width
```

## 🔧 Troubleshooting

### Common Issues:

1. **Model Loading Errors**:
   - Ensure stable internet for initial model download
   - Check available disk space (need ~500MB for models)

2. **PDF Processing Issues**:
   - Some PDFs may have locked text or unusual encoding
   - Try with different PDF files to isolate the issue

3. **Memory Issues on Free Hosting**:
   - Reduce chunk size and top_k retrieval count
   - Process fewer pages if document is very large

4. **Slow Performance**:
   - Models run on CPU for broad compatibility
   - Consider upgrading to GPU-enabled hosting for faster inference

### Performance Tips:

- **For Large Documents**: Increase chunk size and reduce overlap
- **For Better Accuracy**: Increase retrieval chunks (top_k)
- **For Faster Processing**: Use smaller chunk sizes and fewer retrieval chunks

## 🧪 Testing

### Test with Sample Documents:
1. **Research Papers**: Test academic PDF summarization
2. **Reports**: Business or technical reports
3. **Books/Articles**: Long-form content summarization

### Example Queries:
- "What are the main findings of this research?"
- "Summarize the key recommendations"
- "What methodology was used in this study?"
- "List the main conclusions"

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes and test thoroughly**
4. **Submit a pull request**

### Areas for Enhancement:
- Multi-language support
- Document comparison features
- Export to different formats (PDF, Word)
- Integration with cloud storage
- Batch processing capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing pre-trained models
- **FAISS** team for efficient similarity search
- **pdfplumber** for robust PDF text extraction
- **Streamlit** for the excellent web framework

## 📞 Support

For issues, questions, or contributions:
1. **Check the troubleshooting section above**
2. **Review existing GitHub issues**
3. **Create a new issue with detailed information**

---

