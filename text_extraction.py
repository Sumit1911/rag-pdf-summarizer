import pdfplumber
import re
from typing import Optional

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace and formatting issues.
    
    Args:
        text (str): Raw text extracted from PDF
        
    Returns:
        str: Cleaned text
    """
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
    """
    Extract text from PDF file using pdfplumber.
    
    Args:
        pdf_path (str): Path to the PDF file
        max_pages (Optional[int]): Maximum number of pages to process (None for all)
        
    Returns:
        str: Extracted and cleaned text from the PDF
        
    Raises:
        Exception: If PDF cannot be opened or processed
    """
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Maximum number of tokens per chunk
        overlap (int): Number of tokens to overlap between chunks
        
    Returns:
        list[str]: List of text chunks
    """
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

def extract_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary containing PDF metadata
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
            
            # Add additional information
            metadata.update({
                'total_pages': len(pdf.pages),
                'file_path': pdf_path
            })
            
            # Clean metadata values
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    cleaned_metadata[key] = str(value)
            
            return cleaned_metadata
            
    except Exception:
        return {
            'total_pages': 0,
            'file_path': pdf_path,
            'error': 'Could not extract metadata'
        }

def validate_pdf(pdf_path: str) -> tuple[bool, str]:
    """
    Validate if the PDF file can be processed.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return False, "PDF file contains no pages"
            
            # Try to extract text from first page
            first_page_text = pdf.pages[0].extract_text()
            if not first_page_text and not pdf.pages[0].extract_tables():
                return False, "PDF appears to contain no extractable text or tables"
            
            return True, "PDF is valid and contains extractable content"
            
    except Exception as e:
        return False, f"Error validating PDF: {str(e)}"