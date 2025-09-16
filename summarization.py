from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import Optional, List
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    Document summarization using pre-trained transformer models.
    Optimized for lightweight deployment with DistilBART.
    """
    
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the document summarizer.
        
        Args:
            model_name (str): Name of the summarization model to use
        """
        self.model_name = model_name
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
        # Length settings for different summary types
        self.length_settings = {
            "short": {"min_length": 30, "max_length": 80},
            "medium": {"min_length": 80, "max_length": 150},
            "detailed": {"min_length": 150, "max_length": 300}
        }
        
        try:
            logger.info(f"Loading summarization model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                framework="pt",
                device=-1  # CPU only for lightweight deployment
            )
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise Exception(f"Failed to load summarization model: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for summarization.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short sentences that might be artifacts
        sentences = text.split('.')
        filtered_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        text = '. '.join(filtered_sentences)
        
        # Ensure text ends with proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()
    
    def chunk_for_summarization(self, text: str, max_chunk_length: int = 1000) -> List[str]:
        """
        Chunk text for summarization to handle model input limits.
        
        Args:
            text (str): Input text
            max_chunk_length (int): Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # Simple word-based chunking (approximating tokens)
        words = text.split()
        
        if len(words) <= max_chunk_length:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + 1 <= max_chunk_length:
                current_chunk.append(word)
                current_length += 1
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize(self, text: str, length: str = "medium", query: Optional[str] = None) -> str:
        """
        Summarize text with specified length.
        
        Args:
            text (str): Input text to summarize
            length (str): Summary length ("short", "medium", "detailed")
            query (Optional[str]): Optional query to focus the summary
            
        Returns:
            str: Generated summary
        """
        try:
            if not text.strip():
                return "No text provided for summarization."
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return "No meaningful content found for summarization."
            
            # Get length settings
            length_config = self.length_settings.get(length, self.length_settings["medium"])
            
            # Handle long texts by chunking
            chunks = self.chunk_for_summarization(processed_text, max_chunk_length=800)
            
            if len(chunks) == 1:
                # Single chunk - direct summarization
                summary = self._summarize_chunk(
                    processed_text,
                    length_config["min_length"],
                    length_config["max_length"],
                    query
                )
            else:
                # Multiple chunks - summarize each and combine
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                    chunk_summary = self._summarize_chunk(
                        chunk,
                        max(20, length_config["min_length"] // len(chunks)),
                        max(50, length_config["max_length"] // len(chunks)),
                        query
                    )
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)
                
                # Combine chunk summaries
                combined_summary = " ".join(chunk_summaries)
                
                # Final summarization if combined text is still too long
                if len(combined_summary.split()) > length_config["max_length"]:
                    summary = self._summarize_chunk(
                        combined_summary,
                        length_config["min_length"],
                        length_config["max_length"],
                        query
                    )
                else:
                    summary = combined_summary
            
            return self.postprocess_summary(summary, query)
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _summarize_chunk(self, text: str, min_length: int, max_length: int, query: Optional[str] = None) -> str:
        """
        Summarize a single chunk of text.
        
        Args:
            text (str): Text chunk to summarize
            min_length (int): Minimum summary length
            max_length (int): Maximum summary length
            query (Optional[str]): Optional query context
            
        Returns:
            str: Summary of the chunk
        """
        try:
            # Prepare input text
            input_text = text
            if query:
                input_text = f"Question: {query}\nContext: {text}"
            
            # Generate summary
            result = self.summarizer(
                input_text,
                min_length=min_length,
                max_length=max_length,
                do_sample=False,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=1.0,
                num_beams=4,
                early_stopping=True
            )
            
            return result[0]['summary_text'].strip()
            
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return ""
    
    def extract_key_points(self, text: str, num_points: int = 5) -> str:
        """
        Extract key points from text.
        
        Args:
            text (str): Input text
            num_points (int): Number of key points to extract
            
        Returns:
            str: Formatted key points
        """
        try:
            # Create a focused prompt for key point extraction
            prompt_text = f"Extract the main key points from this text:\n\n{text[:2000]}"
            
            summary = self.summarizer(
                prompt_text,
                min_length=50,
                max_length=200,
                do_sample=False,
                temperature=0.5,
                num_beams=4
            )
            
            key_points_text = summary[0]['summary_text']
            
            # Format as bullet points
            sentences = [s.strip() for s in key_points_text.split('.') if s.strip()]
            
            formatted_points = []
            for i, sentence in enumerate(sentences[:num_points]):
                if sentence:
                    formatted_points.append(f"• {sentence.capitalize()}")
            
            if formatted_points:
                return "\n".join(formatted_points)
            else:
                return "• " + key_points_text
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return f"Error extracting key points: {str(e)}"
    
    def query_focused_summary(self, text: str, query: str, length: str = "medium") -> str:
        """
        Generate a query-focused summary.
        
        Args:
            text (str): Input text
            query (str): Focus query
            length (str): Summary length
            
        Returns:
            str: Query-focused summary
        """
        try:
            # Create query-focused prompt
            focused_text = f"Based on the question '{query}', summarize the relevant information from this text:\n\n{text}"
            
            return self.summarize(focused_text, length=length, query=query)
            
        except Exception as e:
            logger.error(f"Error in query-focused summarization: {str(e)}")
            return f"Error generating query-focused summary: {str(e)}"
    
    def postprocess_summary(self, summary: str, query: Optional[str] = None) -> str:
        """
        Post-process the generated summary.
        
        Args:
            summary (str): Raw summary
            query (Optional[str]): Original query for context
            
        Returns:
            str: Post-processed summary
        """
        if not summary:
            return "Unable to generate summary."
        
        # Clean up common issues
        summary = summary.strip()
        
        # Remove redundant phrases
        summary = re.sub(r'\b(the text|the document|this text|this document)\s+', '', summary, flags=re.IGNORECASE)
        
        # Ensure proper capitalization
        sentences = summary.split('. ')
        capitalized_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                capitalized_sentences.append(sentence)
        
        summary = '. '.join(capitalized_sentences)
        
        # Ensure proper ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "Sequence-to-Sequence Summarization",
            "framework": "Transformers (PyTorch)",
            "length_settings": self.length_settings,
            "device": "CPU (optimized for deployment)"
        }