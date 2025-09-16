import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
from .text_extraction import chunk_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever using FAISS for vector similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, top_k: int = 3):
        """
        Initialize the RAG retriever.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
            chunk_size (int): Size of text chunks for processing
            top_k (int): Number of top chunks to retrieve
        """
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
        """
        Process a document by chunking and creating embeddings.
        
        Args:
            document_text (str): Full document text
            overlap (int): Overlap between chunks in tokens
        """
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
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query (str): User query
            top_k (Optional[int]): Number of chunks to retrieve (uses instance default if None)
            
        Returns:
            List[str]: List of relevant text chunks
        """
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
    
    def get_chunk_with_context(self, query: str, context_window: int = 1) -> List[str]:
        """
        Retrieve chunks with surrounding context.
        
        Args:
            query (str): User query
            context_window (int): Number of adjacent chunks to include on each side
            
        Returns:
            List[str]: List of chunks with context
        """
        if self.index is None or not self.chunks:
            raise Exception("Document must be processed before retrieval")
        
        try:
            # Get initial relevant chunks
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            scores, indices = self.index.search(query_embedding.astype('float32'), self.top_k)
            
            # Expand indices to include context
            expanded_indices = set()
            for idx in indices[0]:
                if idx < len(self.chunks):
                    # Add the chunk itself and context around it
                    for i in range(max(0, idx - context_window), 
                                 min(len(self.chunks), idx + context_window + 1)):
                        expanded_indices.add(i)
            
            # Sort indices and get chunks
            sorted_indices = sorted(expanded_indices)
            contextual_chunks = [self.chunks[i] for i in sorted_indices]
            
            logger.info(f"Retrieved {len(contextual_chunks)} chunks with context")
            return contextual_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks with context: {str(e)}")
            raise Exception(f"Failed to retrieve chunks with context: {str(e)}")
    
    def semantic_search(self, query: str, threshold: float = 0.3) -> List[tuple[str, float]]:
        """
        Perform semantic search with similarity scores.
        
        Args:
            query (str): User query
            threshold (float): Minimum similarity threshold
            
        Returns:
            List[tuple[str, float]]: List of (chunk, score) pairs above threshold
        """
        if self.index is None or not self.chunks:
            raise Exception("Document must be processed before search")
        
        try:
            # Encode query
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search all chunks
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                len(self.chunks)
            )
            
            # Filter by threshold and return with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            
            logger.info(f"Found {len(results)} chunks above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise Exception(f"Failed to perform semantic search: {str(e)}")
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the processed document.
        
        Returns:
            dict: Dictionary containing processing statistics
        """
        if not self.chunks:
            return {"status": "No document processed"}
        
        chunk_lengths = [len(chunk.split()) for chunk in self.chunks]
        
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "chunk_size_setting": self.chunk_size,
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_tokens_approx": sum(chunk_lengths),
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def save_index(self, filepath: str) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            filepath (str): Path to save the index
        """
        if self.index is None:
            raise Exception("No index to save")
        
        try:
            faiss.write_index(self.index, filepath)
            logger.info(f"Index saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise Exception(f"Failed to save index: {str(e)}")
    
    def load_index(self, filepath: str, chunks: List[str]) -> None:
        """
        Load a FAISS index from disk.
        
        Args:
            filepath (str): Path to the saved index
            chunks (List[str]): Corresponding text chunks
        """
        try:
            self.index = faiss.read_index(filepath)
            self.chunks = chunks
            logger.info(f"Index loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise Exception(f"Failed to load index: {str(e)}")