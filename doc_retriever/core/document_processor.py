"""
Core document processing service that orchestrates the document processing pipeline.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import re
from functools import lru_cache
from tqdm import tqdm

from doc_retriever.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_RETRIES,
    RETRY_DELAY,
    UPLOADS_DIR,
    BATCH_SIZE
)
from doc_retriever.models.document import Document
from doc_retriever.services.embedding_service import EmbeddingService
from doc_retriever.services.summarizer_service import SummarizerService
from doc_retriever.services.vector_store_service import VectorStoreService

class DocumentProcessor:
    """Service for processing documents through the pipeline."""
    
    def __init__(self):
        """Initialize the document processor with required services."""
        self.embedding_service = EmbeddingService()
        self.summarizer_service = SummarizerService()
        self.vector_store_service = VectorStoreService()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on system resources
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def process_document(self, file_path: str) -> Document:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed Document object
            
        Raises:
            Exception: If document processing fails
        """
        try:
            # Read document
            doc = fitz.open(file_path)
            file_path = Path(file_path)
            
            # Get relative path from UPLOADS_DIR
            try:
                relative_path = file_path.relative_to(UPLOADS_DIR)
            except ValueError:
                relative_path = file_path.name
            
            # Extract text and metadata using parallel processing
            text_chunks = []
            total_pages = len(doc)
            
            print(f"Processing document with {total_pages} pages...")
            for i, page in enumerate(tqdm(doc, desc="Extracting text")):
                text = page.get_text()
                chunks = self._split_text(text)
                text_chunks.extend(chunks)
                # Clear memory after each page
                del text
                del chunks
            
            metadata = {
                "filename": file_path.name,
                "file_path": str(relative_path),
                "absolute_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "mime_type": doc.metadata.get("format", "application/pdf"),
                "page_count": total_pages,
                "created_at": datetime.now().isoformat(),
                "storage_location": "uploads"
            }
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Process chunks in batches
            processed_chunks = []
            previous_summary = None
            previous_end_context = None
            
            print("Processing chunks...")
            for i in tqdm(range(0, len(text_chunks), BATCH_SIZE), desc="Processing chunks"):
                batch = text_chunks[i:i + BATCH_SIZE]
                batch_results = self._process_chunk_batch(
                    batch,
                    previous_summary,
                    previous_end_context
                )
                processed_chunks.extend(batch_results)
                
                # Update context for next batch
                if batch_results:
                    previous_summary = batch_results[-1]["summary"]
                    previous_end_context = batch_results[-1]["end_context"]
                
                # Clear memory after each batch
                del batch
                del batch_results
            
            # Generate embeddings for rewritten chunks in batches
            print("Generating embeddings...")
            embeddings = []
            for i in tqdm(range(0, len(processed_chunks), BATCH_SIZE), desc="Generating embeddings"):
                batch = [chunk["rewritten_text"] for chunk in processed_chunks[i:i + BATCH_SIZE]]
                batch_embeddings = self.embedding_service.get_embeddings_batch(batch)
                embeddings.extend(batch_embeddings)
                del batch
                del batch_embeddings
            
            # Store in vector database in batches
            print("Storing in vector database...")
            for i in tqdm(range(0, len(processed_chunks), BATCH_SIZE), desc="Storing chunks"):
                batch_chunks = processed_chunks[i:i + BATCH_SIZE]
                batch_embeddings = embeddings[i:i + BATCH_SIZE]
                self.vector_store_service.store_chunks(
                    document_id,
                    batch_chunks,
                    batch_embeddings,
                    metadata
                )
                del batch_chunks
                del batch_embeddings
            
            # Create document object
            document = Document(
                id=document_id,
                filename=file_path.name,
                content="\n".join(text_chunks),
                chunks=[chunk["rewritten_text"] for chunk in processed_chunks],
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_size=file_path.stat().st_size,
                mime_type=doc.metadata.get("format", "application/pdf"),
                page_count=total_pages,
                summary="\n".join(chunk["summary"] for chunk in processed_chunks)
            )
            
            return document
            
        except Exception as e:
            print(f"Failed to process document: {str(e)}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _process_chunk_batch(
        self,
        chunks: List[str],
        previous_summary: Optional[str] = None,
        previous_end_context: Optional[str] = None
    ) -> List[Dict]:
        """Process a batch of chunks in parallel."""
        futures = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            future = self.executor.submit(
                self.summarizer_service.process_chunk,
                chunk,
                previous_summary,
                previous_end_context
            )
            futures.append(future)
        
        return [future.result() for future in futures]
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using optimized regex-based splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Use regex for more efficient chunk boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > CHUNK_SIZE:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def __del__(self):
        """Cleanup when the processor is destroyed."""
        self.executor.shutdown(wait=True)
    
    def search_documents(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for similar documents using the query text.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with their metadata
            
        Raises:
            Exception: If search operation fails
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.get_embeddings(query)
            
            # Search vector store
            results = self.vector_store_service.search(
                query_embedding,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            print(f"Failed to search documents: {str(e)}")
            raise

    def search_query(self, query: str, matched_records:list[str]):

        return self.summarizer_service.search_query(query, matched_records)

