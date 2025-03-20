"""
Service for managing vector storage using Milvus.
"""
from typing import List, Dict, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tenacity import retry, stop_after_attempt, wait_exponential

from doc_retriever.config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    DIMENSION,
    MAX_RETRIES,
    RETRY_DELAY
)

class VectorStoreService:
    """Service for managing vector storage using Milvus."""
    
    def __init__(self):
        """Initialize the vector store service."""
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """Establish connection to Milvus server."""
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception as e:
            print(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def _disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect("default")
        except Exception as e:
            print(f"Failed to disconnect from Milvus: {str(e)}")
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper schema."""
        if not utility.has_collection(COLLECTION_NAME):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
                FieldSchema(name="rewritten_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="end_context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields=fields, description="Document chunks with embeddings")
            self.collection = Collection(name=COLLECTION_NAME, schema=schema)
            
            # Create index for vector search
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(name=COLLECTION_NAME)
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def store_chunks(
        self,
        document_id: str,
        chunks_data: List[Dict[str, str]],
        embeddings: List[List[float]],
        metadata: Dict
    ) -> None:
        """
        Store document chunks with their embeddings in Milvus.
        
        Args:
            document_id: Unique identifier for the document
            chunks_data: List of dictionaries containing chunk data
            embeddings: List of embedding vectors
            metadata: Additional metadata for the chunks
            
        Raises:
            Exception: If storage operation fails
        """
        try:
            # Prepare entities for insertion
            entities = []
            for i, (chunk_data, embedding) in enumerate(zip(chunks_data, embeddings)):
                entity = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "embedding": embedding,
                    "rewritten_text": chunk_data["rewritten_text"],
                    "summary": chunk_data["summary"],
                    "end_context": chunk_data["end_context"],
                    "metadata": metadata
                }
                entities.append(entity)
            
            # Insert entities
            self.collection.insert(entities)
            
        except Exception as e:
            print(f"Failed to store chunks: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def search(
        self,
        query_embedding: List[float],
        limit: int = 3,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results to return
            document_id: Optional document ID to filter results
            
        Returns:
            List of similar chunks with their metadata
            
        Raises:
            Exception: If search operation fails
        """
        try:
            # Load collection
            self.collection.load()
            
            # Prepare search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Prepare expression for filtering
            expr = f'document_id == "{document_id}"' if document_id else None
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["document_id", "chunk_index", "rewritten_text", "summary", "end_context", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "score": hit.score,
                        "document_id": hit.entity.get("document_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "rewritten_text": hit.entity.get("rewritten_text"),
                        "summary": hit.entity.get("summary"),
                        "end_context": hit.entity.get("end_context"),
                        "metadata": hit.entity.get("metadata")
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Failed to search chunks: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup when the service is destroyed."""
        self._disconnect() 