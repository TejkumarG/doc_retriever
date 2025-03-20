import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Milvus configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")  # Changed to use localhost IP
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "documents"

def init_milvus():
    """Initialize Milvus connection and create collection if it doesn't exist."""
    try:
        # Connect to Milvus
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_number", dtype=DataType.INT64),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="original_chunk", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="rewritten_chunk", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="end_context", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="previous_summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="previous_end_context", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields=fields, description="Document collection with enhanced metadata")
        
        # Create collection if it doesn't exist
        if not utility.has_collection(COLLECTION_NAME):
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            # Create index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
        
        return Collection(name=COLLECTION_NAME)
    
    except Exception as e:
        print(f"Error connecting to Milvus: {str(e)}")
        raise

def close_milvus():
    """Close Milvus connection."""
    try:
        connections.disconnect("default")
    except Exception as e:
        print(f"Error closing Milvus connection: {str(e)}") 