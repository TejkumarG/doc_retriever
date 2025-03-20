"""
Tests for the EmbeddingService class.
"""
import pytest
import time
from typing import List
from doc_retriever.services.embedding_service import EmbeddingService

def test_get_embeddings(embedding_service, sample_texts):
    """Test the get_embeddings method."""
    # Test with valid text
    text = sample_texts[0]
    embedding = embedding_service.get_embeddings(text)
    
    # Verify embedding properties
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # Test with empty text
    with pytest.raises(ValueError):
        embedding_service.get_embeddings("")
    
    # Test with whitespace text
    with pytest.raises(ValueError):
        embedding_service.get_embeddings("   ")

def test_get_embeddings_batch(embedding_service, sample_texts):
    """Test the get_embeddings_batch method."""
    # Test with valid texts
    embeddings = embedding_service.get_embeddings_batch(sample_texts)
    
    # Verify batch results
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)
    assert all(all(isinstance(x, float) for x in emb) for emb in embeddings)
    
    # Test with empty list
    with pytest.raises(ValueError):
        embedding_service.get_embeddings_batch([])

@pytest.mark.benchmark
def test_embedding_performance(embedding_service, sample_texts):
    """Benchmark the embedding generation performance."""
    # Test single embedding performance
    start_time = time.time()
    embedding_service.get_embeddings(sample_texts[0])
    single_time = time.time() - start_time
    
    # Test batch embedding performance
    start_time = time.time()
    embedding_service.get_embeddings_batch(sample_texts)
    batch_time = time.time() - start_time
    
    # Log performance metrics
    print(f"\nPerformance Metrics:")
    print(f"Single embedding time: {single_time:.2f} seconds")
    print(f"Batch embedding time: {batch_time:.2f} seconds")
    print(f"Average time per embedding in batch: {batch_time/len(sample_texts):.2f} seconds")

def test_embedding_consistency(embedding_service, sample_texts):
    """Test that embeddings are consistent for the same input."""
    # Generate embeddings twice for the same text
    text = sample_texts[0]
    embedding1 = embedding_service.get_embeddings(text)
    embedding2 = embedding_service.get_embeddings(text)
    
    # Verify embeddings are similar (using cosine similarity)
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity = cosine_similarity(
        np.array(embedding1).reshape(1, -1),
        np.array(embedding2).reshape(1, -1)
    )[0][0]
    
    assert similarity > 0.99  # Should be very similar

def test_embedding_dimension(embedding_service, sample_texts):
    """Test that embeddings have the correct dimension."""
    from doc_retriever.config.settings import DIMENSION
    
    # Test single embedding
    embedding = embedding_service.get_embeddings(sample_texts[0])
    assert len(embedding) == DIMENSION
    
    # Test batch embeddings
    embeddings = embedding_service.get_embeddings_batch(sample_texts)
    assert all(len(emb) == DIMENSION for emb in embeddings) 