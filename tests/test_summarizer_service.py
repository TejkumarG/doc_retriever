"""
Tests for the SummarizerService class.
"""
import pytest
import time
from typing import Dict
from doc_retriever.services.summarizer_service import SummarizerService

def test_process_chunk(summarizer_service, sample_chunk):
    """Test the basic functionality of process_chunk."""
    result = summarizer_service.process_chunk(sample_chunk)
    
    # Verify all required keys are present
    assert "rewritten_text" in result
    assert "summary" in result
    assert "end_context" in result
    
    # Verify all values are non-empty strings
    assert isinstance(result["rewritten_text"], str) and result["rewritten_text"]
    assert isinstance(result["summary"], str) and result["summary"]
    assert isinstance(result["end_context"], str) and result["end_context"]

def test_process_chunk_with_context(summarizer_service, sample_chunk):
    """Test the process_chunk method with previous context."""
    # Test with previous summary and context
    previous_summary = "Previous section summary"
    previous_end_context = "Previous section ending points"
    
    result = summarizer_service.process_chunk(
        sample_chunk,
        previous_summary=previous_summary,
        previous_end_context=previous_end_context
    )
    
    # Verify all required keys are present with non-empty values
    assert isinstance(result["rewritten_text"], str) and result["rewritten_text"]
    assert isinstance(result["summary"], str) and result["summary"]
    assert isinstance(result["end_context"], str) and result["end_context"]

@pytest.mark.benchmark
def test_summarizer_performance(summarizer_service, sample_chunk):
    """Benchmark the summarizer performance."""
    # Test chunk processing performance
    start_time = time.time()
    result = summarizer_service.process_chunk(sample_chunk)
    end_time = time.time()
    
    # Processing should complete within a reasonable time (e.g., 5 seconds)
    assert end_time - start_time < 5.0
    assert result is not None

def test_summarizer_quality(summarizer_service, sample_chunk):
    """Test the quality of the summarizer output."""
    result = summarizer_service.process_chunk(sample_chunk)
    
    # Verify summary is shorter than original
    assert len(result["summary"]) < len(sample_chunk)
    
    # Verify rewritten text maintains key information
    assert len(result["rewritten_text"]) > 0
    assert len(result["rewritten_text"]) <= len(sample_chunk) * 1.5  # Should not be much longer
    
    # Verify end context is present and non-empty
    assert len(result["end_context"]) > 0
    assert len(result["end_context"]) <= len(result["rewritten_text"])

def test_summarizer_consistency(summarizer_service, sample_chunk):
    """Test that the summarizer produces consistent results for the same input."""
    # Process the same chunk twice
    result1 = summarizer_service.process_chunk(sample_chunk)
    result2 = summarizer_service.process_chunk(sample_chunk)
    
    # Results should be consistent
    assert result1["rewritten_text"] == result2["rewritten_text"]
    assert result1["summary"] == result2["summary"]
    assert result1["end_context"] == result2["end_context"]

def test_summarizer_error_handling(summarizer_service):
    """Test error handling in the summarizer service."""
    # Test with None values
    with pytest.raises(ValueError):
        summarizer_service.process_chunk(None)
    
    # Test with very long text (should still work)
    long_text = "Test text. " * 1000
    result = summarizer_service.process_chunk(long_text)
    assert result is not None 