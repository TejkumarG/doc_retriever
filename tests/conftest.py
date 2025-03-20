"""
Test configuration and fixtures for the Document Retriever application.
"""
from typing import List
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import AIMessage

from doc_retriever.services.embedding_service import EmbeddingService
from doc_retriever.services.summarizer_service import SummarizerService


@pytest.fixture
def embedding_service():
    """Fixture to provide an EmbeddingService instance."""
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
        mock_instance = MagicMock()
        mock_instance.embed_query.return_value = [0.1] * 1536
        mock_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_embeddings.return_value = mock_instance
        
        with patch('doc_retriever.services.embedding_service.OpenAIEmbeddings', return_value=mock_instance):
            return EmbeddingService()

@pytest.fixture
def summarizer_service():
    """Fixture to provide a SummarizerService instance."""
    with patch('langchain_openai.ChatOpenAI') as mock_chat:
        mock_instance = MagicMock()
        def mock_response(messages):
            if "rewrite" in messages[0].content.lower():
                return AIMessage(content="Rewritten text")
            elif "summarize" in messages[0].content.lower():
                return AIMessage(content="Summary text")
            else:
                return AIMessage(content="End context")
        
        mock_instance.invoke.side_effect = mock_response
        mock_chat.return_value = mock_instance
        
        with patch('doc_retriever.services.summarizer_service.ChatOpenAI', return_value=mock_instance):
            return SummarizerService()

@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture to provide sample texts for testing."""
    return [
        "This is a sample text for testing embeddings.",
        "Another sample text with different content.",
        "A third sample text to test batch processing."
    ]

@pytest.fixture
def sample_chunk() -> str:
    """Fixture to provide a sample text chunk for testing."""
    return """
    The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet
    at least once. Pangrams are often used to display font samples and test keyboards and printers. While
    this particular pangram is well-known, there are many others that serve the same purpose while telling
    different stories or conveying different messages.
    """ 