"""
Service for handling text embeddings using LiteLLM.
"""
import json
import os
from typing import List, Optional, Dict, Any

from langchain_core.embeddings import Embeddings
from litellm import embedding
from tenacity import retry, stop_after_attempt, wait_exponential

from doc_retriever.config.settings import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    MAX_RETRIES,
    RETRY_DELAY
)


class CustomLiteLLMEmbeddings(Embeddings):
    """Custom implementation of LiteLLM embeddings that follows Langchain's interface."""

    def __init__(
            self,
            config_path: Optional[str] = None,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            **kwargs: Any,
    ):
        """Initialize the embeddings with settings loaded from configuration.

        Args:
            config_path: Optional path to the configuration file. If not provided, uses default path.
            model: Optional model name to use for embeddings. Overrides config if provided.
            api_key: Optional API key to use for embeddings. Overrides config if provided.
            api_base: Optional base URL for the API. Overrides config if provided.
            **kwargs: Additional arguments to pass to the embedding call
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Extract embedding settings
        embedding_settings = self.config.get("settings", {}).get("embedding_settings", {})

        # Set properties from config or direct parameters
        self.model = model or embedding_settings.get("embedding_model", "text-embedding-ada-002")
        self.api_key = api_key
        self.base_url = api_base or embedding_settings.get("embedding_base_url", None)
        self.kwargs = kwargs

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from file or use default."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)

        # If no config path provided or file doesn't exist, use hardcoded default config
        return {
            "settings": {
                "embedding_settings": {
                    "embedding_size": 1024,
                    "embedding_model": "text-embedding-ada-002",
                    "chunk_size": 2048,
                    "chunk_overlap": 50,
                    "k": 3
                }
            }
        }

    def _get_embedding_config(self) -> dict:
        """Get the configuration for the embedding call."""
        config = {
            "model": self.model,
            "api_key": self.api_key,
            **self.kwargs
        }
        if self.base_url:
            config["api_base"] = self.base_url
        return config

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text
        """
        embeddings = []
        for text in texts:
            response = embedding(
                input=text,
                **self._get_embedding_config()
            )
            embeddings.append(response["data"][0]["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query.

        Args:
            text: The text to embed

        Returns:
            Embeddings for the text
        """
        response = embedding(
            input=text,
            **self._get_embedding_config()
        )
        return response["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of documents.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text
        """
        # Note: LiteLLM doesn't have async support yet, so we use sync version
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a query.

        Args:
            text: The text to embed

        Returns:
            Embeddings for the text
        """
        # Note: LiteLLM doesn't have async support yet, so we use sync version
        return self.embed_query(text)

class EmbeddingService:
    """Service for generating text embeddings using LiteLLM."""
    
    def __init__(self):
        """Initialize the EmbeddingService with LiteLLM embeddings."""
        self.embeddings = CustomLiteLLMEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            api_base="https://api.openai.com/v1"
        )

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def get_embeddings(
        self,
        text: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Generate embeddings for a given text using LiteLLM.
        
        Args:
            text: The text to generate embeddings for
            trace_id: Optional Langfuse trace ID for tracking
            metadata: Optional metadata to attach to the trace
            
        Returns:
            List of embedding values
            
        Raises:
            ValueError: If the input text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_query(text)
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for
            trace_id: Optional Langfuse trace ID for tracking
            metadata: Optional metadata to attach to the trace
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If the input list is empty
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            return embeddings
            
        except Exception as e:
            print(f"Error processing texts: {str(e)}")
            raise