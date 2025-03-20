"""
Service for handling Langfuse tracking and monitoring.
"""
from typing import Optional, Dict, Any
from langfuse import Langfuse

from doc_retriever.config.settings import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST
)

class LangfuseService:
    """Service for tracking and monitoring LangChain operations using Langfuse."""
    
    def __init__(self):
        """Initialize the LangfuseService with Langfuse client."""
        self.langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
    
    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new trace for tracking operations.
        
        Args:
            name: Name of the trace
            metadata: Optional metadata to attach to the trace
            
        Returns:
            Trace ID
        """
        trace = self.langfuse.trace(
            name=name,
            metadata=metadata or {}
        )
        return trace.id
    
    def log_embedding(
        self,
        trace_id: str,
        input_text: str,
        embedding: list,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an embedding operation.
        
        Args:
            trace_id: ID of the trace to log to
            input_text: Input text that was embedded
            embedding: Generated embedding vector
            metadata: Optional metadata to attach to the observation
        """
        self.langfuse.generation(
            trace_id=trace_id,
            name="embedding",
            input={"text": input_text},
            output={"embedding": embedding},
            metadata=metadata or {}
        )
    
    def log_summarization(
        self,
        trace_id: str,
        input_text: str,
        output: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a summarization operation.
        
        Args:
            trace_id: ID of the trace to log to
            input_text: Input text that was summarized
            output: Dictionary containing rewritten_text, summary, and end_context
            metadata: Optional metadata to attach to the observation
        """
        self.langfuse.generation(
            trace_id=trace_id,
            name="summarization",
            input={"text": input_text},
            output=output,
            metadata=metadata or {}
        )
    
    def get_trace_metadata(self, trace_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific trace.
        
        Args:
            trace_id: ID of the trace to get metadata for
            
        Returns:
            Dictionary containing trace metadata
        """
        trace = self.langfuse.fetch_trace(trace_id)
        return trace.metadata or {}
    
    def get_trace_observations(self, trace_id: str) -> list:
        """
        Get all observations for a specific trace.
        
        Args:
            trace_id: ID of the trace to get observations for
            
        Returns:
            List of observations
        """
        trace = self.langfuse.fetch_trace(trace_id)
        return trace.observations
    
    def flush(self) -> None:
        """
        Flush all pending observations to Langfuse.
        This is important to call before application shutdown.
        """
        self.langfuse.flush() 