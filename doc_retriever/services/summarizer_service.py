"""
Service for generating text summaries using LiteLLM.
"""
from typing import Dict, Optional, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from litellm import completion, RateLimitError
import json
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.chat_models import ChatLiteLLM
import litellm
import streamlit as st

from doc_retriever.config.settings import (
    OPENAI_API_KEY,
    MAX_RETRIES,
    RETRY_DELAY,
    TEMPERATURE,
    CHAT_MODEL
)

# Enable verbose logging for litellm
litellm.success_callback = ['langfuse']
litellm.failure_callback = ['langfuse']


class SummarizerService:
    """Service for generating text summaries using LiteLLM."""
    
    def __init__(self):
        """Initialize the SummarizerService with LiteLLM."""
        self.llm = ChatLiteLLM(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            api_base="https://api.openai.com/v1",
            temperature=TEMPERATURE
        )
        # self.langfuse = LangfuseService()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True
    )
    def process_chunk(
        self,
        chunk: str,
        previous_summary: Optional[str] = None,
        previous_end_context: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Process a text chunk to generate rewritten content, summary, and end context.
        
        Args:
            chunk: The text chunk to process
            previous_summary: Summary from the previous chunk
            previous_end_context: End context from the previous chunk
            trace_id: Optional Langfuse trace ID for tracking
            metadata: Optional metadata to attach to the trace
            
        Returns:
            Dictionary containing rewritten text, summary, and end context
            
        Raises:
            ValueError: If the input chunk is empty
        """
        if not chunk or not chunk.strip():
            raise ValueError("Input chunk cannot be empty")
            
        try:
            # Create trace if not provided
            # if not trace_id:
            #     trace_id = self.langfuse.create_trace(
            #         name="summarization_operation",
            #         metadata=metadata
            #     )
            
            # Prepare context-aware input
            context = ""
            if previous_summary and previous_end_context:
                context = f"Previous summary: {previous_summary}\nPrevious ending: {previous_end_context}\n\n"
            
            full_text = context + chunk

            response = self.llm.invoke([
                SystemMessage(
                    content="You are a highly skilled assistant that can rewrite text for clarity while maintaining key information, summarize key points, and extract the final key points that connect to the next section."),
                HumanMessage(content=f"""
                        Given the following text, perform three tasks:
                        1. Rewrite the text to improve clarity while keeping key information intact.
                        2. Summarize the key points concisely.
                        3. Identify the final key points that connect to the next section.

                        Text:
                        {full_text}

                        Return the results in the following JSON format:
                        {{
                            "rewritten_text": "<your rewritten text>",
                            "summary": "<your summary>",
                            "end_context": "<your end context>"
                        }}
                        """)
            ])

            result = response.content.strip()
            result_dict = json.loads(result)
            
            return result_dict
            
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            raise

    def _get_completion_config(self) -> Dict[str, Any]:
        """Return the configuration for LiteLLM completion."""
        return {
            "model": CHAT_MODEL,
            "api_key": OPENAI_API_KEY,
            "api_base": "https://api.openai.com/v1",
            "temperature": TEMPERATURE
        }

    def _invoke(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Invoke the chat model with the given messages.

        Args:
            messages: List of messages to send to the model
            stop: Optional list of stop sequences

        Returns:
            Dictionary containing the model's response
        """
        prompt = self._convert_messages_to_prompt(messages)
        response = completion(
            messages=[{"role": "user", "content": prompt}],
            **self._get_completion_config()
        )
        # Access the generated text from the response
        generated_text = response.choices[0].message.content
        return {"text": generated_text}

    async def _ainvoke(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Asynchronously invoke the chat model.

        Args:
            messages: List of messages to send to the model
            stop: Optional list of stop sequences

        Returns:
            Dictionary containing the model's response
        """
        # Note: LiteLLM doesn't have async support yet, so we use sync version
        return self._invoke(messages, stop)

    def search_query(self, query: str, matched_records: list):
        input_text = f"""
           You are an AI assistant helping to provide relevant responses based on user queries.
           Given the query and matched records, generate a well-structured response.
           
           Context:
            {"\n".join(matched_records)}
            
            Question: {query}
           """


        response = self.llm.invoke(
            input_text
        )

        return response.content

def main():
    # Initialize session state
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []

    # Your existing code... 