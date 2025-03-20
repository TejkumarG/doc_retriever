"""
Streamlit application for the Document Retriever.
"""
import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Dict
import json
import mimetypes
import shutil
from datetime import datetime

from doc_retriever.core.document_processor import DocumentProcessor
from doc_retriever.models.document import Document
from doc_retriever.config.settings import (
    MAX_FILE_SIZE,
    ALLOWED_MIME_TYPES,
    MAX_CONCURRENT_REQUESTS,
    UPLOADS_DIR
)

# Initialize document processor
@st.cache_resource
def get_document_processor():
    """Get or create the document processor instance."""
    return DocumentProcessor()

def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded file to the uploads directory with a unique name.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path: Path to the saved file
    """
    # Create a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{uploaded_file.name}"
    save_path = UPLOADS_DIR / unique_filename
    
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return save_path

def validate_file(file) -> bool:
    """
    Validate uploaded file for security and compatibility.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file:
        return False
        
    # Check file size
    if file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
        return False
    
    # Check MIME type
    mime_type = mimetypes.guess_type(file.name)[0]
    if mime_type not in ALLOWED_MIME_TYPES:
        st.error(f"File type {mime_type} not supported. Please upload a PDF file.")
        return False
    
    # Check concurrent processing limit
    if st.session_state.processing_count >= MAX_CONCURRENT_REQUESTS:
        st.error(f"Maximum number of concurrent processing requests ({MAX_CONCURRENT_REQUESTS}) reached.")
        return False
    
    return True

def process_uploaded_file(file) -> Document:
    """Process an uploaded file and return the document object."""
    if not validate_file(file):
        raise ValueError("File validation failed")
    
    st.session_state.processing_count += 1
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        processor = get_document_processor()
        document = processor.process_document(tmp_file_path)
        return document
    finally:
        # Clean up temporary file
        Path(tmp_file_path).unlink()
        st.session_state.processing_count -= 1

def display_document_info(document: Document):
    """Display document information in a formatted way."""
    st.subheader("Document Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Filename:**", document.filename)
        st.write("**File Size:**", f"{document.file_size / 1024:.2f} KB")
        st.write("**MIME Type:**", document.mime_type)
    with col2:
        st.write("**Created At:**", document.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Updated At:**", document.updated_at.strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Page Count:**", document.page_count)
    
    st.subheader("Summary")
    st.write(document.summary)

def display_search_results(results: List[Dict]):
    """Display search results in a formatted way."""
    st.subheader("Search Results")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"Result {i} (Score: {result['score']:.4f})"):
            st.write("**Document:**", result["metadata"]["filename"])
            st.write("**Chunk Index:**", result["chunk_index"])
            st.write("**Summary:**", result["summary"])
            st.write("**Context:**", result["end_context"])
            st.write("**Content:**", result["rewritten_text"])

def main():
    """Main application function."""
    # Initialize session state
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    if "processing_count" not in st.session_state:
        st.session_state.processing_count = 0

    st.set_page_config(
        page_title="Document Retriever",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document Retriever")
    st.write("Upload documents and search through them using natural language queries.")
    
    # File upload section
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Maximum file size: {MAX_FILE_SIZE/1024/1024:.1f}MB"
    )
    
    if uploaded_file and not st.session_state.processed_documents and validate_file(uploaded_file):
        try:
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            
            # Process the document
            processor = get_document_processor()
            st.session_state.processing_count += 1
            
            with st.spinner("Processing document..."):
                document = processor.process_document(str(file_path))
                st.session_state.processed_documents.append(document)
                st.session_state.current_document = document
            
            st.success("Document processed successfully!")
            st.session_state.processing_count -= 1
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.session_state.processing_count -= 1
    
    # Search section
    if st.session_state.processed_documents:
        st.header("Search Documents")
        query = st.text_input("Enter your search query")
        
        if query:
            processor = get_document_processor()
            with st.spinner("Searching..."):
                results = processor.search_documents(query)[::-1]

                chucks_data = [individual['rewritten_text'] for individual in results]

                answer = processor.search_query(query, chucks_data)

                st.subheader("Answer")
                st.write(answer)
                
                if results:
                    for i, result in enumerate(results, 1):
                        st.subheader(f"Result {i}")
                        st.write(f"**File:** {result['metadata'].get('filename', 'Unknown')}")
                        st.write(f"**Score:** {result['score']:.2f}")
                        with st.expander("Show Content"):
                            st.write("**Summary:**")
                            st.write(result["summary"])
                            st.write("**Rewritten Text:**")
                            st.write(result["rewritten_text"])
                else:
                    st.info("No results found.")
    
    # Display processed documents
    if st.session_state.processed_documents:
        st.header("Processed Documents")
        for doc in st.session_state.processed_documents:
            with st.expander(f"📄 {doc.filename}"):
                display_document_info(doc)
                
                # Add download button for document data
                doc_data = doc.to_dict()
                st.download_button(
                    label="Download Document Data",
                    data=json.dumps(doc_data, indent=2),
                    file_name=f"{doc.filename}_data.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main() 