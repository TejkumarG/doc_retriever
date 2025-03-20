import PyPDF2
import io
import re

def process_pdf(pdf_file):
    """
    Process a PDF file and convert it to Markdown format.
    
    Args:
        pdf_file: A file-like object containing the PDF data
        
    Returns:
        str: The markdown content of the PDF
    """
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text_content = []
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
        
        # Combine all text
        full_text = "\n".join(text_content)
        
        # Convert to markdown format
        markdown_content = convert_to_markdown(full_text)
        
        return markdown_content
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def convert_to_markdown(text):
    """
    Convert extracted text to markdown format.
    This is a basic implementation that can be enhanced based on specific needs.
    
    Args:
        text (str): The extracted text from PDF
        
    Returns:
        str: The markdown formatted text
    """
    # Split text into lines
    lines = text.split('\n')
    
    # Process each line
    markdown_lines = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            markdown_lines.append('')
            continue
            
        # Detect headings (basic implementation)
        if line.isupper() and len(line.strip()) < 100:
            markdown_lines.append(f"## {line.strip()}")
        else:
            markdown_lines.append(line.strip())
    
    return '\n'.join(markdown_lines) 