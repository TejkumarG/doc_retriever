import re

def create_chunks(markdown_content, chunk_size=4000, overlap=200):
    """
    Split markdown content into chunks while preserving section structure.
    
    Args:
        markdown_content (str): The markdown formatted content
        chunk_size (int): Maximum size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # Split content into sections based on headings
    sections = re.split(r'(?m)^##\s+', markdown_content)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for section in sections:
        if not section.strip():
            continue
            
        # If this is the first section, it won't have a heading
        if not section.startswith('#'):
            section = f"## {section}"
            
        section_size = len(section)
        
        # If a single section is larger than chunk_size, split it
        if section_size > chunk_size:
            # If we have accumulated content, save it as a chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split the large section into smaller chunks
            words = section.split()
            temp_chunk = []
            temp_size = 0
            
            for word in words:
                word_size = len(word) + 1  # +1 for space
                if temp_size + word_size > chunk_size:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_size = word_size
                else:
                    temp_chunk.append(word)
                    temp_size += word_size
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            continue
        
        # If adding this section would exceed chunk_size, save current chunk
        if current_size + section_size > chunk_size:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(section)
        current_size += section_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks 