"""
Main entry point for the Document Retriever application.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from doc_retriever.api.app import main

if __name__ == "__main__":
    main() 