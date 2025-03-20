# Document Retriever

A powerful document processing and retrieval system that allows you to upload PDF documents, process them, and search through them using natural language queries.

## Project Structure

```
doc_retriever/
├── api/            # API and web interface components
├── config/         # Configuration settings
├── core/           # Core business logic
├── models/         # Data models
├── services/       # Service layer implementations
└── utils/          # Utility functions
├── data/           # Data storage directory
├── tests/          # Test files
├── .env            # Environment variables
├── .env.example    # Example environment variables
├── requirements.txt # Project dependencies
├── requirements-dev.txt # Development dependencies
└── run.py          # Main entry point
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd doc_retriever
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

To run the application:

```bash
python run.py
```

This will start the Streamlit web interface. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

## Features

- PDF document upload and processing
- Natural language document search
- Document summarization
- Vector-based similarity search
- Document metadata extraction
- Secure file handling

## Development

For development, install additional dependencies:

```bash
pip install -r requirements-dev.txt
```

## Testing

Run tests using:

```bash
python -m pytest tests/
```

## Docker Support

To run the application using Docker:

```bash
docker-compose up
```

## License

[Your License Here] 