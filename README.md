# PDF Embedding Storage System

This system processes PDF files, generates embeddings using a local AI model, and stores them in PostgreSQL for semantic search capabilities.

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Local AI embedding service running on port 1234

## Setup

1. Install PostgreSQL and enable pgvector extension:
```sql
CREATE EXTENSION vector;
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
Copy the `.env.example` file to `.env` and update the values as needed.

## Usage

1. Process a PDF file:
```python
from pdf_processor import process_pdf

process_pdf("path/to/your/file.pdf")
```

2. Search for similar content:
```python
from pdf_processor import search_similar_content

results = search_similar_content("your search query")
for result in results:
    print(f"File: {result['filename']}")
    print(f"Page: {result['page']}")
    print(f"Preview: {result['content']}")
```

## Database Schema

- `pdf_files`: Stores PDF file metadata
  - id
  - filename
  - file_path
  - upload_date
  - file_size
  - total_pages

- `pdf_embeddings`: Stores page content and embeddings
  - id
  - pdf_file_id
  - page_number
  - page_content
  - embedding (vector) 