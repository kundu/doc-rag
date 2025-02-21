# PDF Embedding and Search System

This system processes PDF files, generates embeddings using a local AI model, and stores them in PostgreSQL for semantic search capabilities. It allows you to search through multiple PDFs and find relevant content based on semantic similarity.

## Features

- Bulk PDF processing
- Automatic text extraction from PDFs
- Semantic search across all processed documents
- Progress tracking and beautiful console output
- Skips already processed files
- Detailed processing summaries
- Similarity scores for search results

## Prerequisites

1. Python 3.8+
2. PostgreSQL 16+ with pgvector extension
3. Local AI embedding service running on port 1234
4. `pip` and `venv` modules

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PostgreSQL and pgvector extension:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-16 postgresql-16-pgvector

# After installation, enable the extension
sudo -u postgres psql -d pdf_storage -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

5. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configurations
nano .env  # or use any text editor
```

## Configuration

Update the following variables in your `.env` file:

```env
# Database Configuration
DB_USER=postgres          # Your PostgreSQL username
DB_PASSWORD=your_password # Your PostgreSQL password
DB_HOST=localhost        # Database host
DB_PORT=5432            # Database port
DB_NAME=pdf_storage     # Database name

# AI Embedding Service Configuration
AI_API_URL=http://127.0.0.1:1234/v1/embeddings  # Your embedding service URL
AI_MODEL=text-embedding-nomic-embed-text-v1.5@f32  # Model name
```

## Usage

1. Place your PDF files in the `pdf` directory:
```bash
mkdir -p pdf
cp your_pdfs/*.pdf pdf/
```

2. Run the processor:
```bash
python pdf_processor.py
```

The script will:
- Process all new PDF files in the `pdf` directory
- Skip already processed files
- Show progress for each file
- Display a summary table after processing
- Perform a sample search

3. Search through processed PDFs:
```python
from pdf_processor import search_similar_content

# Search for specific content
results = search_similar_content("your search query")
```

## Project Structure

```
.
├── pdf/                  # Directory for PDF files
├── models.py            # Database models
├── pdf_processor.py     # Main processing script
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (create from .env.example)
└── README.md           # This file
```

## Database Schema

- `pdf_files`: Stores PDF file metadata
  - id (Primary Key)
  - filename
  - file_path
  - upload_date
  - file_size
  - total_pages

- `pdf_embeddings`: Stores page content and embeddings
  - id (Primary Key)
  - pdf_file_id (Foreign Key)
  - page_number
  - page_content
  - embedding (JSON array)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Your chosen license]

## Troubleshooting

1. **Database Connection Issues**
   - Ensure PostgreSQL is running
   - Verify database credentials in `.env`
   - Check if pgvector extension is installed

2. **Embedding Service Issues**
   - Verify the embedding service is running
   - Check the API URL in `.env`
   - Ensure the model name is correct

3. **PDF Processing Issues**
   - Ensure PDF files are readable
   - Check file permissions
   - Verify sufficient disk space

## Support

For support, please [create an issue](your-repo-issues-url) or contact [your-contact-info]. 