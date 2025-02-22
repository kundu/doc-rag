# PDF Embedding and Search System

This system processes PDF files, generates embeddings using a local AI model, and stores them in PostgreSQL for semantic search capabilities. It features advanced OCR capabilities, semantic chunking, and hierarchical document understanding.

## Features

- Bulk PDF processing with progress tracking
- Automatic text and image extraction from PDFs
- Advanced OCR with image preprocessing:
  - Automatic image format detection and conversion
  - Image enhancement for better OCR quality
  - Support for multiple image formats (JPEG, PNG, GIF, BMP, TIFF)
  - Confidence scoring for OCR results
  - Automatic image resizing and contrast enhancement
- AI-powered text embeddings generation for semantic understanding
- Hierarchical document processing:
  - Document-level embeddings
  - Section-level chunking
  - Paragraph-level analysis
  - Element-level processing (tables, images, forms)
- Semantic metadata extraction:
  - Entity recognition
  - Key phrase extraction
  - Sentiment analysis
- Vector-based semantic search across all processed documents
- Detailed processing summaries and error handling
- Progress tracking with rich console output
- Efficient storage of embeddings in PostgreSQL using JSON format

## Prerequisites

1. Python 3.8+
2. PostgreSQL 16+ with pgvector extension
3. Local AI embedding service running on port 1234
4. Tesseract OCR engine
5. `pip` and `venv` modules

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

4. Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-eng  # English language pack

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

5. Install PostgreSQL and pgvector extension:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-16 postgresql-16-pgvector

# After installation, enable the extension
sudo -u postgres psql -d pdf_storage -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

6. Set up environment variables:
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

# QA System Configuration
QA_API_URL=http://localhost:1234/v1/chat/completions
QA_MODEL=qwen2-0.5b-instruct
```

## Usage

1. Place your PDF files in the `pdf` directory:
```bash
mkdir -p pdf
cp your_pdfs/*.pdf pdf/
```

2. Run the processor:
```bash
bash -c 'source venv/bin/activate && python pdf_processor.py'
```

3. Run the QA system:
```bash
bash -c 'source venv/bin/activate && python qa_system.py'
```

The script will:
- Process all new PDF files in the `pdf` directory
- Extract and OCR text from both document content and images
- Generate hierarchical embeddings
- Extract semantic metadata
- Store all information in the database
- Show detailed progress and processing summaries

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
├── models.py            # Database models and enums
├── pdf_processor.py     # Main processing script
├── qa_system.py         # Question answering system
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
└── README.md           # This file
```

## Database Schema

- `pdf_files`: Stores PDF file metadata and document-level embeddings
  - id (Primary Key)
  - filename
  - file_path
  - upload_date
  - file_size
  - total_pages
  - pdf_metadata (JSON)
  - document_embedding (JSON)

- `pdf_embeddings`: Stores content embeddings and metadata
  - id (Primary Key)
  - pdf_file_id (Foreign Key)
  - page_number
  - hierarchy_level (DOCUMENT, SECTION, PARAGRAPH, ELEMENT)
  - content_type (TEXT, TABLE, IMAGE, FORM)
  - page_content
  - embedding (JSON)
  - position (JSON)
  - content_format (JSON)
  - context
  - semantic_metadata (JSON)
  - confidence

## Error Handling

The system includes comprehensive error handling:
- Graceful handling of corrupted PDFs
- Recovery from OCR failures
- Fallback chunking for large documents
- Image format conversion and validation
- Detailed error reporting and logging

## Performance Optimization

- Efficient memory usage through streaming processing
- Image preprocessing for optimal OCR results
- Chunked processing of large documents
- Configurable chunk sizes and limits
- Background processing capabilities

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Troubleshooting

1. **Database Connection Issues**
   - Ensure PostgreSQL is running
   - Verify database credentials in `.env`
   - Check if pgvector extension is installed

2. **OCR Issues**
   - Verify Tesseract OCR is installed
   - Check image quality and format
   - Adjust preprocessing parameters if needed

3. **Embedding Service Issues**
   - Verify the embedding service is running
   - Check the API URL in `.env`
   - Ensure the model name is correct

4. **Memory Issues**
   - Adjust chunk sizes in the configuration
   - Process fewer files simultaneously
   - Check available system resources

## Support

For support, please [create an issue](https://github.com/kundu/doc-rag/issues/new) or contact [Sauvik Kundu](https://www.linkedin.com/in/sauvik-kundu).
 

## Visuals

![Processing Flow](output/processing_flow.png)
![System Architecture](output/system_architecture.png) 