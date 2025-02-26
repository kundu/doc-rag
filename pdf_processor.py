import os
import json
from datetime import datetime
from dotenv import load_dotenv
import PyPDF2
import pdfplumber
import requests
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, PDFFile, PDFEmbedding, ContentType, HierarchyLevel
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import spacy
from transformers import AutoTokenizer
import re
from typing import List, Dict, Any
import imghdr

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# AI API configuration
AI_API_URL = os.getenv('AI_API_URL')
AI_MODEL = os.getenv('AI_MODEL')
POD_API_KEY = os.getenv('POD_API_KEY')

# Create database connection
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Create tables
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, context=None):
    """Get embedding from the AI API with optional context."""
    try:
        input_text = text
        if context:
            input_text = f"{context}: {text}"
            
        response = requests.post(
            AI_API_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {POD_API_KEY}"},
            json={
                "model": AI_MODEL,
                "input": input_text
            }
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        console.print(f"[red]Error getting embedding: {e}[/red]")
        return None

def extract_text_with_position(pdf_path, page_num):
    """Extract text with position information using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        words = page.extract_words()
        return [
            {
                'text': word['text'],
                'position': {
                    'x0': word['x0'],
                    'y0': word['top'],
                    'x1': word['x1'],
                    'y1': word['bottom']
                }
            }
            for word in words
        ]

def process_tables(pdf_path, page_num):
    """Extract and process tables from a page."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        for table in page.extract_tables():
            if table:
                # Convert table to string representation
                table_str = '\n'.join(['\t'.join([str(cell) for cell in row if cell]) for row in table if any(row)])
                tables.append({
                    'content': table_str,
                    'structure': table
                })
    return tables

def make_json_serializable(obj, _depth=0, _max_depth=20):
    """Convert PDF-specific objects to JSON-serializable types.
    
    Args:
        obj: The object to convert
        _depth: Current recursion depth (internal use)
        _max_depth: Maximum recursion depth to prevent infinite recursion
    """
    # Check recursion depth
    if _depth > _max_depth:
        return str(obj)
    
    try:
        # Try direct JSON serialization first
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError, ValueError):
        pass
    
    # Handle different types
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item, _depth + 1, _max_depth) for item in obj]
    elif isinstance(obj, dict):
        return {
            str(key): make_json_serializable(value, _depth + 1, _max_depth)
            for key, value in obj.items()
        }
    elif hasattr(obj, 'get_object'):  # Handle PDF objects with get_object method
        try:
            return make_json_serializable(obj.get_object(), _depth + 1, _max_depth)
        except:
            return str(obj)
    elif hasattr(obj, 'pdfObject'):  # Handle PDF objects with pdfObject attribute
        try:
            return make_json_serializable(obj.pdfObject, _depth + 1, _max_depth)
        except:
            return str(obj)
    elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
        try:
            return make_json_serializable(obj.__dict__, _depth + 1, _max_depth)
        except:
            return str(obj)
    
    # Convert to string as fallback
    return str(obj)

def process_images(pdf_path, page_num):
    """Extract and process embedded images from a page."""
    try:
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            
            # Extract images from the page
            image_list = page.images
            
            for img_obj in image_list:
                try:
                    # Get raw image data
                    stream = img_obj['stream']
                    if not stream:
                        continue

                    image_bytes = stream.get_data()
                    if not image_bytes:
                        continue

                    # Try to determine image format
                    img_format = imghdr.what(None, image_bytes)
                    
                    if not img_format:
                        # If format detection fails, try common formats
                        for format_name in ['jpeg', 'png', 'gif', 'bmp', 'tiff']:
                            try:
                                img = Image.open(io.BytesIO(image_bytes))
                                img.verify()
                                img_format = format_name
                                break
                            except:
                                continue
                    
                    if not img_format:
                        console.print(f"[yellow]Warning: Unrecognized image format in {pdf_path} page {page_num + 1}[/yellow]")
                        continue

                    # Create a fresh BytesIO object for the image
                    img_buffer = io.BytesIO(image_bytes)
                    img = Image.open(img_buffer)
                    
                    # Convert RGBA to RGB if necessary
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode == 'P':  # Convert palette images
                        img = img.convert('RGB')
                    
                    # Ensure good quality for OCR
                    # Resize if too small
                    min_size = 1000
                    ratio = max(min_size / img.width, min_size / img.height)
                    if ratio > 1:
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                    # Convert to grayscale for better OCR
                    img_gray = img.convert('L')
                    
                    # Improve contrast
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(img_gray)
                    img_gray = enhancer.enhance(2.0)
                    
                    # Perform OCR with improved settings
                    text = pytesseract.image_to_string(
                        img_gray,
                        config='--psm 3 --oem 3 -l eng'  # Automatic page segmentation with OEM LSTM OCR Engine
                    )
                    
                    # Get confidence scores
                    ocr_data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)
                    confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    if text.strip():
                        # Convert metadata to JSON-serializable format
                        metadata = {
                            'width': float(img_obj.get('width', 0)),
                            'height': float(img_obj.get('height', 0)),
                            'type': str(img_obj.get('type', '')),
                            'colorspace': make_json_serializable(img_obj.get('colorspace', '')),
                            'format': img_format,
                            'original_size': {
                                'width': img.width,
                                'height': img.height
                            },
                            'ocr_confidence': avg_confidence
                        }
                        
                        results.append({
                            'content': text.strip(),
                            'type': 'image',
                            'position': {
                                'x0': float(img_obj.get('x0', 0)),
                                'y0': float(img_obj.get('y0', 0)),
                                'x1': float(img_obj.get('x1', 0)),
                                'y1': float(img_obj.get('y1', 0))
                            },
                            'confidence': avg_confidence / 100.0,  # Convert to 0-1 range
                            'metadata': metadata
                        })
                    
                    # Clean up
                    img.close()
                    img_gray.close()
                    img_buffer.close()
                    
                except Exception as img_error:
                    console.print(f"[yellow]Warning: Could not process image in {pdf_path} page {page_num + 1}: {str(img_error)}[/yellow]")
                    continue
                    
        return results
    except Exception as e:
        console.print(f"[red]Error processing images in {pdf_path} page {page_num + 1}: {str(e)}[/red]")
        return []

def process_forms(pdf_path, page_num):
    """Extract and process form fields."""
    form_fields = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        if hasattr(page, 'form_fields') and page.form_fields:
            for field in page.form_fields:
                form_fields.append({
                    'name': field.get('name', ''),
                    'value': field.get('value', ''),
                    'type': field.get('type', ''),
                    'position': {
                        'x0': field.get('x0', 0),
                        'y0': field.get('y0', 0),
                        'x1': field.get('x1', 0),
                        'y1': field.get('y1', 0)
                    }
                })
    return form_fields

def is_pdf_processed(filename, session):
    """Check if a PDF file has already been processed."""
    return session.query(PDFFile).filter_by(filename=filename).first() is not None

def get_surrounding_text(words, position, window=100):
    """Get text surrounding a specific position."""
    surrounding = []
    x0, y0 = position.get('x0', 0), position.get('y0', 0)
    
    for word in words:
        word_pos = word['position']
        # Check if word is within vertical window
        if abs(word_pos['y0'] - y0) <= window:
            surrounding.append(word['text'])
    
    return ' '.join(surrounding)

def load_nlp():
    """Load spaCy model for text processing."""
    try:
        return spacy.load("en_core_web_sm")
    except:
        console.print("[yellow]Installing spaCy model...[/yellow]")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def semantic_chunk(text: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """Break text into semantic chunks while preserving context."""
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in doc.sents:
        # Check if sentence starts a new section
        is_section_start = bool(re.match(r'^(Section|Chapter|\d+\.|\[|\()', sent.text.strip()))
        
        # If it's a new section or chunk is too large, start new chunk
        if is_section_start or current_length + len(sent.text) > max_chunk_size:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'type': 'section' if is_section_start else 'paragraph',
                    'metadata': {
                        'length': len(chunk_text),
                        'sentences': len(current_chunk)
                    }
                })
            current_chunk = [sent.text]
            current_length = len(sent.text)
        else:
            current_chunk.append(sent.text)
            current_length += len(sent.text)
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'type': 'paragraph',
            'metadata': {
                'length': len(chunk_text),
                'sentences': len(current_chunk)
            }
        })
    
    return chunks

def create_hierarchical_embeddings(text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create embeddings at different hierarchical levels."""
    if not text or not text.strip():
        return {
            'document_embedding': None,
            'chunks': []
        }

    
    try:
        # Document level embedding - limit text size for embedding
        max_text_length = 5000  # Reduced from 10000 to prevent timeouts
        doc_text = text[:max_text_length] if len(text) > max_text_length else text
        
        # Add timeout for document embedding
        try:
            doc_embedding = get_embedding(doc_text, context="full document")
        except Exception as e:
            console.print(f"[red]Error getting document embedding: {e}[/red]")
            doc_embedding = None

        # Create semantic chunks with size limit and timeout
        chunks = []
        try:
            # Add timeout for chunking
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout(seconds):
                def signal_handler(signum, frame):
                    raise TimeoutError("Operation timed out")
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)

            with timeout(30):  # 30 seconds timeout for chunking
                chunks = semantic_chunk(text, max_chunk_size=300)  # Reduced chunk size further
        except TimeoutError:
            console.print("[yellow]Chunking timed out, using simple chunking[/yellow]")
            # Fallback to simple chunking
            words = text.split()
            chunk_size = 300
            chunks = [{
                'text': ' '.join(words[i:i + chunk_size]),
                'type': 'paragraph',
                'metadata': {
                    'length': len(' '.join(words[i:i + chunk_size])),
                    'sentences': 1
                }
            } for i in range(0, len(words), chunk_size)]
        except Exception as chunk_error:
            console.print(f"[yellow]Error in semantic chunking: {chunk_error}. Using simple chunking.[/yellow]")
            # Fallback to simple chunking
            words = text.split()
            chunk_size = 300
            chunks = [{
                'text': ' '.join(words[i:i + chunk_size]),
                'type': 'paragraph',
                'metadata': {
                    'length': len(' '.join(words[i:i + chunk_size])),
                    'sentences': 1
                }
            } for i in range(0, len(words), chunk_size)]

        # Process each chunk with error handling and timeout
        processed_chunks = []
        for chunk in chunks:
            try:
                if not chunk['text'].strip():
                    continue
                    
                # Get chunk embedding with context
                chunk_context = {
                    'type': chunk['type'],
                    'document_context': str(context or {})[:500],  # Further limit context size
                    'metadata': chunk['metadata']
                }
                
                # Add timeout for chunk embedding
                with timeout(10):  # 10 seconds timeout per chunk
                    chunk_embedding = get_embedding(chunk['text'], context=str(chunk_context))
                
                if chunk_embedding:  # Only add if embedding was successful
                    processed_chunks.append({
                        'text': chunk['text'],
                        'type': chunk['type'],
                        'embedding': chunk_embedding,
                        'metadata': chunk['metadata']
                    })
            except TimeoutError:
                console.print(f"[yellow]Chunk embedding timed out for chunk of length {len(chunk['text'])}[/yellow]")
                continue
            except Exception as chunk_error:
                console.print(f"[yellow]Error processing chunk: {chunk_error}[/yellow]")
                continue

        return {
            'document_embedding': doc_embedding,
            'chunks': processed_chunks
        }
    except Exception as e:
        console.print(f"[red]Critical error in hierarchical embedding creation: {e}[/red]")
        # Return simplified structure in case of error
        return {
            'document_embedding': None,
            'chunks': []
        }

def extract_semantic_metadata(text: str) -> Dict[str, Any]:
    """Extract semantic metadata from text."""
    doc = nlp(text)
    
    return {
        'entities': [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            for ent in doc.ents
        ],
        'key_phrases': [
            chunk.text for chunk in doc.noun_chunks
        ],
        'sentiment': doc.sentiment
    }

def process_content_with_context(content: str, content_type: str, position: Dict, words: List[Dict], 
                               context_window: int = 100) -> Dict[str, Any]:
    """Process content with enhanced context awareness."""
    # Get surrounding text
    context_text = get_surrounding_text(words, position, context_window)
    
    # Extract semantic metadata
    semantic_data = extract_semantic_metadata(content)
    
    # Create hierarchical embeddings
    context_dict = {
        'content_type': content_type,
        'surrounding_text': context_text,
        'semantic_data': semantic_data
    }
    
    embeddings = create_hierarchical_embeddings(content, context_dict)
    
    return {
        'content': content,
        'context': context_text,
        'semantic_metadata': semantic_data,
        'embeddings': embeddings
    }

def process_pdf(file_path):
    """Process a PDF file with enhanced semantic understanding."""
    try:
        session = Session()
        filename = os.path.basename(file_path)

        # Delete existing records for this file
        existing_pdf = session.query(PDFFile).filter_by(filename=filename).first()
        if existing_pdf:
            # Delete all embeddings associated with this PDF
            session.query(PDFEmbedding).filter_by(pdf_file_id=existing_pdf.id).delete()
            # Delete the PDF file record
            session.delete(existing_pdf)
            session.commit()
            console.print(f"[yellow]Deleted existing records for {filename}[/yellow]")

        with Progress() as progress:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text with error handling
                try:
                    full_text = []
                    for page in pdf_reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                full_text.append(page_text)
                        except Exception as page_error:
                            console.print(f"[yellow]Warning: Could not extract text from page in {filename}: {page_error}[/yellow]")
                            continue
                    
                    full_text = ' '.join(full_text)
                except Exception as text_error:
                    console.print(f"[yellow]Warning: Error extracting text from {filename}: {text_error}[/yellow]")
                    full_text = ""

                # Create PDF file record without document embedding
                try:
                    pdf_metadata = make_json_serializable(pdf_reader.metadata)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not process metadata for {filename}: {e}[/yellow]")
                    pdf_metadata = {}
                
                pdf_file = PDFFile(
                    filename=filename,
                    file_path=file_path,
                    file_size=os.path.getsize(file_path),
                    total_pages=num_pages,
                    pdf_metadata=pdf_metadata,
                    document_embedding=None  # No longer needed
                )
                session.add(pdf_file)
                session.commit()

                pages_task = progress.add_task(f"[cyan]Processing {filename}[/cyan]", total=num_pages)
                
                for page_num in range(num_pages):
                    # Extract text with position
                    words = extract_text_with_position(file_path, page_num)
                    
                    if words:
                        # Process text content
                        text = ' '.join([word['text'] for word in words])
                        processed_text = process_content_with_context(
                            text, 
                            ContentType.TEXT,
                            {'x0': 0, 'y0': 0},  # Page-level position
                            words
                        )
                        
                        # Store text embeddings with hierarchy
                        for chunk in processed_text['embeddings']['chunks']:
                            pdf_embedding = PDFEmbedding(
                                pdf_file_id=pdf_file.id,
                                page_number=page_num + 1,
                                hierarchy_level=HierarchyLevel.PARAGRAPH,
                                content_type=ContentType.TEXT,
                                page_content=chunk['text'],
                                embedding=chunk['embedding'],
                                semantic_metadata=chunk['metadata'],
                                confidence=0.9
                            )
                            session.add(pdf_embedding)
                    
                    # Process other content types (tables, images, forms)
                    for processor, content_type in [
                        (process_tables, ContentType.TABLE),
                        (process_images, ContentType.IMAGE),
                        (process_forms, ContentType.FORM)
                    ]:
                        contents = processor(file_path, page_num)
                        for content in contents:
                            processed_content = process_content_with_context(
                                content['content'],
                                content_type,
                                content.get('position', {'x0': 0, 'y0': 0}),
                                words
                            )
                            
                            pdf_embedding = PDFEmbedding(
                                pdf_file_id=pdf_file.id,
                                page_number=page_num + 1,
                                hierarchy_level=HierarchyLevel.ELEMENT,
                                content_type=content_type,
                                page_content=content['content'],
                                embedding=processed_content['embeddings']['document_embedding'],
                                position=content.get('position'),
                                content_format=content.get('metadata', {}),
                                context=processed_content['context'],
                                semantic_metadata=processed_content['semantic_metadata'],
                                confidence=content.get('confidence', 0.8)
                            )
                            session.add(pdf_embedding)
                    
                    progress.update(pages_task, advance=1)
                    session.commit()
                
                console.print(f"[green]Successfully processed {filename}[/green]")
                return True
            
    except Exception as e:
        error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
        console.print(f"[red]Error processing PDF {file_path}: {error_msg}[/red]")
        session.rollback()
        return False
    finally:
        session.close()

def process_pdf_folder(folder_path):
    """Process all PDF files in a folder."""
    console.print(Panel.fit("PDF Processing System", style="bold magenta"))
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        console.print("[yellow]No PDF files found in the specified folder[/yellow]")
        return

    # Create summary table
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("File Name")
    table.add_column("Status")
    table.add_column("Processing Time")

    console.print(f"\n[cyan]Found {len(pdf_files)} PDF files to process[/cyan]\n")

    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        start_time = time.time()
        
        success = process_pdf(file_path)
        
        processing_time = time.time() - start_time
        status = "[green]Success[/green]" if success else "[red]Skipped/Failed[/red]"
        table.add_row(
            pdf_file,
            status,
            f"{processing_time:.2f}s"
        )

    console.print("\n[bold]Processing Summary:[/bold]")
    console.print(table)

if __name__ == "__main__":
    # Process PDFs in the pdf directory
    process_pdf_folder("pdf") 