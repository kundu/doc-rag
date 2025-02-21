import os
import json
from datetime import datetime
from dotenv import load_dotenv
import PyPDF2
import requests
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, PDFFile, PDFEmbedding
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

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

def get_embedding(text):
    """Get embedding from the AI API."""
    try:
        response = requests.post(
            AI_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": AI_MODEL,
                "input": text
            }
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        console.print(f"[red]Error getting embedding: {e}[/red]")
        return None

def is_pdf_processed(filename, session):
    """Check if a PDF file has already been processed."""
    return session.query(PDFFile).filter_by(filename=filename).first() is not None

def process_pdf(file_path):
    """Process a PDF file and store its embeddings."""
    try:
        session = Session()
        filename = os.path.basename(file_path)

        # Check if file already processed
        if is_pdf_processed(filename, session):
            console.print(f"[yellow]Skipping {filename} - Already processed[/yellow]")
            return False

        # Create progress bars
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Open and read PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                file_size = os.path.getsize(file_path)
                
                # Create PDF file record
                pdf_file = PDFFile(
                    filename=filename,
                    file_path=file_path,
                    file_size=file_size,
                    total_pages=num_pages
                )
                session.add(pdf_file)
                session.commit()

                # Process pages task
                pages_task = progress.add_task(
                    f"[cyan]Processing {filename}[/cyan]",
                    total=num_pages
                )
                
                # Process each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():  # Only process non-empty pages
                        embedding = get_embedding(text)
                        
                        if embedding:
                            pdf_embedding = PDFEmbedding(
                                pdf_file_id=pdf_file.id,
                                page_number=page_num + 1,
                                page_content=text,
                                embedding=embedding
                            )
                            session.add(pdf_embedding)
                    
                    progress.update(pages_task, advance=1)
                    
                session.commit()
                console.print(f"[green]Successfully processed {filename}[/green]")
                return True
            
    except Exception as e:
        console.print(f"[red]Error processing PDF {file_path}: {e}[/red]")
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

def search_similar_content(query, limit=5):
    """Search for similar content using embeddings."""
    try:
        console.print(f"\n[cyan]Searching for: '{query}'[/cyan]")
        session = Session()
        query_embedding = get_embedding(query)
        search dector 
        if query_embedding:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Calculating similarities...[/cyan]")
                
                # Get all embeddings
                results = session.query(
                    PDFEmbedding,
                    PDFFile
                ).join(PDFFile).all()
                
                # Calculate similarities
                similarities = [
                    (
                        cosine_similarity(
                            np.array(query_embedding),
                            np.array(result.PDFEmbedding.embedding)
                        ),
                        result
                    )
                    for result in results
                ]
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                progress.update(task, completed=True)
                
                # Create results table
                table = Table(show_header=True, header_style="bold blue")
                table.add_column("File Name")
                table.add_column("Page")
                table.add_column("Similarity")
                table.add_column("Preview")
                
                # Return top results
                for similarity, result in similarities[:limit]:
                    table.add_row(
                        result.PDFFile.filename,
                        str(result.PDFEmbedding.page_number),
                        f"{similarity:.2f}",
                        result.PDFEmbedding.page_content[:200] + "..."
                    )
                
                console.print("\n[bold]Search Results:[/bold]")
                console.print(table)
                return similarities[:limit]
                
    except Exception as e:
        console.print(f"[red]Error searching: {e}[/red]")
        return []
    finally:
        session.close()

if __name__ == "__main__":
    # Process all PDFs in the folder
    pdf_folder = "pdf"
    process_pdf_folder(pdf_folder)
    
    # Example search
    search_similar_content("railway safety regulations") 