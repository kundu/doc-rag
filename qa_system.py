import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, PDFFile, PDFEmbedding
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

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
EMBEDDING_API_URL = os.getenv('AI_API_URL')
EMBEDDING_MODEL = os.getenv('AI_MODEL')
QA_API_URL = os.getenv('QA_API_URL')
QA_MODEL = os.getenv('QA_MODEL')

# Create database connection
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def get_embedding(text, context=None):
    """Get embedding from the AI API with optional context."""
    try:
        input_text = text
        if context:
            input_text = f"{context}: {text}"
            
        response = requests.post(
            EMBEDDING_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": EMBEDDING_MODEL,
                "input": input_text
            }
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        console.print(f"[red]Error getting embedding: {e}[/red]")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_content(query, limit=5):
    """Search for similar content in the database."""
    try:
        session = Session()
        
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Get all embeddings from database
        results = session.query(PDFEmbedding, PDFFile).join(PDFFile).all()
        
        # Calculate similarities
        similarities = []
        for result in results:
            embedding = result.PDFEmbedding.embedding
            if embedding:
                similarity = cosine_similarity(
                    np.array(query_embedding),
                    np.array(embedding)
                )
                
                similarities.append({
                    'similarity': similarity,
                    'content': result.PDFEmbedding.page_content,
                    'file_name': result.PDFFile.filename,
                    'page_number': result.PDFEmbedding.page_number,
                    'metadata': {
                        'file_name': result.PDFFile.filename,
                        'page': result.PDFEmbedding.page_number,
                        'type': result.PDFEmbedding.content_type.value if result.PDFEmbedding.content_type else 'unknown',
                        'level': result.PDFEmbedding.hierarchy_level.value if result.PDFEmbedding.hierarchy_level else 'unknown'
                    }
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
                
    except Exception as e:
        console.print(f"[red]Error searching: {e}[/red]")
        return []
    finally:
        session.close()

def get_llm_response(query, context_docs):
    """Get response from LLM using the context."""
    try:
        # Prepare context
        context = "\n\n".join([
            f"Document: {doc['file_name']} (Page {doc['page_number']})\n{doc['content']}"
            for doc in context_docs
        ])
        
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "Always cite your sources using the document name and page number. "
                    "If you're not sure about something, say so. "
                    "Format your response in markdown."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based on the context provided. Include relevant quotes and citations."
            }
        ]
        
        # Call the LLM API
        response = requests.post(
            QA_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": QA_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }
        )
        response.raise_for_status()
        
        # Extract and return the response
        return response.json()['choices'][0]['message']['content']
        
    except Exception as e:
        console.print(f"[red]Error getting LLM response: {e}[/red]")
        return f"Error: Could not generate response. {str(e)}"

def main():
    """Main QA loop."""
    console.print(Panel.fit("PDF Question Answering System", style="bold magenta"))
    
    while True:
        try:
            # Get question from user
            console.print("\n[cyan]Enter your question (or 'quit' to exit):[/cyan]")
            query = input().strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            console.print("\n[cyan]Searching for relevant content...[/cyan]")
            
            # Search for relevant content
            results = search_similar_content(query)
            
            if not results:
                console.print("[yellow]No relevant content found.[/yellow]")
                continue
            
            # Get LLM response
            console.print("\n[cyan]Generating answer...[/cyan]")
            response = get_llm_response(query, results)
            
            # Display results
            console.print("\n[bold]Answer:[/bold]")
            console.print(Markdown(response))
            
            # Display sources
            console.print("\n[bold]Sources:[/bold]")
            for result in results:
                console.print(f"- {result['file_name']} (Page {result['page_number']}) [Similarity: {result['similarity']:.2f}]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue
    
    console.print("\n[cyan]Thank you for using the PDF QA System![/cyan]")

if __name__ == "__main__":
    main() 