from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, JSON, Float, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()

class ContentType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    FORM = "form"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    SECTION = "section"
    DOCUMENT = "document"

class HierarchyLevel(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    ELEMENT = "element"

class PDFFile(Base):
    __tablename__ = 'pdf_files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)  # in bytes
    total_pages = Column(Integer)
    pdf_metadata = Column(JSON)  # Store PDF metadata
    document_embedding = Column(JSON)  # Store document-level embedding
    
    embeddings = relationship("PDFEmbedding", back_populates="pdf_file")

class PDFEmbedding(Base):
    __tablename__ = 'pdf_embeddings'
    
    id = Column(Integer, primary_key=True)
    pdf_file_id = Column(Integer, ForeignKey('pdf_files.id'))
    page_number = Column(Integer)
    hierarchy_level = Column(SQLEnum(HierarchyLevel))  # Document, Section, Paragraph, Element
    content_type = Column(SQLEnum(ContentType))  # Type of content
    content_format = Column(JSON)  # Format-specific metadata
    page_content = Column(Text)  # The actual content
    embedding = Column(JSON)  # Store embedding as JSON array
    position = Column(JSON)  # Store position information
    confidence = Column(Float)  # Confidence score
    parent_id = Column(Integer, ForeignKey('pdf_embeddings.id'), nullable=True)  # For hierarchical relationships
    context = Column(JSON)  # Store contextual information
    semantic_metadata = Column(JSON)  # Store semantic analysis results
    
    pdf_file = relationship("PDFFile", back_populates="embeddings")
    children = relationship("PDFEmbedding", backref="parent", remote_side=[id])