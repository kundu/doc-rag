from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class PDFFile(Base):
    __tablename__ = 'pdf_files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)  # in bytes
    total_pages = Column(Integer)
    
    embeddings = relationship("PDFEmbedding", back_populates="pdf_file")

class PDFEmbedding(Base):
    __tablename__ = 'pdf_embeddings'
    
    id = Column(Integer, primary_key=True)
    pdf_file_id = Column(Integer, ForeignKey('pdf_files.id'))
    page_number = Column(Integer)
    page_content = Column(Text)
    embedding = Column(JSON)  # Store embedding as JSON array
    
    pdf_file = relationship("PDFFile", back_populates="embeddings")