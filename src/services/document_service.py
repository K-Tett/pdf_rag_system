"""
Document processing service for PDF ingestion and chunking.
"""
import structlog
import os
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import re

from src.services.vector_service import VectorService
from src.core.models import DocumentInfo

logger = structlog.get_logger()


class DocumentChunker:
    """
    Document chunking with semantic and sentence-based strategies.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using sentence-based strategy.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunks with content and metadata
        """
        if not text.strip():
            return []
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(cleaned_text)
        
        # Create chunks
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    current_sentences,
                    len(chunks),
                    metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_sentences)
                current_chunk = overlap_text
                current_sentences = self._get_overlap_sentences(current_sentences)
            
            current_chunk += " " + sentence if current_chunk else sentence
            current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                current_sentences,
                len(chunks),
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)\[\]\{\}\"\']+', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix broken words
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace again
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentence_endings = r'[.!?]+(?=\s+[A-Z]|\s*$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunk(
        self,
        content: str,
        sentences: List[str],
        chunk_index: int,
        base_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a chunk with metadata."""
        chunk_metadata = {
            "chunk_index": chunk_index,
            "chunk_size": len(content),
            "sentence_count": len(sentences),
            "created_at": datetime.utcnow().isoformat()
        }
        
        if base_metadata:
            chunk_metadata.update(base_metadata)
        
        return {
            "content": content,
            "metadata": chunk_metadata
        }
    
    def _get_overlap_text(self, sentences: List[str]) -> str:
        """Get overlap text from the end of current sentences."""
        if not sentences:
            return ""
        
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text if overlap_text else sentence
            else:
                break
        
        return overlap_text.strip()
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences from the end of current sentences."""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences


class PDFProcessor:
    """
    PDF processing using pdfminer.
    """
    
    def __init__(self):
        self.laparams = LAParams(
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            boxes_flow=0.5
        )
    
    async def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with text content and metadata
        """
        logger.info("Extracting text from PDF", file_path=file_path)
        
        try:
            # Extract text using pdfminer
            text = extract_text(file_path, laparams=self.laparams)
            
            # Get file metadata
            file_stats = os.stat(file_path)
            filename = os.path.basename(file_path)
            
            # Extract basic document info
            doc_info = self._extract_document_info(text, filename)
            
            metadata = {
                "filename": filename,
                "file_size": file_stats.st_size,
                "processed_at": datetime.utcnow().isoformat(),
                "text_length": len(text),
                "estimated_pages": max(1, len(text) // 3000),  # Rough estimate
                **doc_info
            }
            
            logger.info(
                "PDF text extraction completed",
                filename=filename,
                text_length=len(text)
            )
            
            return {
                "text": text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error("Failed to extract text from PDF", file_path=file_path, error=str(e))
            raise
    
    def _extract_document_info(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract basic document information from text."""
        info = {}
        
        # Try to extract title (first non-empty line that looks like a title)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.startswith(('Page', 'www', 'http')):
                info["title"] = line
                break
        
        # Try to extract author (look for common patterns)
        author_patterns = [
            r'(?:Author|Authors?)[:]\s*([A-Za-z\s,\.]+)',
            r'(?:By|BY)[:]\s*([A-Za-z\s,\.]+)',
            r'^([A-Za-z\s,\.]+)(?:\n|\s+)(?:University|Institute|Department)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text[:2000], re.MULTILINE | re.IGNORECASE)
            if match:
                info["author"] = match.group(1).strip()
                break
        
        return info


class DocumentService:
    """
    Main document service for PDF ingestion and management.
    """
    
    def __init__(self, vector_service: VectorService, data_dir: str = "./data"):
        self.vector_service = vector_service
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.pdf_processor = PDFProcessor()
        self.chunker = DocumentChunker()
        
        # Document metadata storage
        self.metadata_file = self.data_dir / "documents_metadata.json"
        self.documents_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error("Failed to load metadata", error=str(e))
        
        return {}
    
    def _save_metadata(self):
        """Save document metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.documents_metadata, f, indent=2)
        except Exception as e:
            logger.error("Failed to save metadata", error=str(e))
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID based on file content."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return f"doc_{file_hash[:12]}"
        except Exception:
            # Fallback to filename-based ID
            filename = os.path.basename(file_path)
            return f"doc_{hashlib.md5(filename.encode()).hexdigest()[:12]}"
    
    async def process_pdf(self, file_path: str, process_immediately: bool = True) -> str:
        """
        Process a PDF file and add to vector database.
        
        Args:
            file_path: Path to PDF file
            process_immediately: Whether to process immediately or queue
            
        Returns:
            Document ID
        """
        logger.info("Processing PDF", file_path=file_path)
        
        try:
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            # Check if already processed
            if document_id in self.documents_metadata:
                logger.info("Document already processed", document_id=document_id)
                return document_id
            
            # Extract text and metadata
            extraction_result = await self.pdf_processor.extract_text_from_pdf(file_path)
            text = extraction_result["text"]
            metadata = extraction_result["metadata"]
            
            # Create document metadata
            doc_metadata = {
                "id": document_id,
                "filename": os.path.basename(file_path),
                "title": metadata.get("title", os.path.basename(file_path)),
                "author": metadata.get("author"),
                "upload_date": datetime.utcnow().isoformat(),
                "processing_status": "processing" if process_immediately else "pending",
                "file_path": file_path,
                "metadata": metadata
            }
            
            self.documents_metadata[document_id] = doc_metadata
            self._save_metadata()
            
            if process_immediately:
                # Chunk the text
                chunks = self.chunker.chunk_text(text, {
                    "document_id": document_id,
                    "filename": metadata["filename"],
                    "title": metadata.get("title"),
                    "author": metadata.get("author")
                })
                
                # Add to vector database
                success = await self.vector_service.add_documents(document_id, chunks)
                
                if success:
                    doc_metadata["processing_status"] = "completed"
                    doc_metadata["num_chunks"] = len(chunks)
                    doc_metadata["processed_at"] = datetime.utcnow().isoformat()
                else:
                    doc_metadata["processing_status"] = "failed"
                    doc_metadata["error"] = "Failed to add to vector database"
                
                self.documents_metadata[document_id] = doc_metadata
                self._save_metadata()
            
            logger.info(
                "PDF processing completed",
                document_id=document_id,
                status=doc_metadata["processing_status"]
            )
            
            return document_id
            
        except Exception as e:
            logger.error("Failed to process PDF", file_path=file_path, error=str(e))
            
            # Update metadata with error
            if document_id in self.documents_metadata:
                self.documents_metadata[document_id]["processing_status"] = "failed"
                self.documents_metadata[document_id]["error"] = str(e)
                self._save_metadata()
            
            raise
    
    async def load_existing_documents(self):
        """Load and process any existing documents in the data directory."""
        logger.info("Loading existing documents", data_dir=str(self.data_dir))
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                try:
                    await self.process_pdf(str(pdf_file), process_immediately=True)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}", error=str(e))
        else:
            logger.info("No existing PDF files found")
    
    def get_document_info(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document information by ID."""
        if document_id not in self.documents_metadata:
            return None
        
        metadata = self.documents_metadata[document_id]
        
        return DocumentInfo(
            id=metadata["id"],
            filename=metadata["filename"],
            title=metadata.get("title"),
            author=metadata.get("author"),
            upload_date=datetime.fromisoformat(metadata["upload_date"]),
            processing_status=metadata["processing_status"],
            num_chunks=metadata.get("num_chunks", 0),
            metadata=metadata.get("metadata", {})
        )
    
    def list_documents(self) -> List[DocumentInfo]:
        """List all documents."""
        documents = []
        
        for doc_metadata in self.documents_metadata.values():
            try:
                doc_info = DocumentInfo(
                    id=doc_metadata["id"],
                    filename=doc_metadata["filename"],
                    title=doc_metadata.get("title"),
                    author=doc_metadata.get("author"),
                    upload_date=datetime.fromisoformat(doc_metadata["upload_date"]),
                    processing_status=doc_metadata["processing_status"],
                    num_chunks=doc_metadata.get("num_chunks", 0),
                    metadata=doc_metadata.get("metadata", {})
                )
                documents.append(doc_info)
            except Exception as e:
                logger.error("Failed to parse document metadata", error=str(e))
        
        return documents
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            # Delete from vector database
            success = await self.vector_service.delete_document(document_id)
            
            if success:
                # Remove from metadata
                if document_id in self.documents_metadata:
                    del self.documents_metadata[document_id]
                    self._save_metadata()
                
                logger.info("Document deleted successfully", document_id=document_id)
                return True
            else:
                logger.error("Failed to delete document from vector database", document_id=document_id)
                return False
                
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            return False