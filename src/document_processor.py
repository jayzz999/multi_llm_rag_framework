"""
Document Processor
Handles loading and chunking documents from various sources
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# Import for PDF processing
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Import for tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from src.vector_stores.base import Document


class DocumentProcessor:
    """
    Process documents from various sources and chunk them for RAG.
    
    Supports:
    - PDF files
    - Text files (.txt, .md)
    - Multiple chunking strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each chunk (in characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            separators: List of separators for splitting (default: paragraphs, sentences)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        # Try to initialize tokenizer for token-based chunking
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
            
        Example:
            docs = processor.load_pdf("document.pdf")
        """
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is required for PDF processing. "
                            "Install with: pip install pypdf")
        
        documents = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if text.strip():  # Only process non-empty pages
                        metadata = {
                            "source": file_path,
                            "page": page_num + 1,
                            "total_pages": len(pdf_reader.pages),
                            "file_type": "pdf"
                        }
                        
                        # Chunk the page text
                        chunks = self._chunk_text(text)
                        
                        for i, chunk in enumerate(chunks):
                            doc_metadata = metadata.copy()
                            doc_metadata["chunk"] = i + 1
                            doc_metadata["total_chunks"] = len(chunks)
                            
                            documents.append(Document(
                                content=chunk,
                                metadata=doc_metadata
                            ))
        
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")
        
        return documents
    
    def load_text(self, file_path: str) -> List[Document]:
        """
        Load and process a text file (.txt, .md).
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
            
        Example:
            docs = processor.load_text("document.txt")
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Get file extension
            file_ext = Path(file_path).suffix
            
            metadata = {
                "source": file_path,
                "file_type": file_ext[1:] if file_ext else "txt"
            }
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = i + 1
                doc_metadata["total_chunks"] = len(chunks)
                
                documents.append(Document(
                    content=chunk,
                    metadata=doc_metadata
                ))
            
            return documents
        
        except Exception as e:
            raise Exception(f"Error loading text file {file_path}: {str(e)}")
    
    def load_directory(
        self, 
        directory_path: str,
        file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory_path: Path to the directory
            file_types: List of file extensions to process (e.g., ['.pdf', '.txt'])
                       If None, processes all supported types
            
        Returns:
            List of Document objects from all files
            
        Example:
            docs = processor.load_directory("./data/documents", ['.pdf', '.txt'])
        """
        if file_types is None:
            file_types = ['.pdf', '.txt', '.md']
        
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all matching files
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in file_types:
                try:
                    if file_path.suffix == '.pdf':
                        docs = self.load_pdf(str(file_path))
                    else:
                        docs = self.load_text(str(file_path))
                    
                    documents.extend(docs)
                    print(f"✓ Processed: {file_path.name} ({len(docs)} chunks)")
                
                except Exception as e:
                    print(f"✗ Error processing {file_path.name}: {str(e)}")
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        chunks = []
        
        # If text is smaller than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Recursive splitting with separators
        chunks = self._split_text_recursive(text, self.separators)
        
        return chunks
    
    def _split_text_recursive(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """
        Recursively split text using different separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        final_chunks = []
        
        # Get the separator to use
        separator = separators[0] if separators else ""
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Combine small splits and process large ones
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split + separator
            else:
                # Save current chunk if not empty
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                
                # If split is too large, use next separator
                if len(split) > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._split_text_recursive(
                        split, 
                        separators[1:]
                    )
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split + separator
        
        # Add remaining chunk
        if current_chunk:
            final_chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_overlap(final_chunks)
        
        return final_chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with overlap
        """
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap = prev_chunk[-self.chunk_overlap:]
                overlapped_chunks.append(overlap + " " + chunk)
        
        return overlapped_chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def get_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc.content) for doc in documents)
        avg_chars = total_chars / len(documents) if documents else 0
        
        # Count unique sources
        sources = set(doc.metadata.get("source", "") for doc in documents)
        
        stats = {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": int(avg_chars),
            "unique_sources": len(sources),
            "sources": list(sources)
        }
        
        return stats


# Convenience function
def process_documents(
    file_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Convenience function to process multiple documents.
    
    Args:
        file_paths: List of file paths to process
        chunk_size: Chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
        
    Example:
        docs = process_documents(["doc1.pdf", "doc2.txt"])
    """
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_documents = []
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            print(f"⚠ File not found: {file_path}")
            continue
        
        try:
            if path.suffix == '.pdf':
                docs = processor.load_pdf(str(path))
            else:
                docs = processor.load_text(str(path))
            
            all_documents.extend(docs)
            print(f"✓ Processed: {path.name} ({len(docs)} chunks)")
        
        except Exception as e:
            print(f"✗ Error processing {path.name}: {str(e)}")
    
    return all_documents
