"""Document processing utilities for text extraction."""

from __future__ import annotations

from typing import Tuple, BinaryIO
import io
import os

import pdfplumber
import pypdf

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Service for processing and extracting text from documents."""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes, filename: str) -> str:
        """
        Extract text from PDF file using pdfplumber with pypdf fallback.
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename for logging
            
        Returns:
            Extracted text content
            
        Raises:
            RuntimeError: If text extraction fails
        """
        try:
            # Try pdfplumber first (better for complex layouts)
            try:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    extracted_text = "\n\n".join(text_parts)
                    if extracted_text.strip():
                        logger.info(f"Successfully extracted text from {filename} using pdfplumber")
                        return extracted_text
            
            except Exception as e:
                logger.warning(f"pdfplumber failed for {filename}: {e}, trying pypdf")
            
            # Fallback to pypdf
            try:
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
                text_parts = []
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                extracted_text = "\n\n".join(text_parts)
                if extracted_text.strip():
                    logger.info(f"Successfully extracted text from {filename} using pypdf")
                    return extracted_text
                else:
                    raise RuntimeError("No text content found in PDF")
            
            except Exception as e:
                logger.error(f"pypdf also failed for {filename}: {e}")
                raise RuntimeError(f"Failed to extract text from PDF: {e}")
        
        except Exception as e:
            logger.error(f"PDF text extraction failed for {filename}: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}")
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes, filename: str) -> str:
        """
        Extract text from text file with encoding detection.
        
        Args:
            file_content: Text file content as bytes
            filename: Original filename for logging
            
        Returns:
            Extracted text content
            
        Raises:
            RuntimeError: If text extraction fails
        """
        try:
            # Try common encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    logger.info(f"Successfully extracted text from {filename} using {encoding} encoding")
                    return text
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            text = file_content.decode('utf-8', errors='replace')
            logger.warning(f"Used utf-8 with error replacement for {filename}")
            return text
        
        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {e}")
            raise RuntimeError(f"Failed to extract text from file: {e}")
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Get file extension from filename.
        
        Args:
            filename: Original filename
            
        Returns:
            File extension in lowercase (without dot)
        """
        return os.path.splitext(filename)[1].lower().lstrip('.')
    
    @staticmethod
    def is_supported_file(filename: str, allowed_extensions: list[str]) -> bool:
        """
        Check if file type is supported.
        
        Args:
            filename: Original filename
            allowed_extensions: List of allowed extensions
            
        Returns:
            True if file is supported
        """
        extension = DocumentProcessor.get_file_extension(filename)
        return extension in [ext.lower() for ext in allowed_extensions]
    
    @staticmethod
    def extract_text(file_content: bytes, filename: str) -> str:
        """
        Extract text from file based on its extension.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
            RuntimeError: If text extraction fails
        """
        extension = DocumentProcessor.get_file_extension(filename)
        
        if extension == 'pdf':
            return DocumentProcessor.extract_text_from_pdf(file_content, filename)
        elif extension == 'txt':
            return DocumentProcessor.extract_text_from_txt(file_content, filename)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    @staticmethod
    def validate_file_size(file_content: bytes, max_size_mb: int) -> bool:
        """
        Validate file size.
        
        Args:
            file_content: File content as bytes
            max_size_mb: Maximum allowed size in MB
            
        Returns:
            True if file size is within limits
        """
        file_size_mb = len(file_content) / (1024 * 1024)
        return file_size_mb <= max_size_mb
    
    @staticmethod
    def get_file_info(file_content: bytes, filename: str) -> dict:
        """
        Get file information including size and type.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with file information
        """
        return {
            "filename": filename,
            "size_bytes": len(file_content),
            "size_mb": round(len(file_content) / (1024 * 1024), 2),
            "extension": DocumentProcessor.get_file_extension(filename),
            "content_type": DocumentProcessor._get_content_type(filename)
        }
    
    @staticmethod
    def _get_content_type(filename: str) -> str:
        """
        Get MIME content type for file.
        
        Args:
            filename: Original filename
            
        Returns:
            MIME content type
        """
        extension = DocumentProcessor.get_file_extension(filename)
        
        content_types = {
            'pdf': 'application/pdf',
            'txt': 'text/plain'
        }
        
        return content_types.get(extension, 'application/octet-stream')
