"""Document ingestion API endpoint."""

from __future__ import annotations

import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database import Document as DBDocument, DocumentChunk as DBDocumentChunk
from app.models.schemas import (
    ChunkingStrategy,
    IngestionResponse,
    ChunkInfo,
    ErrorResponse
)
from app.services.chunking import chunk_document
from app.services.storage_milvus import get_milvus_service
from app.utils.document_processing import DocumentProcessor

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest (PDF or TXT)"),
    chunking_strategy: ChunkingStrategy = Form(
        default=ChunkingStrategy.LATE,
        description="Chunking strategy to use"
    ),
    chunk_size: int = Form(
        default=None,
        description="Override default chunk size"
    ),
    chunk_overlap: int = Form(
        default=None,
        description="Override default chunk overlap"
    ),
    db: Session = Depends(get_db)
):
    """
    Ingest a document by extracting text, chunking, generating embeddings, and storing.
    
    This endpoint handles the complete document ingestion pipeline:
    1. Validates file type and size
    2. Extracts text content
    3. Chunks the text using specified strategy
    4. Generates embeddings for chunks
    5. Stores chunks in Milvus and metadata in PostgreSQL
    
    Args:
        file: Uploaded document file
        chunking_strategy: Strategy for text chunking
        chunk_size: Override default chunk size
        chunk_overlap: Override default chunk overlap
        db: Database session
        
    Returns:
        IngestionResponse with processing results
        
    Raises:
        HTTPException: If ingestion fails
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        allowed_exts = settings.get_allowed_extensions()
        if not DocumentProcessor.is_supported_file(file.filename, allowed_exts):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_exts)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size
        if not DocumentProcessor.validate_file_size(file_content, settings.max_file_size_mb):
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Get file info
        file_info = DocumentProcessor.get_file_info(file_content, file.filename)
        logger.info(f"Processing file: {file_info}")
        
        # Extract text
        try:
            extracted_text = DocumentProcessor.extract_text(file_content, file.filename)
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No text content found in document")
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")
        
        # Use provided chunk parameters or defaults
        final_chunk_size = chunk_size or settings.default_chunk_size
        final_chunk_overlap = chunk_overlap or settings.default_chunk_overlap
        
        # Chunk the document
        try:
            chunks = chunk_document(
                text=extracted_text,
                strategy=chunking_strategy,
                chunk_size=final_chunk_size,
                overlap=final_chunk_overlap
            )
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks generated from document")
                
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to chunk document: {str(e)}")
        
        # Create document record
        document_id = str(uuid.uuid4())
        
        try:
            db_document = DBDocument(
                id=document_id,
                filename=file.filename,
                file_size=len(file_content),
                content_type=file_info["content_type"],
                chunking_strategy=chunking_strategy.value,
                total_chunks=len(chunks),
                processing_time_seconds=0  # Will be updated later
            )
            
            db.add(db_document)
            db.flush()  # Get the ID without committing
            
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create document record")
        
        # Process chunks and store in database
        chunk_infos = []
        chunk_data_for_milvus = []
        
        try:
            for i, (chunk_content, start_pos, end_pos) in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                # Create chunk record
                db_chunk = DBDocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk_content,
                    start_position=start_pos,
                    end_position=end_pos
                )
                
                db.add(db_chunk)
                
                # Prepare data for Milvus
                chunk_data_for_milvus.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content
                })
                
                # Prepare response data
                chunk_infos.append(ChunkInfo(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    start_pos=start_pos,
                    end_pos=end_pos
                ))
            
            db.flush()
            
        except Exception as e:
            logger.error(f"Failed to create chunk records: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create chunk records")
        
        # Store embeddings in Milvus
        try:
            milvus_service = get_milvus_service()
            embedding_ids = milvus_service.insert_chunks(
                document_id=document_id,
                chunks=chunk_data_for_milvus,
                filename=file.filename,
                chunking_strategy=chunking_strategy.value
            )
            
            # Update chunk records with embedding IDs
            for i, embedding_id in enumerate(embedding_ids):
                chunk_infos[i].embedding_id = embedding_id
                # Update database record
                db_chunk = db.query(DBDocumentChunk).filter(
                    DBDocumentChunk.id == chunk_infos[i].chunk_id
                ).first()
                if db_chunk:
                    db_chunk.embedding_id = embedding_id
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {str(e)}")
        
        # Calculate processing time and update document
        processing_time = time.time() - start_time
        db_document.processing_time_seconds = processing_time
        
        # Commit all changes
        try:
            db.commit()
            logger.info(f"Successfully ingested document {document_id} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to save ingestion results")
        
        # Return response
        return IngestionResponse(
            file_id=document_id,
            filename=file.filename,
            file_size=len(file_content),
            chunking_strategy=chunking_strategy,
            total_chunks=len(chunks),
            chunks=chunk_infos,
            processing_time_seconds=processing_time,
            created_at=db_document.created_at
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")


@router.get("/health")
async def ingestion_health():
    """Health check endpoint for ingestion service."""
    try:
        # Test Milvus connection
        milvus_service = get_milvus_service()
        milvus_stats = milvus_service.get_collection_stats()
        
        return {
            "status": "healthy",
            "milvus": milvus_stats,
            "supported_formats": settings.get_allowed_extensions(),
            "max_file_size_mb": settings.max_file_size_mb
        }
    except Exception as e:
        logger.error(f"Ingestion health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
