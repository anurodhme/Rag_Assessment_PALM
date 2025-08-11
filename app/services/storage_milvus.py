"""Milvus vector storage service."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding import get_embedding_service

logger = get_logger(__name__)


class MilvusService:
    """Service for managing vector storage in Milvus."""
    
    def __init__(self):
        """Initialize Milvus service."""
        self.collection_name = "document_chunks"
        self.connection_alias = "default"
        self.collection = None
        self.embedding_service = get_embedding_service()
        self._connect()
        self._setup_collection()
    
    def _connect(self) -> None:
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias=self.connection_alias,
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            logger.info(f"Connected to Milvus at {settings.milvus_host}:{settings.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self) -> None:
        """Set up the collection schema and create collection if it doesn't exist."""
        try:
            # Define collection schema
            embedding_dim = self.embedding_service.get_embedding_dimension()
            
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="chunking_strategy", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document chunks with embeddings for RAG"
            )
            
            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self.connection_alias
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                self.collection = Collection(
                    name=self.collection_name,
                    using=self.connection_alias
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            
            # Create index for vector search
            self._create_index()
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _create_index(self) -> None:
        """Create vector index for efficient similarity search."""
        try:
            # Check if index already exists
            if self.collection.has_index():
                logger.info("Index already exists")
                return
            
            # Create IVF_FLAT index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info("Created vector index")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def insert_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        filename: str,
        chunking_strategy: str
    ) -> List[str]:
        """
        Insert document chunks with embeddings into Milvus.
        
        Args:
            document_id: Document identifier
            chunks: List of chunk dictionaries with content and metadata
            filename: Original filename
            chunking_strategy: Strategy used for chunking
            
        Returns:
            List of embedding IDs assigned to chunks
            
        Raises:
            RuntimeError: If insertion fails
        """
        try:
            if not chunks:
                return []
            
            # Extract text content for embedding
            texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_service.encode_batch(texts)
            
            # Prepare data for insertion
            embedding_ids = [str(uuid.uuid4()) for _ in chunks]
            
            data = [
                embedding_ids,  # id
                [document_id] * len(chunks),  # document_id
                [chunk["chunk_id"] for chunk in chunks],  # chunk_id
                texts,  # content
                [filename] * len(chunks),  # filename
                [chunking_strategy] * len(chunks),  # chunking_strategy
                embeddings  # embedding
            ]
            
            # Insert data
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise RuntimeError(f"Failed to insert chunks: {e}")
    
    def search_similar_chunks(
        self,
        query_text: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_text: Query text to search for
            top_k: Number of top results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of similar chunks with metadata and scores
            
        Raises:
            RuntimeError: If search fails
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_single(query_text)
            
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Build filter expression if document_ids provided
            filter_expr = None
            if document_ids:
                doc_id_list = "', '".join(document_ids)
                filter_expr = f"document_id in ['{doc_id_list}']"
            
            # Load collection for search
            self.collection.load()
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["document_id", "chunk_id", "content", "filename", "chunking_strategy"]
            )
            
            # Process results
            similar_chunks = []
            for hits in results:
                for hit in hits:
                    similar_chunks.append({
                        "embedding_id": hit.id,
                        "document_id": hit.entity.get("document_id"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "content": hit.entity.get("content"),
                        "filename": hit.entity.get("filename"),
                        "chunking_strategy": hit.entity.get("chunking_strategy"),
                        "similarity_score": float(hit.score)
                    })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise RuntimeError(f"Failed to search similar chunks: {e}")
    
    def delete_document_chunks(self, document_id: str) -> None:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Raises:
            RuntimeError: If deletion fails
        """
        try:
            # Delete by document_id
            filter_expr = f"document_id == '{document_id}'"
            self.collection.delete(filter_expr)
            self.collection.flush()
            
            logger.info(f"Deleted chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            raise RuntimeError(f"Failed to delete document chunks: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.collection.num_entities
            return {
                "total_chunks": stats,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            connections.disconnect(self.connection_alias)
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {e}")


# Global Milvus service instance
_milvus_service = None


def get_milvus_service() -> MilvusService:
    """
    Get the global Milvus service instance.
    
    Returns:
        MilvusService instance
    """
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
    return _milvus_service
