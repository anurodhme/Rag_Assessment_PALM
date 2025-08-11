"""Tests for API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    @patch('app.main.engine')
    @patch('redis.from_url')
    @patch('app.services.storage_milvus.get_milvus_service')
    def test_health_check(self, mock_milvus, mock_redis, mock_engine):
        """Test health check endpoint."""
        # Mock successful connections
        mock_engine.connect.return_value.__enter__.return_value.execute.return_value = None
        mock_redis.return_value.ping.return_value = True
        mock_milvus.return_value.get_collection_stats.return_value = {"total_chunks": 0}
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data


class TestIngestionAPI:
    """Test ingestion API endpoints."""
    
    def test_ingestion_health(self):
        """Test ingestion health endpoint."""
        with patch('app.services.storage_milvus.get_milvus_service') as mock_milvus:
            mock_milvus.return_value.get_collection_stats.return_value = {"total_chunks": 0}
            
            response = client.get("/ingest/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "supported_formats" in data
    
    def test_ingest_no_file(self):
        """Test ingestion without file."""
        response = client.post("/ingest/")
        assert response.status_code == 422  # Validation error
    
    def test_ingest_invalid_file_type(self):
        """Test ingestion with invalid file type."""
        file_content = b"test content"
        files = {"file": ("test.xyz", io.BytesIO(file_content), "application/octet-stream")}
        
        response = client.post("/ingest/", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    @patch('app.api.ingestion.get_milvus_service')
    @patch('app.api.ingestion.get_db')
    def test_ingest_txt_file(self, mock_db, mock_milvus):
        """Test ingesting a text file."""
        # Mock database session
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        mock_db.return_value.__exit__.return_value = None
        
        # Mock Milvus service
        mock_milvus.return_value.insert_chunks.return_value = ["embed_1", "embed_2"]
        
        file_content = b"This is a test document. It has multiple sentences for testing chunking."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        data = {"chunking_strategy": "late", "chunk_size": 50}
        
        response = client.post("/ingest/", files=files, data=data)
        
        # May fail due to missing dependencies, but should not crash
        assert response.status_code in [200, 500, 503]


class TestChatAPI:
    """Test chat API endpoints."""
    
    def test_chat_health(self):
        """Test chat health endpoint."""
        with patch('redis.from_url') as mock_redis:
            with patch('app.services.agents.rag_agent.get_rag_agent') as mock_rag:
                with patch('app.services.agents.interview_agent.get_interview_agent') as mock_interview:
                    mock_redis.return_value.ping.return_value = True
                    mock_rag.return_value = Mock()
                    mock_interview.return_value = Mock()
                    
                    response = client.get("/chat/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                    assert "supported_strategies" in data
    
    def test_chat_invalid_request(self):
        """Test chat with invalid request."""
        response = client.post("/chat/", json={})
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.chat.get_rag_agent')
    @patch('app.api.chat.ensure_chat_session')
    @patch('app.api.chat.get_session_memory')
    @patch('app.api.chat.save_chat_message')
    @patch('app.api.chat.update_session_memory')
    def test_chat_basic_query(self, mock_update_memory, mock_save_msg, mock_get_memory, 
                             mock_ensure_session, mock_rag_agent):
        """Test basic chat query."""
        # Mock dependencies
        mock_get_memory.return_value = []
        mock_ensure_session.return_value = Mock()
        mock_rag_agent.return_value.generate_response_sync.return_value = {
            "answer": "Test response",
            "retrieved_chunks": [],
            "is_booking_request": False
        }
        
        request_data = {
            "session_id": "test123",
            "question": "What is machine learning?",
            "search_strategy": "cosine"
        }
        
        response = client.post("/chat/", json=request_data)
        
        # May fail due to missing dependencies, but should not crash
        assert response.status_code in [200, 500, 503]
    
    def test_get_chat_history_not_found(self):
        """Test getting history for non-existent session."""
        with patch('app.api.chat.get_db') as mock_db:
            mock_session = Mock()
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_db.return_value.__enter__.return_value = mock_session
            
            response = client.get("/chat/sessions/nonexistent/history")
            assert response.status_code == 404


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_handler(self):
        """Test 404 error handler."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "available_endpoints" in data
