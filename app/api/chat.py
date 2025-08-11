"""Conversational RAG API endpoint with session memory and booking support."""

from __future__ import annotations

import time
import json
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import redis

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database import ChatSession as DBChatSession, ChatMessage as DBChatMessage
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    RetrievedChunk,
    SearchStrategy,
    ErrorResponse
)
from app.services.agents.rag_agent import get_rag_agent
from app.services.agents.interview_agent import get_interview_agent

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# Redis connection for session memory
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def get_session_memory(session_id: str) -> List[Dict[str, str]]:
    """
    Retrieve conversation history from Redis.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of conversation messages
    """
    try:
        memory_key = f"session:{session_id}:memory"
        memory_data = redis_client.get(memory_key)
        
        if memory_data:
            return json.loads(memory_data)
        return []
        
    except Exception as e:
        logger.error(f"Failed to retrieve session memory: {e}")
        return []


def update_session_memory(
    session_id: str,
    user_message: str,
    assistant_message: str,
    max_messages: int = 10
) -> None:
    """
    Update conversation history in Redis.
    
    Args:
        session_id: Session identifier
        user_message: User's message
        assistant_message: Assistant's response
        max_messages: Maximum number of messages to keep
    """
    try:
        memory_key = f"session:{session_id}:memory"
        
        # Get existing memory
        memory = get_session_memory(session_id)
        
        # Add new messages
        memory.extend([
            {"type": "user", "content": user_message},
            {"type": "assistant", "content": assistant_message}
        ])
        
        # Keep only the last N messages
        if len(memory) > max_messages:
            memory = memory[-max_messages:]
        
        # Store updated memory with expiration (24 hours)
        redis_client.setex(
            memory_key,
            86400,  # 24 hours in seconds
            json.dumps(memory)
        )
        
    except Exception as e:
        logger.error(f"Failed to update session memory: {e}")


def ensure_chat_session(session_id: str, db: Session) -> DBChatSession:
    """
    Ensure chat session exists in database.
    
    Args:
        session_id: Session identifier
        db: Database session
        
    Returns:
        Chat session record
    """
    # Check if session exists
    db_session = db.query(DBChatSession).filter(
        DBChatSession.session_id == session_id
    ).first()
    
    if not db_session:
        # Create new session
        db_session = DBChatSession(session_id=session_id)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        logger.info(f"Created new chat session: {session_id}")
    else:
        # Update last activity
        from datetime import datetime
        db_session.last_activity = datetime.utcnow()
        db.commit()
    
    return db_session


def save_chat_message(
    session_id: str,
    message_type: str,
    content: str,
    search_strategy: Optional[SearchStrategy] = None,
    response_time: Optional[float] = None,
    db: Session = None
) -> None:
    """
    Save chat message to database.
    
    Args:
        session_id: Session identifier
        message_type: 'user' or 'assistant'
        content: Message content
        search_strategy: Search strategy used (for assistant messages)
        response_time: Response time in seconds
        db: Database session
    """
    try:
        # Get session
        db_session = db.query(DBChatSession).filter(
            DBChatSession.session_id == session_id
        ).first()
        
        if db_session:
            message = DBChatMessage(
                session_id=db_session.id,
                message_type=message_type,
                content=content,
                search_strategy=search_strategy.value if search_strategy else None,
                response_time_seconds=response_time
            )
            db.add(message)
            db.commit()
            
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Handle conversational RAG queries with session memory and booking support.
    
    This endpoint provides:
    1. Conversational RAG with session memory
    2. Multiple search strategies (cosine, hybrid)
    3. Interview booking detection and handling
    4. Persistent conversation history
    
    Args:
        request: Chat request with question and parameters
        db: Database session
        
    Returns:
        ChatResponse with answer and metadata
        
    Raises:
        HTTPException: If chat processing fails
    """
    start_time = time.time()
    
    try:
        # Ensure chat session exists
        ensure_chat_session(request.session_id, db)
        
        # Get conversation history from Redis
        conversation_history = get_session_memory(request.session_id)
        
        # Save user message
        save_chat_message(
            session_id=request.session_id,
            message_type="user",
            content=request.question,
            db=db
        )
        
        # Get RAG agent
        rag_agent = get_rag_agent()
        
        # Generate response using RAG agent
        rag_response = rag_agent.generate_response_sync(
            question=request.question,
            conversation_history=conversation_history,
            search_strategy=request.search_strategy,
            top_k=request.top_k,
            document_ids=None  # Could be extended to filter by specific documents
        )
        
        # Check if this is a booking request
        if rag_response.get("is_booking_request", False):
            # Handle booking request with Interview Agent
            interview_agent = get_interview_agent()
            
            # Check if we have booking information in the request
            booking_info = {}
            if request.name:
                booking_info["name"] = request.name
            if request.email:
                booking_info["email"] = request.email
            if request.date:
                booking_info["date"] = request.date
            if request.time:
                booking_info["time"] = request.time
            
            # Handle booking request
            booking_result = await interview_agent.handle_booking_request(
                user_message=request.question,
                session_id=request.session_id,
                existing_info=booking_info if booking_info else None
            )
            
            # Prepare response based on booking result
            if booking_result["status"] == "booking_created":
                answer = booking_result["message"]
                is_booking_request = True
                booking_id = booking_result["booking_id"]
            else:
                answer = booking_result.get("message", "I can help you book an interview. Please provide your name, email, preferred date, and time.")
                is_booking_request = True
                booking_id = None
            
            retrieved_chunks = []
            
        else:
            # Regular RAG response
            answer = rag_response["answer"]
            retrieved_chunks = rag_response.get("retrieved_chunks", [])
            is_booking_request = False
            booking_id = None
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Save assistant message
        save_chat_message(
            session_id=request.session_id,
            message_type="assistant",
            content=answer,
            search_strategy=request.search_strategy,
            response_time=response_time,
            db=db
        )
        
        # Update session memory in Redis
        update_session_memory(
            session_id=request.session_id,
            user_message=request.question,
            assistant_message=answer
        )
        
        # Prepare response
        response = ChatResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            response_time_seconds=response_time,
            is_booking_request=is_booking_request,
            booking_id=booking_id
        )
        
        logger.info(f"Chat response generated for session {request.session_id}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        
        # Save error message
        try:
            save_chat_message(
                session_id=request.session_id,
                message_type="assistant",
                content="I apologize, but I encountered an error. Please try again.",
                db=db
            )
        except:
            pass
        
        raise HTTPException(status_code=500, detail="Internal server error during chat")


@router.get("/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get chat history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return
        db: Database session
        
    Returns:
        List of chat messages
    """
    try:
        # Get session
        db_session = db.query(DBChatSession).filter(
            DBChatSession.session_id == session_id
        ).first()
        
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get messages
        messages = db.query(DBChatMessage).filter(
            DBChatMessage.session_id == db_session.id
        ).order_by(DBChatMessage.created_at.desc()).limit(limit).all()
        
        # Format response
        history = []
        for msg in reversed(messages):  # Reverse to get chronological order
            history.append({
                "type": msg.message_type,
                "content": msg.content,
                "search_strategy": msg.search_strategy,
                "response_time_seconds": msg.response_time_seconds,
                "created_at": msg.created_at.isoformat()
            })
        
        return {
            "session_id": session_id,
            "messages": history,
            "total_messages": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Clear chat session and its memory.
    
    Args:
        session_id: Session identifier
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Clear Redis memory
        memory_key = f"session:{session_id}:memory"
        redis_client.delete(memory_key)
        
        # Delete database records
        db_session = db.query(DBChatSession).filter(
            DBChatSession.session_id == session_id
        ).first()
        
        if db_session:
            # Delete messages (cascade should handle this, but being explicit)
            db.query(DBChatMessage).filter(
                DBChatMessage.session_id == db_session.id
            ).delete()
            
            # Delete session
            db.delete(db_session)
            db.commit()
        
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clear session")


@router.get("/health")
async def chat_health():
    """Health check endpoint for chat service."""
    try:
        # Test Redis connection
        redis_client.ping()
        
        # Test agents
        rag_agent = get_rag_agent()
        interview_agent = get_interview_agent()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "rag_agent": "initialized",
            "interview_agent": "initialized",
            "supported_strategies": [strategy.value for strategy in SearchStrategy]
        }
    except Exception as e:
        logger.error(f"Chat health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
