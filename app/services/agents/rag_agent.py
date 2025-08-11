"""RAG Agent implementation using Google ADK."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import asyncio

import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import SearchStrategy, RetrievedChunk
from app.services.retrieval import get_retrieval_service

logger = get_logger(__name__)


class RAGAgent:
    """RAG Agent for answering questions using retrieved context."""
    
    def __init__(self):
        """Initialize the RAG agent with Google ADK and Gemini."""
        # Configure Google AI
        genai.configure(api_key=settings.google_api_key)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel(settings.gemini_model)
        
        # Initialize retrieval service
        self.retrieval_service = get_retrieval_service()
        
        # Agent configuration
        self.system_prompt = self._build_system_prompt()
        
        logger.info("RAG Agent initialized")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the RAG agent."""
        return """You are a helpful AI assistant that answers questions based on provided context from documents.

INSTRUCTIONS:
1. Use ONLY the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your responses
4. Cite specific parts of the context when relevant
5. If you detect that the user wants to book an interview, respond with: "BOOKING_REQUEST_DETECTED"
6. Maintain a conversational and helpful tone

CONTEXT FORMAT:
Each piece of context will be provided with:
- Content: The actual text content
- Source: Filename and chunk information
- Relevance Score: How relevant this content is to the query

Remember: Only use the provided context. Do not use your general knowledge to supplement the answer."""

    def _format_context(self, retrieved_chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_part = f"""
Context {i}:
Content: {chunk.content}
Source: {chunk.filename} (Chunk ID: {chunk.chunk_id})
Relevance Score: {chunk.similarity_score:.3f}
Search Strategy: {chunk.search_strategy}
---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _detect_booking_intent(self, question: str) -> bool:
        """
        Sophisticated heuristic to detect interview booking intent.
        
        Uses multiple criteria to reduce false positives:
        1. Strong booking indicators (explicit booking language)
        2. Personal information patterns (name, email)
        3. Time/date specifications
        4. Excludes general information queries
        
        Args:
            question: User question
            
        Returns:
            True if booking intent is detected
        """
        import re
        
        question_lower = question.lower()
        
        # Strong indicators that this is NOT a booking request
        information_query_patterns = [
            r'what (is|are|about)',
            r'tell me about',
            r'can you explain',
            r'how (does|do|can)',
            r'what (kind|type|sort) of',
            r'what.*requirements?',
            r'what.*positions?',
            r'what.*services?',
            r'what.*benefits?',
            r'what.*company',
            r'describe.*',
            r'explain.*',
            r'information about',
        ]
        
        # If it matches information query patterns, it's likely NOT a booking
        for pattern in information_query_patterns:
            if re.search(pattern, question_lower):
                # However, if it also has strong booking indicators, it might still be a booking
                if not self._has_strong_booking_indicators(question_lower):
                    return False
        
        # Strong booking intent indicators
        strong_booking_phrases = [
            r'(book|schedule|arrange).*interview',
            r'(want|would like|need) to (book|schedule)',
            r'i (want|would like|need) (an|to book|to schedule)',
            r'can (i|we) (book|schedule)',
            r'(book|schedule) (an|my) interview',
            r'interview.*booking',
            r'make.*appointment',
            r'set up.*interview',
        ]
        
        for phrase in strong_booking_phrases:
            if re.search(phrase, question_lower):
                return True
        
        # Check for personal information + booking context
        has_personal_info = self._has_personal_information(question_lower)
        has_time_date = self._has_time_date_info(question_lower)
        has_booking_context = self._has_booking_context(question_lower)
        
        # If has personal info + time/date + booking context, likely a booking
        if has_personal_info and (has_time_date or has_booking_context):
            return True
        
        # If has strong personal info pattern (name + email), likely a booking
        if self._has_strong_personal_pattern(question_lower):
            return True
        
        return False
    
    def _has_strong_booking_indicators(self, question_lower: str) -> bool:
        """Check for very strong booking language that overrides information queries."""
        import re
        strong_indicators = [
            r'my name is.*my email',
            r'i am.*and.*email',
            r'book.*for.*at.*',
            r'schedule.*for.*at.*',
        ]
        return any(re.search(pattern, question_lower) for pattern in strong_indicators)
    
    def _has_personal_information(self, question_lower: str) -> bool:
        """Check if question contains personal information like name or email."""
        import re
        personal_patterns = [
            r'my name is',
            r'i am [a-z]+ [a-z]+',
            r'i\'m [a-z]+ [a-z]+',
            r'[a-z]+@[a-z]+\.[a-z]+',  # email pattern
            r'my email',
            r'email.*is',
        ]
        return any(re.search(pattern, question_lower) for pattern in personal_patterns)
    
    def _has_time_date_info(self, question_lower: str) -> bool:
        """Check if question contains time or date information."""
        import re
        time_date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}:\d{2}',  # HH:MM
            r'\d{1,2}\s*(am|pm)',  # 2 PM
            r'at \d{1,2}',  # at 2
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(tomorrow|today|next week)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)',
        ]
        return any(re.search(pattern, question_lower) for pattern in time_date_patterns)
    
    def _has_booking_context(self, question_lower: str) -> bool:
        """Check for booking context words in appropriate context."""
        booking_context_words = ['book', 'schedule', 'appointment', 'interview', 'meeting']
        return any(word in question_lower for word in booking_context_words)
    
    def _has_strong_personal_pattern(self, question_lower: str) -> bool:
        """Check for strong personal information patterns that indicate booking intent."""
        import re
        strong_patterns = [
            r'my name is [a-z]+ [a-z]+.*my email',
            r'i am [a-z]+ [a-z]+.*@[a-z]+\.[a-z]+',
            r'[a-z]+ [a-z]+.*@[a-z]+\.[a-z]+.*\d{4}-\d{2}-\d{2}',  # name + email + date
        ]
        return any(re.search(pattern, question_lower) for pattern in strong_patterns)
    
    async def generate_response(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        search_strategy: SearchStrategy = SearchStrategy.COSINE,
        top_k: int = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a question using RAG.
        
        Args:
            question: User question
            conversation_history: Previous conversation messages
            search_strategy: Strategy for retrieving context
            top_k: Number of chunks to retrieve
            document_ids: Optional document IDs to filter by
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Check for booking intent first
            is_booking_request = self._detect_booking_intent(question)
            if is_booking_request:
                return {
                    "answer": "BOOKING_REQUEST_DETECTED",
                    "retrieved_chunks": [],
                    "is_booking_request": True,
                    "search_strategy": search_strategy
                }
            
            # Retrieve relevant context
            retrieved_chunks = self.retrieval_service.retrieve_chunks(
                query=question,
                strategy=search_strategy,
                top_k=top_k,
                document_ids=document_ids
            )
            
            # Format context for the LLM
            context = self._format_context(retrieved_chunks)
            
            # Build conversation history
            messages = []
            if conversation_history:
                for msg in conversation_history[-5:]:  # Keep last 5 messages
                    role = "user" if msg["type"] == "user" else "model"
                    messages.append({"role": role, "parts": [msg["content"]]})
            
            # Build the prompt with context and question
            prompt = f"""
{self.system_prompt}

CONTEXT:
{context}

CONVERSATION HISTORY:
{json.dumps(conversation_history[-3:] if conversation_history else [], indent=2)}

QUESTION: {question}

Please provide a helpful answer based on the context provided above."""

            # Add current question
            messages.append({"role": "user", "parts": [prompt]})
            
            # Generate response using Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                messages
            )
            
            answer = response.text if response.text else "I apologize, but I couldn't generate a response."
            
            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "is_booking_request": False,
                "search_strategy": search_strategy
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "retrieved_chunks": [],
                "is_booking_request": False,
                "search_strategy": search_strategy,
                "error": str(e)
            }
    
    def generate_response_sync(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        search_strategy: SearchStrategy = SearchStrategy.COSINE,
        top_k: int = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_response.
        
        Args:
            question: User question
            conversation_history: Previous conversation messages
            search_strategy: Strategy for retrieving context
            top_k: Number of chunks to retrieve
            document_ids: Optional document IDs to filter by
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Use asyncio.create_task to run in existing event loop
            import asyncio
            import concurrent.futures
            
            # Check if we're in an async context
            try:
                # If we're already in an event loop, use run_in_executor
                loop = asyncio.get_running_loop()
                
                # Create a thread pool executor for the async function
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._run_async_in_thread,
                        question,
                        conversation_history,
                        search_strategy,
                        top_k,
                        document_ids
                    )
                    return future.result()
                    
            except RuntimeError:
                # No event loop running, safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.generate_response(
                            question=question,
                            conversation_history=conversation_history,
                            search_strategy=search_strategy,
                            top_k=top_k,
                            document_ids=document_ids
                        )
                    )
                finally:
                    loop.close()
                    
        except Exception as e:
            logger.error(f"Error in sync RAG response: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "retrieved_chunks": [],
                "is_booking_request": False,
                "search_strategy": search_strategy,
                "error": str(e)
            }
    
    def _run_async_in_thread(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        search_strategy: SearchStrategy = SearchStrategy.COSINE,
        top_k: int = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Helper method to run async code in a separate thread.
        """
        import asyncio
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_response(
                    question=question,
                    conversation_history=conversation_history,
                    search_strategy=search_strategy,
                    top_k=top_k,
                    document_ids=document_ids
                )
            )
        finally:
            loop.close()


# Global RAG agent instance
_rag_agent = None


def get_rag_agent() -> RAGAgent:
    """
    Get the global RAG agent instance.
    
    Returns:
        RAGAgent instance
    """
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent
