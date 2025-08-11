"""Interview Agent implementation for handling booking requests."""

from __future__ import annotations

from typing import Dict, Any, Optional
import uuid
import re
from datetime import datetime
import asyncio

import google.generativeai as genai
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.database import get_db
from app.models.database import InterviewBooking as DBInterviewBooking
from app.models.schemas import InterviewBooking, BookingConfirmation
from app.utils.email import send_booking_confirmation_email

logger = get_logger(__name__)


class InterviewAgent:
    """Agent for handling interview booking requests."""
    
    def __init__(self):
        """Initialize the Interview agent with Google Gemini."""
        # Configure Google AI
        genai.configure(api_key=settings.google_api_key)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel(settings.gemini_model)
        
        # System prompt for booking extraction
        self.system_prompt = self._build_system_prompt()
        
        logger.info("Interview Agent initialized")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the interview agent."""
        return """You are an AI assistant specialized in extracting interview booking information from user messages.

Your task is to extract the following information from user messages:
- Name: The person's full name
- Email: Valid email address
- Date: Preferred interview date (format: YYYY-MM-DD)
- Time: Preferred interview time (format: HH:MM)

INSTRUCTIONS:
1. Extract information only if explicitly provided by the user
2. For dates, convert natural language to YYYY-MM-DD format (e.g., "tomorrow" → actual date)
3. For times, convert to 24-hour format (e.g., "2 PM" → "14:00")
4. If information is missing, ask for it politely
5. Validate email format
6. Return response in JSON format

RESPONSE FORMAT:
{
    "status": "complete" | "incomplete" | "error",
    "extracted_info": {
        "name": "string or null",
        "email": "string or null", 
        "date": "YYYY-MM-DD or null",
        "time": "HH:MM or null"
    },
    "missing_fields": ["list of missing required fields"],
    "message": "response message to user",
    "validation_errors": ["list of validation errors if any"]
}

Example responses:
- If all info provided: {"status": "complete", "extracted_info": {...}, "message": "Great! I have all the information needed..."}
- If info missing: {"status": "incomplete", "missing_fields": ["email"], "message": "I need your email address to complete the booking..."}
- If validation fails: {"status": "error", "validation_errors": ["Invalid email format"], "message": "Please provide a valid email address..."}"""

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date format (YYYY-MM-DD)."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def _validate_time(self, time_str: str) -> bool:
        """Validate time format (HH:MM)."""
        try:
            datetime.strptime(time_str, '%H:%M')
            return True
        except ValueError:
            return False
    
    async def extract_booking_info(
        self,
        user_message: str,
        existing_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract booking information from user message.
        
        Args:
            user_message: User's message containing booking information
            existing_info: Previously extracted information to merge with
            
        Returns:
            Dictionary with extraction results and status
        """
        try:
            # Build context with existing info if available
            context = ""
            if existing_info:
                context = f"\nPreviously extracted information: {existing_info}"
            
            prompt = f"""
{self.system_prompt}

{context}

USER MESSAGE: {user_message}

Please extract the booking information and respond in the specified JSON format."""

            # Generate response using Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            # Parse JSON response with improved error handling
            import json
            import re
            
            response_text = response.text.strip()
            logger.info(f"Raw Gemini response: {response_text}")
            
            try:
                # Try to parse as JSON first
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        result = None
                else:
                    result = None
                
                # If JSON parsing completely fails, use regex fallback
                if result is None:
                    logger.warning("JSON parsing failed, using regex fallback")
                    result = self._extract_booking_info_fallback(user_message)
                    if result:
                        logger.info(f"Fallback extraction successful: {result}")
                    else:
                        return {
                            "status": "error",
                            "message": "I had trouble processing your booking request. Could you please provide your name, email, preferred date, and time?",
                            "validation_errors": ["Failed to parse booking information"]
                        }
            
            # Additional validation
            extracted_info = result.get("extracted_info", {})
            validation_errors = []
            
            if extracted_info.get("email"):
                if not self._validate_email(extracted_info["email"]):
                    validation_errors.append("Invalid email format")
            
            if extracted_info.get("date"):
                if not self._validate_date(extracted_info["date"]):
                    validation_errors.append("Invalid date format")
            
            if extracted_info.get("time"):
                if not self._validate_time(extracted_info["time"]):
                    validation_errors.append("Invalid time format")
            
            if validation_errors:
                result["status"] = "error"
                result["validation_errors"] = validation_errors
                result["message"] = f"Please correct the following: {', '.join(validation_errors)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting booking info: {e}")
            return {
                "status": "error",
                "message": "I encountered an error processing your booking request. Please try again.",
                "validation_errors": [str(e)]
            }
    
    def _extract_booking_info_fallback(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Fallback method to extract booking information using regex patterns.
        
        Args:
            user_message: User's message containing booking information
            
        Returns:
            Dictionary with extracted information or None if extraction fails
        """
        import re
        
        try:
            extracted_info = {}
            
            # Extract name patterns
            name_patterns = [
                r'(?:my name is|i am|i\'m|name:?)\s+([a-zA-Z\s]+?)(?:\s*[,.]|\s+and|\s+my|$)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # First Last name pattern
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if len(name.split()) >= 2:  # At least first and last name
                        extracted_info["name"] = name
                        break
            
            # Extract email patterns
            email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            email_match = re.search(email_pattern, user_message)
            if email_match:
                extracted_info["email"] = email_match.group(1)
            
            # Extract date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY or M/D/YYYY
                r'(\d{1,2}-\d{1,2}-\d{4})',  # MM-DD-YYYY
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, user_message)
                if match:
                    date_str = match.group(1)
                    # Convert to YYYY-MM-DD format if needed
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    elif '-' in date_str and len(date_str.split('-')[0]) <= 2:
                        parts = date_str.split('-')
                        if len(parts) == 3:
                            month, day, year = parts
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    
                    extracted_info["date"] = date_str
                    break
            
            # Extract time patterns (HH:MM, H:MM AM/PM, etc.)
            time_patterns = [
                r'(\d{1,2}:\d{2})(?:\s*(?:AM|PM))?',  # HH:MM with optional AM/PM
                r'at\s+(\d{1,2})(?:\s*(?:AM|PM))?',  # "at 2 PM" format
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    time_str = match.group(1)
                    # Convert to 24-hour format if needed
                    if 'PM' in user_message.upper() and ':' in time_str:
                        hour, minute = time_str.split(':')
                        hour = int(hour)
                        if hour != 12:
                            hour += 12
                        time_str = f"{hour:02d}:{minute}"
                    elif 'AM' in user_message.upper() and ':' in time_str:
                        hour, minute = time_str.split(':')
                        hour = int(hour)
                        if hour == 12:
                            hour = 0
                        time_str = f"{hour:02d}:{minute}"
                    elif ':' not in time_str:
                        # Handle "at 2 PM" format
                        hour = int(time_str)
                        if 'PM' in user_message.upper() and hour != 12:
                            hour += 12
                        elif 'AM' in user_message.upper() and hour == 12:
                            hour = 0
                        time_str = f"{hour:02d}:00"
                    
                    extracted_info["time"] = time_str
                    break
            
            # Check if we extracted any information
            if extracted_info:
                # Determine status based on completeness
                required_fields = ["name", "email", "date", "time"]
                missing_fields = [field for field in required_fields if field not in extracted_info or not extracted_info[field]]
                
                if not missing_fields:
                    status = "complete"
                    message = f"Great! I have all the information needed to book your interview for {extracted_info['name']} on {extracted_info['date']} at {extracted_info['time']}."
                else:
                    status = "incomplete"
                    message = f"I need the following information to complete your booking: {', '.join(missing_fields)}"
                
                return {
                    "status": status,
                    "extracted_info": extracted_info,
                    "missing_fields": missing_fields,
                    "message": message
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fallback extraction: {e}")
            return None
    
    def create_booking(
        self,
        name: str,
        email: str,
        date: str,
        time: str,
        session_id: str,
        db: Session
    ) -> BookingConfirmation:
        """
        Create an interview booking in the database.
        
        Args:
            name: Person's name
            email: Email address
            date: Interview date (YYYY-MM-DD)
            time: Interview time (HH:MM)
            session_id: Chat session ID
            db: Database session
            
        Returns:
            BookingConfirmation object
        """
        try:
            # Generate booking ID
            booking_id = str(uuid.uuid4())[:8].upper()
            
            # Create booking record
            booking = DBInterviewBooking(
                booking_id=booking_id,
                name=name,
                email=email,
                preferred_date=date,
                preferred_time=time,
                session_id=session_id,
                status="confirmed"
            )
            
            db.add(booking)
            db.commit()
            db.refresh(booking)
            
            # Send confirmation email
            email_sent = False
            try:
                send_booking_confirmation_email(
                    to_email=email,
                    name=name,
                    booking_id=booking_id,
                    date=date,
                    time=time
                )
                email_sent = True
                
                # Update booking record
                booking.email_sent = True
                db.commit()
                
            except Exception as e:
                logger.error(f"Failed to send confirmation email: {e}")
            
            logger.info(f"Created booking {booking_id} for {name}")
            
            return BookingConfirmation(
                booking_id=booking_id,
                message=f"Your interview has been booked successfully! Booking ID: {booking_id}. You should receive a confirmation email shortly.",
                email_sent=email_sent
            )
            
        except Exception as e:
            logger.error(f"Error creating booking: {e}")
            db.rollback()
            raise RuntimeError(f"Failed to create booking: {e}")
    
    async def handle_booking_request(
        self,
        user_message: str,
        session_id: str,
        existing_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a complete booking request flow.
        
        Args:
            user_message: User's message
            session_id: Chat session ID
            existing_info: Previously extracted information
            
        Returns:
            Dictionary with booking result and response
        """
        try:
            # Extract booking information
            extraction_result = await self.extract_booking_info(user_message, existing_info)
            
            # If extraction is complete, create the booking
            if extraction_result["status"] == "complete":
                extracted_info = extraction_result["extracted_info"]
                
                # Get database session
                db = next(get_db())
                try:
                    booking_confirmation = self.create_booking(
                        name=extracted_info["name"],
                        email=extracted_info["email"],
                        date=extracted_info["date"],
                        time=extracted_info["time"],
                        session_id=session_id,
                        db=db
                    )
                    
                    return {
                        "status": "booking_created",
                        "booking_id": booking_confirmation.booking_id,
                        "message": booking_confirmation.message,
                        "email_sent": booking_confirmation.email_sent
                    }
                    
                finally:
                    db.close()
            
            # Return extraction result for incomplete or error cases
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error handling booking request: {e}")
            return {
                "status": "error",
                "message": "I encountered an error while processing your booking. Please try again.",
                "error": str(e)
            }


# Global interview agent instance
_interview_agent = None


def get_interview_agent() -> InterviewAgent:
    """
    Get the global interview agent instance.
    
    Returns:
        InterviewAgent instance
    """
    global _interview_agent
    if _interview_agent is None:
        _interview_agent = InterviewAgent()
    return _interview_agent
