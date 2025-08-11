"""Email utility service for sending confirmation emails."""

from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_booking_confirmation_email(
    name: str,
    booking_id: str,
    date: str,
    time: str
) -> str:
    """
    Create HTML content for booking confirmation email.
    
    Args:
        name: Person's name
        booking_id: Booking identifier
        date: Interview date
        time: Interview time
        
    Returns:
        HTML email content
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            .booking-details {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .booking-id {{ font-size: 24px; font-weight: bold; color: #3498db; text-align: center; margin: 20px 0; }}
            .footer {{ text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Interview Booking Confirmation</h1>
            </div>
            
            <p>Dear {name},</p>
            
            <p>Thank you for booking an interview with us! Your interview has been successfully scheduled.</p>
            
            <div class="booking-id">
                Booking ID: {booking_id}
            </div>
            
            <div class="booking-details">
                <h3>Interview Details:</h3>
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Date:</strong> {date}</p>
                <p><strong>Time:</strong> {time}</p>
                <p><strong>Booking ID:</strong> {booking_id}</p>
            </div>
            
            <p>Please save this booking ID for your records. You may need it for any future correspondence regarding your interview.</p>
            
            <p>If you need to reschedule or have any questions, please contact us with your booking ID.</p>
            
            <p>We look forward to meeting with you!</p>
            
            <div class="footer">
                <p>Best regards,<br>The RAG Assessment Team</p>
                <p>This is an automated message. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


def send_booking_confirmation_email(
    to_email: str,
    name: str,
    booking_id: str,
    date: str,
    time: str
) -> bool:
    """
    Send booking confirmation email via SMTP.
    
    Args:
        to_email: Recipient email address
        name: Person's name
        booking_id: Booking identifier
        date: Interview date
        time: Interview time
        
    Returns:
        True if email sent successfully, False otherwise
        
    Raises:
        RuntimeError: If email sending fails
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Interview Booking Confirmation - {booking_id}"
        msg['From'] = settings.from_email
        msg['To'] = to_email
        
        # Create HTML content
        html_content = create_booking_confirmation_email(name, booking_id, date, time)
        
        # Create plain text version
        text_content = f"""
Interview Booking Confirmation

Dear {name},

Thank you for booking an interview with us! Your interview has been successfully scheduled.

Booking ID: {booking_id}

Interview Details:
- Name: {name}
- Date: {date}
- Time: {time}
- Booking ID: {booking_id}

Please save this booking ID for your records. You may need it for any future correspondence regarding your interview.

If you need to reschedule or have any questions, please contact us with your booking ID.

We look forward to meeting with you!

Best regards,
The RAG Assessment Team

This is an automated message. Please do not reply to this email.
        """
        
        # Attach parts
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)
        
        logger.info(f"Booking confirmation email sent to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send booking confirmation email: {e}")
        raise RuntimeError(f"Failed to send email: {e}")


def send_test_email(to_email: str) -> bool:
    """
    Send a test email to verify SMTP configuration.
    
    Args:
        to_email: Test recipient email
        
    Returns:
        True if test email sent successfully
    """
    try:
        msg = MIMEText("This is a test email from the RAG Assessment API.")
        msg['Subject'] = "RAG Assessment API - Test Email"
        msg['From'] = settings.from_email
        msg['To'] = to_email
        
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)
        
        logger.info(f"Test email sent to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send test email: {e}")
        return False
