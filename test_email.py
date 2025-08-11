#!/usr/bin/env python3
"""Simple email test script to verify MailTrap functionality."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

# MailTrap configuration
SMTP_HOST = "sandbox.smtp.mailtrap.io"
SMTP_PORT = 2525
SMTP_USERNAME = "03fde81563f09d"
SMTP_PASSWORD = "d00990a2587a6f"
FROM_EMAIL = "test@ragassessment.com"

def test_email_connection():
    """Test basic SMTP connection to MailTrap."""
    print("🔧 Testing MailTrap SMTP connection...")
    
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            print("✅ SMTP connection successful!")
            return True
    except Exception as e:
        print(f"❌ SMTP connection failed: {e}")
        return False

def send_test_email(to_email="test@example.com"):
    """Send a test email through MailTrap."""
    print(f"📧 Sending test email to {to_email}...")
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "RAG Assessment - Email Test"
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .header { color: #2c3e50; text-align: center; }
                .content { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
                .success { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 RAG Assessment Email Test</h1>
                </div>
                <div class="content">
                    <p class="success">✅ Email functionality is working correctly!</p>
                    <p>This test email confirms that:</p>
                    <ul>
                        <li>MailTrap SMTP connection is successful</li>
                        <li>Authentication is working</li>
                        <li>Email sending functionality is operational</li>
                    </ul>
                    <p>Your RAG Assessment system is ready to send interview booking confirmations!</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = """
        RAG Assessment - Email Test
        
        ✅ Email functionality is working correctly!
        
        This test email confirms that:
        - MailTrap SMTP connection is successful
        - Authentication is working
        - Email sending functionality is operational
        
        Your RAG Assessment system is ready to send interview booking confirmations!
        """
        
        # Attach parts
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        print("✅ Test email sent successfully!")
        print("📬 Check your MailTrap inbox at: https://mailtrap.io/inboxes")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send test email: {e}")
        return False

def send_booking_confirmation_test():
    """Send a sample booking confirmation email."""
    print("📋 Sending sample booking confirmation email...")
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Interview Booking Confirmation - RAG123"
        msg['From'] = FROM_EMAIL
        msg['To'] = "john.doe@example.com"
        
        # Create HTML content (similar to the actual booking email)
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                .booking-details { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .booking-id { font-size: 24px; font-weight: bold; color: #3498db; text-align: center; margin: 20px 0; }
                .footer { text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Interview Booking Confirmation</h1>
                </div>
                
                <p>Dear John Doe,</p>
                
                <p>Thank you for booking an interview with us! Your interview has been successfully scheduled.</p>
                
                <div class="booking-id">
                    Booking ID: RAG123
                </div>
                
                <div class="booking-details">
                    <h3>Interview Details:</h3>
                    <p><strong>Name:</strong> John Doe</p>
                    <p><strong>Date:</strong> 2024-01-15</p>
                    <p><strong>Time:</strong> 14:00</p>
                    <p><strong>Booking ID:</strong> RAG123</p>
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
        
        # Create plain text version
        text_content = """
        Interview Booking Confirmation
        
        Dear John Doe,
        
        Thank you for booking an interview with us! Your interview has been successfully scheduled.
        
        Booking ID: RAG123
        
        Interview Details:
        - Name: John Doe
        - Date: 2024-01-15
        - Time: 14:00
        - Booking ID: RAG123
        
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
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        print("✅ Sample booking confirmation email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send booking confirmation email: {e}")
        return False

def main():
    """Run all email tests."""
    print("🧪 RAG Assessment - MailTrap Email Testing")
    print("=" * 50)
    
    # Test 1: Basic connection
    if not test_email_connection():
        print("❌ Basic connection test failed. Exiting.")
        sys.exit(1)
    
    print()
    
    # Test 2: Simple test email
    if not send_test_email():
        print("❌ Test email failed.")
    
    print()
    
    # Test 3: Booking confirmation email
    if not send_booking_confirmation_test():
        print("❌ Booking confirmation email failed.")
    
    print()
    print("🎉 Email testing completed!")
    print("📬 Check your MailTrap inbox at: https://mailtrap.io/inboxes")
    print("   Username: 03fde81563f09d")

if __name__ == "__main__":
    main()
