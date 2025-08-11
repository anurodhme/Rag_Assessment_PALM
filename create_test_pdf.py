#!/usr/bin/env python3
"""Create a test PDF for ingestion testing."""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def create_test_pdf():
    """Create a test PDF document."""
    doc = SimpleDocTemplate("test_document.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("RAG Assessment Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Content
    content = [
        "This is a test PDF document for the RAG Assessment system.",
        "",
        "Company Information:",
        "We are a leading AI technology company specializing in machine learning and natural language processing.",
        "",
        "Available Positions:",
        "• Senior Machine Learning Engineer",
        "• Data Scientist",
        "• AI Research Scientist",
        "• Python Developer",
        "",
        "Interview Process:",
        "Candidates can schedule interviews Monday through Friday, 9 AM to 5 PM.",
        "Please provide your name, email, and preferred time slot.",
        "",
        "Requirements:",
        "• 3+ years of Python experience",
        "• Knowledge of TensorFlow or PyTorch",
        "• Experience with data preprocessing",
        "• Strong problem-solving skills",
        "",
        "Benefits:",
        "• Competitive salary package",
        "• Health and dental insurance",
        "• Flexible working hours",
        "• Professional development budget",
        "• Remote work opportunities",
    ]
    
    for line in content:
        if line:
            para = Paragraph(line, styles['Normal'])
            story.append(para)
        story.append(Spacer(1, 6))
    
    doc.build(story)
    print("✅ Created test_document.pdf")

if __name__ == "__main__":
    try:
        create_test_pdf()
    except ImportError:
        print("⚠️  reportlab not installed. Creating simple text file instead.")
        with open("test_document_alt.txt", "w") as f:
            f.write("""RAG Assessment Test Document

This is an alternative test document for the RAG Assessment system.

Company Information:
We are a leading AI technology company specializing in machine learning and natural language processing.

Available Positions:
- Senior Machine Learning Engineer
- Data Scientist  
- AI Research Scientist
- Python Developer

Interview Process:
Candidates can schedule interviews Monday through Friday, 9 AM to 5 PM.
Please provide your name, email, and preferred time slot.

Requirements:
- 3+ years of Python experience
- Knowledge of TensorFlow or PyTorch
- Experience with data preprocessing
- Strong problem-solving skills

Benefits:
- Competitive salary package
- Health and dental insurance
- Flexible working hours
- Professional development budget
- Remote work opportunities
""")
        print("✅ Created test_document_alt.txt (alternative to PDF)")
