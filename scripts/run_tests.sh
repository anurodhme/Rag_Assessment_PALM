#!/bin/bash

# RAG Assessment Test Runner Script

set -e

echo "🧪 RAG Assessment Test Suite"
echo "============================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run linting
echo "🔍 Running code quality checks..."
echo "  - Running ruff..."
ruff check app/ --fix || echo "⚠️  Ruff found issues"

echo "  - Running black..."
black app/ tests/ --check || echo "⚠️  Black formatting issues found"

echo "  - Running mypy..."
mypy app/ --ignore-missing-imports || echo "⚠️  MyPy type checking issues found"

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v --tb=short

echo "✅ Test suite completed!"
echo ""
echo "💡 To run individual test categories:"
echo "   pytest tests/ -m unit          # Unit tests only"
echo "   pytest tests/ -m integration   # Integration tests only"
echo "   pytest tests/ -k test_chunking # Specific test pattern"
