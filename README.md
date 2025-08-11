# RAG Assessment API

A comprehensive Retrieval-Augmented Generation (RAG) system built with FastAPI, featuring document ingestion, conversational chat, and interview booking capabilities.

## 🚀 Features

- **Document Ingestion**: Upload and process PDF/TXT files with multiple chunking strategies
- **Conversational RAG**: Chat interface with session memory and context retrieval
- **Interview Booking**: Intelligent booking system with email confirmations
- **Multiple Search Strategies**: Cosine similarity and hybrid (cosine + BM25) search
- **Evaluation Framework**: Comprehensive benchmarking of chunking and search strategies
- **Scalable Architecture**: Dockerized with PostgreSQL, Milvus, and Redis

## 🏗️ Architecture

```
├── app/
│   ├── api/                  # FastAPI routers
│   │   ├── ingestion.py      # Document ingestion endpoint
│   │   └── chat.py           # Conversational RAG endpoint
│   ├── core/                 # Configuration and database
│   ├── services/             # Business logic
│   │   ├── chunking.py       # Document chunking strategies
│   │   ├── embedding.py      # Text embedding generation
│   │   ├── storage_milvus.py # Vector storage with Milvus
│   │   ├── retrieval.py      # Search and retrieval strategies
│   │   └── agents/           # AI agents
│   │       ├── rag_agent.py      # RAG conversation agent
│   │       └── interview_agent.py # Interview booking agent
│   ├── models/               # Data models
│   └── utils/                # Utilities
├── evaluation/               # Evaluation framework
├── tests/                    # Test suite
└── docker-compose.yml        # Container orchestration
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | REST API server |
| **Vector Database** | Milvus | Embedding storage and similarity search |
| **Database** | PostgreSQL | Metadata and session storage |
| **Cache/Memory** | Redis | Session memory and caching |
| **Embeddings** | Sentence Transformers | Text vectorization |
| **LLM** | Google Gemini 2.5-flash | Response generation |
| **Agents** | Google ADK | Agent orchestration |
| **Email** | SMTP (MailTrap) | Booking confirmations |

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Google API key for Gemini
- MailTrap account for email testing

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rag-assessment
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```bash
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key_here

# Email Configuration (MailTrap)
SMTP_USERNAME=your_mailtrap_username
SMTP_PASSWORD=your_mailtrap_password

# Other settings (defaults should work for development)
SECRET_KEY=your_secret_key_here
```

### 3. Start Services

```bash
# Start all services
docker-compose up --build

# Or start in background
docker-compose up -d --build
```

### 4. Verify Installation

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Milvus Admin: http://localhost:9001 (minioadmin/minioadmin)

## 📚 API Usage

### Document Ingestion

Upload and process documents:

```bash
curl -X POST "http://localhost:8000/ingest/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "chunking_strategy=semantic" \
  -F "chunk_size=800"
```

### Conversational Chat

Start a conversation:

```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "question": "What is machine learning?",
    "search_strategy": "hybrid",
    "top_k": 3
  }'
```

### Interview Booking

Book an interview through chat:

```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "question": "I would like to book an interview",
    "name": "John Doe",
    "email": "john@example.com",
    "date": "2024-01-15",
    "time": "14:00"
  }'
```

## 🔧 Configuration

### Chunking Strategies

- **Late Chunking**: Fixed window with overlap
- **Semantic Chunking**: Similarity-based splitting

### Search Strategies

- **Cosine**: Pure vector similarity search
- **Hybrid**: Combines cosine similarity with BM25

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_CHUNK_SIZE` | Chunk size in tokens | 800 |
| `DEFAULT_CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `DEFAULT_TOP_K` | Chunks to retrieve | 3 |
| `MAX_FILE_SIZE_MB` | Max upload size | 20 |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |

## 📊 Evaluation

Run comprehensive evaluation:

```bash
# Inside the container or with proper Python environment
python evaluation/benchmark.py
```

This generates a detailed report comparing all strategy combinations:
- Precision, Recall, F1-score metrics
- Latency measurements
- Strategy recommendations

## 🧪 Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_ingestion.py -v
```

### Code Quality

```bash
# Type checking
mypy app/

# Linting
ruff check app/

# Formatting
black app/
```

## 📁 Project Structure

```
rag-assessment/
├── app/                      # Main application
│   ├── api/                  # API endpoints
│   ├── core/                 # Core configuration
│   ├── models/               # Data models
│   ├── services/             # Business logic
│   │   ├── agents/           # AI agents
│   │   ├── chunking.py       # Text chunking
│   │   ├── embedding.py      # Embeddings
│   │   ├── retrieval.py      # Search strategies
│   │   └── storage_milvus.py # Vector storage
│   └── utils/                # Utilities
├── evaluation/               # Evaluation framework
├── tests/                    # Test suite
├── docker-compose.yml        # Container setup
├── Dockerfile               # Application container
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔍 Monitoring and Debugging

### Health Checks

- Application: `GET /health`
- Ingestion: `GET /ingest/health`
- Chat: `GET /chat/health`

### Logs

```bash
# View application logs
docker-compose logs api

# View all service logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f
```

### Database Access

```bash
# PostgreSQL
docker-compose exec postgres psql -U postgres -d ragdb

# Redis CLI
docker-compose exec redis redis-cli
```

## 🚨 Troubleshooting

### Common Issues

1. **Milvus Connection Failed**
   - Ensure all containers are running
   - Check if ports 19530 and 9091 are available

2. **Google API Quota Exceeded**
   - Check your Google Cloud Console for API limits
   - Implement rate limiting if needed

3. **Email Not Sending**
   - Verify MailTrap credentials in `.env`
   - Check SMTP settings and firewall rules

4. **Out of Memory**
   - Reduce batch sizes in embedding generation
   - Increase Docker memory limits

### Performance Tuning

- Adjust `DEFAULT_CHUNK_SIZE` based on your documents
- Use `HYBRID` search for better relevance
- Enable Redis persistence for session recovery
- Configure Milvus index parameters for your data size

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the [documentation](http://localhost:8000/docs)
- Review the [health endpoints](http://localhost:8000/health)
- Open an issue in the repository

---

Built with ❤️ using FastAPI, Milvus, and Google ADK
