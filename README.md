# 🧠 RAG Q&A System with Qdrant and Local LLMs

A beautiful, Windows-compatible document Q&A system that uses RAG (Retrieval-Augmented Generation) to answer questions about your documents. Built with Streamlit, Qdrant vector database, and Ollama for local LLM inference.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **🎨 Beautiful Modern UI** - Gradient animations, glass-morphism effects, and smooth transitions
- **📄 Multi-format Support** - PDF, TXT, DOCX, and Markdown files
- **🔍 Vector Search** - Powered by Qdrant for fast, semantic search
- **🤖 Local LLMs** - Run models locally with Ollama - no API keys needed!
- **💾 Persistent Storage** - Your documents stay indexed between sessions
- **🔐 100% Private** - Everything runs locally, no data leaves your machine
- **🪟 Windows Compatible** - No Unix-specific dependencies

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Docker** (for Qdrant)
3. **Ollama** - Download from [ollama.ai](https://ollama.ai)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-qa-system

# Run setup script
python setup.py

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Services

#### Start Qdrant
```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Start Ollama
```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama3:latest
# or
ollama pull mistral:latest
```

### 3. Run the Application

```bash
streamlit run main.py
```

Open http://localhost:8501 in your browser.

## 📖 How to Use

1. **Start Services**: Ensure Qdrant and Ollama are running
2. **Upload Documents**: Drag and drop files in the Upload tab
3. **Process**: Click "Process Documents" to create embeddings
4. **Ask Questions**: Switch to Chat tab and start asking questions
5. **View Sources**: Expand the sources section to see which documents were used

## 🤖 Available Models

Ollama supports many models. Popular choices include:

- **llama3** - Meta's latest model, great balance of speed and quality
- **llama2** - Previous generation, still very capable
- **mistral** - Fast and efficient 7B model
- **codellama** - Specialized for code-related queries
- **phi** - Microsoft's small but capable model

Pull any model with:
```bash
ollama pull <model-name>
```

## 🛠️ Technical Details

### Document Processing
- Uses native Python libraries (PyPDF2, python-docx) instead of langchain
- Custom text splitter with configurable chunk size and overlap
- Automatic metadata extraction (page numbers, timestamps)

### Embeddings
- Sentence Transformers (all-MiniLM-L6-v2) for high-quality embeddings
- 384-dimensional vectors for efficient storage and search

### Vector Storage
- Qdrant for scalable vector search
- Cosine similarity for relevance matching
- Batch processing for efficient uploads

### LLM Integration
- Ollama for local LLM inference
- No API keys required - 100% local
- Support for multiple models
- Context-aware responses with source attribution

## 📁 Project Structure

```
rag-qa-system/
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── setup.py               # Setup script
├── .env                   # Configuration (optional)
├── data/                  # Uploaded documents (auto-created)
├── qdrant_storage/        # Vector database storage
└── README.md              # This file
```

## 🔧 Configuration

Edit the `Config` class in `main.py` to customize:

```python
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    OLLAMA_BASE_URL = "http://localhost:11434"
    COLLECTION_NAME = "document_qa"
```

## 🐛 Troubleshooting

### Ollama Issues

**Issue**: "Ollama is not running"
```bash
# Start Ollama service
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

**Issue**: "No models available"
```bash
# Pull a model
ollama pull llama3:latest

# List available models
ollama list
```

### Qdrant Issues

**Issue**: "Cannot connect to Qdrant"
```bash
# Ensure Docker is running
docker ps

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Check health
curl http://localhost:6333/health
```

### Windows-Specific Issues

**Issue**: Module import errors
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
```

### Common Issues

**Issue**: "Model is slow to respond"
- First run of a model downloads and loads it (can take time)
- Subsequent runs are faster
- Consider using smaller models like phi or mistral for speed

**Issue**: Document processing fails
- Check file isn't corrupted
- Ensure file type is supported (PDF, TXT, DOCX, MD)
- Try with a smaller file first

## 🚀 Deployment

### Local Deployment
The app is designed to run locally with Docker for Qdrant and Ollama for LLMs.

### Docker Deployment
```bash
# Start all services
docker-compose up -d
```

### Remote Ollama
You can connect to a remote Ollama instance by setting the URL in the sidebar or .env file:
```
OLLAMA_BASE_URL=http://remote-server:11434
```

## 📊 Performance Tips

1. **Model Selection**: 
   - Use smaller models (phi, mistral) for faster responses
   - Use larger models (llama3-70b) for better quality

2. **Batch Processing**: Process multiple documents at once

3. **Chunk Size**: Adjust based on your document types

4. **GPU Acceleration**: Ollama automatically uses GPU if available

## 🔒 Privacy & Security

- **100% Local**: All processing happens on your machine
- **No API Keys**: No external services or API keys required
- **Data Privacy**: Your documents never leave your computer
- **Secure Storage**: Documents are stored locally in Qdrant

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use in your own projects!

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Qdrant](https://qdrant.tech/) for vector search
- [Ollama](https://ollama.ai/) for local LLM inference
- [Sentence Transformers](https://www.sbert.net/) for embeddings

---

Made with ❤️ for the RAG community