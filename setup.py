# setup.py - Quick setup script for the RAG Q&A System

import os
import sys
from pathlib import Path

def setup_project():
    """Setup the RAG Q&A System project"""
    print("🚀 Setting up RAG Q&A System with Local LLMs...")
    
    # Create necessary directories
    directories = ['data', 'qdrant_storage']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Create .env file if it doesn't exist
    if not Path('.env').exists():
        env_content = """# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Qdrant Configuration (optional - defaults to localhost)
QDRANT_HOST=localhost
QDRANT_PORT=6333
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    
    # Create .gitignore if it doesn't exist
    if not Path('.gitignore').exists():
        gitignore_content = """# Environment files
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/

# Data and storage
qdrant_storage/
data/
*.pdf
*.docx
*.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")
    
    print("\n📝 Next steps:")
    print("1. Install Ollama from https://ollama.ai")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start Ollama: ollama serve")
    print("4. Pull a model: ollama pull llama3:latest")
    print("5. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("6. Run the app: streamlit run main.py")
    print("\n🎉 Setup complete!")

if __name__ == "__main__":
    setup_project()