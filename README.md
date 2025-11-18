## Features
- No API keys, no cloud usage
- Local embeddings (MiniLM)
- Local vector store (Chroma)
- Local LLM (Mistral 7B via Ollama)
- Command-line based question answering

---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/Narenderchary85/AmbedkarGPT-Intern-Task

cd AmbedkarGPT-Intern-Task

### 2. Create virtual environment
python -m venv venv
source venv/bin/activate   

### 3. Install Python dependencies
pip install -r requirements.txt

### 4. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

### 5. Pull the Mistral model
ollama pull mistral

### 6. Run the project
python main.py

