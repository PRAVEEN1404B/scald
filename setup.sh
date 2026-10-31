#!/bin/bash
set -e

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p logs/ results/ demo/

# Bootstrap local environment file
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "NOTE: edit .env to set your LLM backend and credentials."
fi

# Check ollama
if ! command -v ollama &> /dev/null; then
    echo "WARNING: ollama command not found. Please install Ollama if you plan to use local models."
else
    # Check if any model is pulled
    # ollama list outputs a header, so if there's only 1 line, there are no models
    models=$(ollama list | wc -l)
    if [ "$models" -le 1 ]; then
        echo "WARNING: No models pulled in Ollama. Please run e.g., 'ollama pull qwen2.5:7b'."
    fi
fi

echo "Setup completed successfully!"
