#!/bin/bash

# Ensure OPENAI_API_KEY is present: env -> repo .env.local -> backend/.env
function find_key_in_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    local line
    line=$(grep -E '^\s*OPENAI_API_KEY\s*=' "$file" | head -n1 || true)
    if [ -n "$line" ]; then
        # extract value after =, strip quotes and whitespace
        echo "$line" | sed -E 's/^\s*OPENAI_API_KEY\s*=\s*//; s/^"|"$//g; s/^'"'|'"'\$//g'
        return 0
    fi
    return 1
}

if [ -z "$OPENAI_API_KEY" ]; then
    # try repo .env.local then backend/.env
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    repo_env_local="$repo_root/.env.local"
    backend_env="$repo_root/backend/.env"
    if key=$(find_key_in_file "$repo_env_local"); then
        export OPENAI_API_KEY="$key"
    elif key=$(find_key_in_file "$backend_env"); then
        export OPENAI_API_KEY="$key"
    else
        echo "" >&2
        echo "ERROR: Missing OpenAI API key." >&2
        echo "Set OPENAI_API_KEY in your shell, add it to .env.local in the repository root, or add it to backend/.env." >&2
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
        echo "Creating .env file..."
        cp env_example.txt .env
        echo ""
        echo "⚠️  Please edit the .env file and add your OpenAI API key:"
        echo "   OPENAI_API_KEY=your_actual_api_key_here"
        echo ""
fi

# Start the server
echo "Starting FastAPI server..."
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 