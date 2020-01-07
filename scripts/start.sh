#!/bin/bash
set -e

echo "Starting Document Search API..."

# 检查 Ollama 是否运行
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not running. Please start it with 'ollama serve'"
    exit 1
fi

# 检查 BGE-M3 模型
if ! ollama list | grep -q "bge-m3"; then
    echo "Downloading bge-m3 model..."
    ollama pull bge-m3
fi

# 启动应用
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload