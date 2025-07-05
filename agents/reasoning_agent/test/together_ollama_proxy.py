#!/usr/bin/env python3
"""
Together.ai to Ollama API Proxy
Chuyển đổi Together.ai API thành Ollama-compatible endpoint cho GitHub Copilot BYOK
"""

import os
import json
import asyncio
from flask import Flask, request, jsonify, Response
import requests
from typing import Dict, Any
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Together.ai configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
LLAMA_2_7B_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEEPSEEK_V3_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_MODEL = DEEPSEEK_V3_MODEL


def convert_ollama_to_together(ollama_request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Ollama API request to Together.ai format"""
    together_request = {
        "model": ollama_request.get("model", DEFAULT_MODEL),
        "messages": [],
        "max_tokens": ollama_request.get("options", {}).get("num_predict", 2048),
        "temperature": ollama_request.get("options", {}).get("temperature", 0.7),
        "stream": ollama_request.get("stream", False),
    }

    # Convert prompt to messages format
    prompt = ollama_request.get("prompt", "")
    if prompt:
        together_request["messages"] = [{"role": "user", "content": prompt}]

    return together_request


def convert_together_to_ollama(together_response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Together.ai response to Ollama format"""
    if "choices" in together_response and together_response["choices"]:
        choice = together_response["choices"][0]
        content = ""

        if "message" in choice:
            content = choice["message"].get("content", "")
        elif "text" in choice:
            content = choice["text"]

        ollama_response = {
            "model": together_response.get("model", DEFAULT_MODEL),
            "created_at": together_response.get("created", ""),
            "response": content,
            "done": True,
            "context": [],
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": len(content.split()),
            "eval_duration": 0,
        }

        return ollama_response

    return {"error": "Invalid response from Together.ai"}


@app.route("/api/generate", methods=["POST"])
def generate():
    """Ollama /api/generate endpoint proxy"""
    try:
        ollama_request = request.json
        together_request = convert_ollama_to_together(ollama_request)

        # Call Together.ai API
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        }

        together_response = requests.post(
            f"{TOGETHER_BASE_URL}/chat/completions",
            json=together_request,
            headers=headers,
        )

        if together_response.status_code == 200:
            together_data = together_response.json()
            ollama_data = convert_together_to_ollama(together_data)
            return jsonify(ollama_data)
        else:
            return (
                jsonify(
                    {"error": f"Together.ai API error: {together_response.status_code}"}
                ),
                500,
            )

    except Exception as e:
        logging.error(f"Proxy error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tags", methods=["GET"])
def tags():
    """Ollama /api/tags endpoint - list available models"""
    return jsonify(
        {
            "models": [
                {
                    "name": "deepseek-ai/DeepSeek-V3",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 0,
                    "digest": "sha256:placeholder",
                },
                {
                    "name": "meta-llama/Llama-2-7b-chat-hf",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 0,
                    "digest": "sha256:placeholder",
                },
            ]
        }
    )


@app.route("/api/version", methods=["GET"])
def version():
    """Ollama version endpoint"""
    return jsonify({"version": "0.1.0-together-proxy"})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "proxy": "together.ai"})


if __name__ == "__main__":
    if not TOGETHER_API_KEY:
        print("⚠️ Warning: TOGETHER_API_KEY environment variable not set")
        print("Set it with: export TOGETHER_API_KEY=your_api_key")

    print("🚀 Starting Together.ai to Ollama API Proxy")
    print(f"📡 Proxy endpoint: http://127.0.0.1:11434")
    print(f"🎯 Target: {TOGETHER_BASE_URL}")
    print("💡 Usage: Configure GitHub Copilot BYOK to use http://127.0.0.1:11434")

    app.run(host="127.0.0.1", port=11434, debug=False)
