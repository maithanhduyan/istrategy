"""Ollama client for DeepSeek-R1 model"""

import requests
import json
from typing import Dict, Any
from config import OLLAMA_ENDPOINT, MODEL_NAME, TEMPERATURE


class OllamaClient:
    """Client to interact with Ollama API"""

    def __init__(self, endpoint: str = OLLAMA_ENDPOINT, model: str = MODEL_NAME):
        self.endpoint = endpoint
        self.model = model
        self.api_url = f"{endpoint}/api/generate"

    def generate(self, prompt: str, temperature: float = TEMPERATURE) -> str:
        """Generate response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            },
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()

            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Error parsing Ollama response: {str(e)}"

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=10)
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except:
            return []
