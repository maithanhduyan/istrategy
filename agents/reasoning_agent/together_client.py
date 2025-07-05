"""Together.xyz cloud AI client for DeepSeek-R1-Distill-Llama-70B"""

import requests
import json
import os
import time
from typing import Dict, Any, Optional
from config import TEMPERATURE


class TogetherAIClient:
    """Client to interact with Together.xyz API"""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "deepseek-ai/DeepSeek-V3"
    ):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.model = model
        self.api_url = "https://api.together.xyz/v1/chat/completions"

        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

    def generate(self, prompt: str, temperature: float = TEMPERATURE) -> str:
        """Generate response from Together.xyz API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
            "stream": False,
        }

        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                )

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                        print(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "Error: Rate limit exceeded. Please try again later."

                response.raise_for_status()

                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"].strip()

                    # Handle DeepSeek-R1 thinking format
                    if content.startswith("<think>") and "</think>" in content:
                        # Extract content after </think>
                        end_think = content.find("</think>")
                        if end_think != -1:
                            content = content[end_think + 8 :].strip()

                    return content if content else "Error: Empty response content"
                else:
                    return "Error: No response content from Together.xyz API"

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"API error, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                    continue
                return f"Error calling Together.xyz API: {str(e)}"
            except json.JSONDecodeError as e:
                return f"Error parsing Together.xyz API response: {str(e)}"
            except KeyError as e:
                return (
                    f"Error: Unexpected response format from Together.xyz API: {str(e)}"
                )

        return "Error: Max retries exceeded"

    def is_available(self) -> bool:
        """Check if Together.xyz API is available"""
        if not self.api_key:
            return False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Test with a simple request
        test_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }

        try:
            response = requests.post(
                self.api_url, headers=headers, json=test_payload, timeout=10
            )
            # Accept both 200 and 201 status codes
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"Debug: API availability check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "provider": "Together.xyz",
            "api_endpoint": self.api_url,
            "available": self.is_available(),
        }
