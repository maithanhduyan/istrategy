"""Test Together.xyz API directly"""

import os
import requests
import json


def test_together_api():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("❌ TOGETHER_API_KEY not set")
        return

    print(f"✅ API Key found: {api_key[:10]}...")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "messages": [{"role": "user", "content": "What is 15 + 27?"}],
        "temperature": 0.1,
        "max_tokens": 100,
    }

    try:
        print("🔄 Calling Together.xyz API...")
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )

        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")


if __name__ == "__main__":
    test_together_api()
