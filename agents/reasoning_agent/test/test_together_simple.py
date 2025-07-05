"""Simple test for Together.xyz integration"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.together_client import TogetherAIClient


def test_basic_math():
    """Test basic math with Together.xyz"""
    
    try:
        client = TogetherAIClient()
        print(f"✅ Client initialized: {client.model}")
        print(f"📊 Available: {client.is_available()}")
        
        # Test simple math
        prompt = """
You are a math assistant. Answer briefly: What is 10 + 5?
Answer with just the number.
"""
        
        print("🔄 Testing basic math...")
        response = client.generate(prompt, temperature=0.1)
        print(f"📝 Response: {response}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    test_basic_math()
