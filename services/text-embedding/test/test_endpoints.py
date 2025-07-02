"""Test script for text embedding endpoints."""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/")
    print("Health check:", response.json())

def test_sentence_transformer_embed():
    """Test standard embedding endpoint."""
    data = {"text": "Hello world, this is a test sentence."}
    response = requests.post(f"{BASE_URL}/embed", json=data)
    result = response.json()
    print(f"\nSentence Transformer Embedding:")
    print(f"Model: {result['model']}")
    print(f"Text: {result['text']}")
    print(f"Embedding dimension: {len(result['embeddings'])}")
    print(f"First 5 values: {result['embeddings'][:5]}")

def test_nomic_embed():
    """Test Nomic AI embedding endpoint."""
    data = {
        "text": "Hello world, this is a test sentence.",
        "prompt_name": "passage"
    }
    response = requests.post(f"{BASE_URL}/embed/nomic", json=data)
    result = response.json()
    print(f"\nNomic AI Embedding:")
    print(f"Model: {result['model']}")
    print(f"Text: {result['text']}")
    print(f"Embedding dimension: {len(result['embeddings'])}")
    print(f"First 5 values: {result['embeddings'][:5]}")

def test_nomic_with_different_prompts():
    """Test Nomic with different prompt names."""
    text = "Machine learning is transforming the world."
    prompts = ["passage", "query"]
    
    print(f"\nTesting Nomic with different prompts:")
    for prompt in prompts:
        data = {"text": text, "prompt_name": prompt}
        response = requests.post(f"{BASE_URL}/embed/nomic", json=data)
        result = response.json()
        print(f"Prompt '{prompt}': dimension {len(result['embeddings'])}, first 3 values: {result['embeddings'][:3]}")

if __name__ == "__main__":
    try:
        test_health()
        test_sentence_transformer_embed()
        test_nomic_embed()
        test_nomic_with_different_prompts()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
