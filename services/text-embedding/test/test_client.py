"""Test script for the text embedding service."""
import requests
import json

def test_embedding_service():
    """Test the embedding service endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Health check: {response.json()}")
    
    # Test embedding endpoint
    print("\nTesting embedding endpoint...")
    test_texts = [
        "Hello world",
        "This is a test sentence for embedding",
        "FastAPI and sentence-transformers work great together!"
    ]
    
    for text in test_texts:
        data = {"text": text}
        response = requests.post(
            f"{base_url}/embed",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Text: {result['text']}")
            print(f"Model: {result['model']}")
            print(f"Embedding dimension: {len(result['embeddings'])}")
            print(f"First 5 values: {result['embeddings'][:5]}")
            print("-" * 50)
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_embedding_service()
