"""
Demo embedding đa ngôn ngữ với sentence-transformers.
Tối ưu tốc độ load model bằng cách chỉ khởi tạo model một lần ở cấp module.
Nếu chạy nhiều lần, model sẽ được cache local bởi HuggingFace.
"""
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Tối ưu: chỉ load model một lần ở cấp module
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
MODEL = SentenceTransformer(MODEL_NAME)


def main():
    """Main function to demonstrate multilingual embeddings."""
    texts = [
        "Xin chào thế giới!",  # Vietnamese
        "Hello world!",        # English
        "你好，世界！",           # Chinese
        "Bonjour le monde!",   # French
        "Hallo Welt!",         # German
        "こんにちは世界！",        # Japanese
    ]
    embeddings = MODEL.encode(texts)
    for text, emb in zip(texts, embeddings):
        print(f"Text: {text}")
        print(f"Embedding (first 8 dims): {emb[:8]}")
        print(f"Vector length: {len(emb)}\n")


def demo_multilingual_embedding() -> None:
    """Embed multilingual sentences and print embedding info."""
    sentences = [
        "Hello, how are you?",  # English
        "Xin chào, bạn khỏe không?",  # Vietnamese
        "Bonjour, comment ça va?",  # French
        "Hola, ¿cómo estás?",  # Spanish
        "Hallo, wie geht's dir?",  # German
        "你好，你怎么样？",  # Chinese
        "こんにちは、お元気ですか？",  # Japanese
    ]

    print(f"Embedding {len(sentences)} sentences with model '{MODEL_NAME}'...")
    embeddings = MODEL.encode(sentences)

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print("First vector (truncated):", embeddings[0][:8], "...")

    assert isinstance(embeddings, np.ndarray), "Embeddings should be a numpy array."
    assert embeddings.shape[0] == len(sentences), "Mismatch in number of sentences."
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0."
    print("Demo passed: Embeddings generated successfully.")


if __name__ == "__main__":
    main()
    demo_multilingual_embedding()
