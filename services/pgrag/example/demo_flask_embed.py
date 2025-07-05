"""
Flask API demo giữ model sentence-transformers ở RAM.
Gửi POST /embed với JSON: {"sentences": ["text1", "text2", ...]}
Trả về embedding dạng list.
"""
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Model chỉ load một lần, giữ ở RAM suốt vòng đời app
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
MODEL = SentenceTransformer(MODEL_NAME)

@app.route('/embed', methods=['POST'])
def embed():
    """Nhận sentences, trả về embeddings."""
    data = request.get_json()
    sentences = data.get('sentences', [])
    if not sentences or not isinstance(sentences, list):
        return jsonify({'error': 'sentences must be a non-empty list'}), 400
    embeddings = MODEL.encode(sentences)
    # Chuyển numpy array sang list để trả về JSON
    return jsonify({'embeddings': embeddings.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
