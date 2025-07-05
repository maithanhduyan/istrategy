"""
Test script for Flask embedding API.
Gửi request tới endpoint /embed và kiểm tra phản hồi.
"""
import requests
import json

def test_embed_single():
    url = "http://localhost:5000/embed"
    data = {"sentences": ["Xin chào thế giới!"]}
    resp = requests.post(url, json=data)
    print("Test 1 - Single sentence:", resp.status_code, resp.json())

def test_embed_multi():
    url = "http://localhost:5000/embed"
    data = {"sentences": ["Hello world!", "Bonjour le monde!", "こんにちは世界！"]}
    resp = requests.post(url, json=data)
    print("Test 2 - Multi-language:", resp.status_code, resp.json())

def test_embed_invalid():
    url = "http://localhost:5000/embed"
    data = {"text": "Hello"}
    resp = requests.post(url, json=data)
    print("Test 3 - Invalid payload:", resp.status_code, resp.json())

def test_embed_empty():
    url = "http://localhost:5000/embed"
    data = {}
    resp = requests.post(url, json=data)
    print("Test 4 - Empty payload:", resp.status_code, resp.json())

if __name__ == "__main__":
    test_embed_single()
    test_embed_multi()
    test_embed_invalid()
    test_embed_empty()
