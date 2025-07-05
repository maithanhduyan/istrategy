"""
Tự động test các endpoint của FastAPI embedding service.
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def wait_for_server():
    for _ in range(20):
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print("Server is up.")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("Server not responding.")
    return False

def test_embed():
    data = {"text": "Xin chào thế giới!"}
    r = requests.post(f"{BASE_URL}/embedding", json=data)
    print("/embed:", r.status_code, r.json())

def test_save():
    data = {"text": "Đây là một đoạn văn tiếng Việt để kiểm tra lưu embedding."}
    r = requests.post(f"{BASE_URL}/save", json=data)
    print("/save:", r.status_code, r.json())
    return r.json() if r.status_code == 200 else None

def test_get_embedding(row_id):
    r = requests.get(f"{BASE_URL}/embedding/{row_id}")
    print(f"/embedding/{{row_id}}:", r.status_code, r.json())

def test_search():
    r = requests.get(f"{BASE_URL}/search", params={"query": "đoạn"})
    print("/search:", r.status_code, r.json())

def test_update(row_id):
    data = {"text": "Đoạn văn đã được cập nhật."}
    r = requests.put(f"{BASE_URL}/update/{row_id}", json=data)
    print(f"/update/{{row_id}}:", r.status_code, r.json())

def test_delete(row_id):
    r = requests.delete(f"{BASE_URL}/delete/{row_id}")
    print(f"/delete/{{row_id}}:", r.status_code, r.json())

def main():
    if not wait_for_server():
        return
    test_embed()
    row_id = test_save()
    if row_id:
        test_get_embedding(row_id)
        test_search()
        test_update(row_id)
        test_get_embedding(row_id)
        test_delete(row_id)
        test_get_embedding(row_id)

if __name__ == "__main__":
    main()
