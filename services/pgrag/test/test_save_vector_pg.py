"""
Test lưu, đọc, truy vấn, cập nhật, xóa vector embedding với PostgreSQL (pgvector).
Chạy: uv run test_save_vector_pg.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sentence_transformers import SentenceTransformer
from app.db import get_pool, close_pool, execute_query, fetch_one, fetch_all, get_db_connection, put_db_connection
from app.config import DB_CONFIG

EMBED_TABLE = "test_embeddings"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL = SentenceTransformer(MODEL_NAME)


def setup_table():
    pool = get_pool()
    conn = get_db_connection(pool)
    try:
        # Cài extension pgvector nếu chưa có
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        sql = f"""
        CREATE TABLE IF NOT EXISTS {EMBED_TABLE} (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR(384)
        );
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    finally:
        put_db_connection(pool, conn)


def insert_embedding(text: str) -> int:
    vec = MODEL.encode([text])[0]
    sql = f"INSERT INTO {EMBED_TABLE} (text, embedding) VALUES (%s, %s) RETURNING id;"
    return fetch_one(sql, (text, vec.tolist()))[0]


def get_embedding_by_id(row_id: int):
    sql = f"SELECT id, text, embedding FROM {EMBED_TABLE} WHERE id = %s;"
    return fetch_one(sql, (row_id,))


def query_by_text(text: str):
    sql = f"SELECT id, text FROM {EMBED_TABLE} WHERE text ILIKE %s;"
    return fetch_all(sql, (f"%{text}%",))


def update_text(row_id: int, new_text: str):
    sql = f"UPDATE {EMBED_TABLE} SET text = %s WHERE id = %s;"
    execute_query(sql, (new_text, row_id))


def delete_row(row_id: int):
    sql = f"DELETE FROM {EMBED_TABLE} WHERE id = %s;"
    execute_query(sql, (row_id,))


def main():
    setup_table()
    print("Tạo bảng xong.")
    text = "Đây là một đoạn văn tiếng Việt để kiểm tra lưu embedding."
    row_id = insert_embedding(text)
    print(f"Đã lưu embedding, id={row_id}")
    row = get_embedding_by_id(row_id)
    print("Dữ liệu vừa lưu:", row)
    print("Query theo text:", query_by_text("đoạn văn"))
    update_text(row_id, "Đoạn văn đã được cập nhật.")
    print("Sau khi update:", get_embedding_by_id(row_id))
    delete_row(row_id)
    print("Sau khi xóa:", get_embedding_by_id(row_id))
    close_pool()

if __name__ == "__main__":
    main()

