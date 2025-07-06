"""
Database configuration for pgrag project.
"""

import os

POSTGRES_DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "pgrag"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres#2025"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

SQLITE_DB_CONFIG = {
    "dbname": os.getenv("SQLITE_DB", "pgrag.db"),
    "user": os.getenv("SQLITE_USER", ""),
    "password": os.getenv("SQLITE_PASSWORD", ""),
    "host": os.getenv("SQLITE_HOST", ""),
    "port": os.getenv("SQLITE_PORT", ""),
    "uri": os.getenv("SQLITE_URI", "sqlite:///pgrag.db"),
}

CHROMA_DB_CONFIG = {
    "chroma_db_impl": os.getenv("CHROMA_DB_IMPL", "duckdb+parquet"),
    "persist_directory": os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
}

