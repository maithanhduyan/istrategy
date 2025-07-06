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


# JWT Configuration
JWT_SECRET_KEY= os.getenv("JWT_SECRET_KEY","your-super-secret-jwt-key-change-in-production")

# MCP API Key for VS Code integration
ASSISTANT_API_KEY= os.getenv("ASSISTANT_API_KEY", "assistant-mcp-key-2025-super-secure-token")

# Default database path for SQLite
DB_PATH= os.getenv("DB_PATH", "assistant.db")