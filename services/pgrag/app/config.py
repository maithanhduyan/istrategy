"""
Database configuration for pgrag project.
"""

import os

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "pgrag"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres#2025"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}

