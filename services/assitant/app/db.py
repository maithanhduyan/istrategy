# -*- coding: utf-8 -*-
# app/db.py
import sqlite3
import hashlib
import os
from typing import Optional, Dict, Any
from app.logger import get_logger

logger = get_logger(__name__)

DB_PATH = "assistant.db"


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database with users table."""
    conn = get_db_connection()
    try:
        # Create users table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create default admin user if not exists
        admin_password = hash_password("admin123")
        conn.execute('''
            INSERT OR IGNORE INTO users (username, password_hash) 
            VALUES (?, ?)
        ''', ("admin", admin_password))
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()


def hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == password_hash


def create_user(username: str, password: str) -> bool:
    """Create new user."""
    conn = get_db_connection()
    try:
        password_hash = hash_password(password)
        conn.execute('''
            INSERT INTO users (username, password_hash) 
            VALUES (?, ?)
        ''', (username, password_hash))
        conn.commit()
        logger.info(f"User {username} created successfully")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"User {username} already exists")
        return False
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials."""
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            SELECT id, username, password_hash 
            FROM users 
            WHERE username = ?
        ''', (username,))
        user = cursor.fetchone()
        
        if user and verify_password(password, user['password_hash']):
            return {
                'id': user['id'],
                'username': user['username']
            }
        return None
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username."""
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            SELECT id, username, created_at
            FROM users 
            WHERE username = ?
        ''', (username,))
        user = cursor.fetchone()
        
        if user:
            return {
                'id': user['id'],
                'username': user['username'],
                'created_at': user['created_at']
            }
        return None
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None
    finally:
        conn.close()


def get_all_users() -> list:
    """Get all users from database."""
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            SELECT id, username, created_at
            FROM users 
            ORDER BY created_at DESC
        ''')
        users = cursor.fetchall()
        
        return [
            {
                'id': user['id'],
                'username': user['username'],
                'created_at': user['created_at']
            }
            for user in users
        ]
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        return []
    finally:
        conn.close()