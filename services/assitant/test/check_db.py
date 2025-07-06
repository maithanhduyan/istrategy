#!/usr/bin/env python3
import sqlite3

def check_database():
    """Check database structure and content."""
    try:
        conn = sqlite3.connect('assistant.db')
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables:", tables)
        
        # Get users table schema
        cursor.execute("PRAGMA table_info(users);")
        schema = cursor.fetchall()
        print("Users table schema:")
        for column in schema:
            print(f"  {column}")
        
        # Get users data
        cursor.execute("SELECT * FROM users;")
        users = cursor.fetchall()
        print("Users data:")
        for user in users:
            print(f"  {user}")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_database()
