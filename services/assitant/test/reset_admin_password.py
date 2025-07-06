#!/usr/bin/env python3
"""
Reset admin password to 'admin123'
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db import get_db_connection, hash_password
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_admin_password():
    """Reset admin password to 'admin123'"""
    conn = get_db_connection()
    try:
        # Hash the new password
        new_password_hash = hash_password("admin123")
        
        # Update admin password
        cursor = conn.execute('''
            UPDATE users 
            SET password_hash = ? 
            WHERE username = 'admin'
        ''', (new_password_hash,))
        
        if cursor.rowcount > 0:
            conn.commit()
            logger.info("Admin password reset successfully to 'admin123'")
            print(f"Username: admin")
            print(f"Password: admin123")
            print(f"New password hash: {new_password_hash}")
        else:
            logger.warning("Admin user not found")
            
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    reset_admin_password()
