#!/usr/bin/env python3
"""
Generate a new API key for MCP assistant service
"""

import secrets
import string
import os
from datetime import datetime

def generate_api_key(prefix="assistant-mcp-key", length=32):
    """Generate a secure API key"""
    # Generate random string
    alphabet = string.ascii_letters + string.digits + "-"
    random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d")
    
    return f"{prefix}-{timestamp}-{random_part}"

def update_env_file(api_key):
    """Update .env file with new API key"""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print(f"❌ Error: {env_path} file not found")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update API key line
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("ASSISTANT_API_KEY="):
            lines[i] = f"ASSISTANT_API_KEY={api_key}\n"
            updated = True
            break
    
    if not updated:
        # Add new line if not found
        lines.append(f"\nASSISTANT_API_KEY={api_key}\n")
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Updated {env_path} with new API key")
    return True

def update_mcp_json(api_key):
    """Show instructions to update mcp.json"""
    mcp_path = "../.vscode/mcp.json"
    
    print(f"\n📝 To update VS Code MCP configuration:")
    print(f"   File: {mcp_path}")
    print(f"   Update the 'assistant' server headers:")
    print(f'   "headers": {{"X-API-Key": "{api_key}"}}')

def main():
    """Generate new API key and update configuration"""
    print("=" * 60)
    print("MCP ASSISTANT API KEY GENERATOR")
    print("=" * 60)
    
    # Generate new API key
    new_key = generate_api_key()
    print(f"🔑 Generated new API key: {new_key}")
    
    # Update .env file
    if update_env_file(new_key):
        print("✅ Environment file updated")
    
    # Show mcp.json update instructions
    update_mcp_json(new_key)
    
    print("\n⚠️  IMPORTANT:")
    print("   1. Restart the assistant server to use the new API key")
    print("   2. Update mcp.json file manually")
    print("   3. Restart VS Code to reload MCP configuration")
    
    print("\n🔒 Security Notes:")
    print("   - Keep this API key secure")
    print("   - Don't commit it to public repositories")
    print("   - Regenerate if compromised")

if __name__ == "__main__":
    main()
