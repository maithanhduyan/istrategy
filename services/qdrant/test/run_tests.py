#!/usr/bin/env python3
"""Test runner cho tất cả tests trong thư mục test."""

import sys
import os
import asyncio

# Add parent directory to path để import được modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from test_mcp_endpoint import test_mcp_endpoint


async def main():
    """Chạy tất cả tests."""
    print("🧪 Running Qdrant MCP Tests")
    print("=" * 40)
    
    try:
        print("\n1. Testing MCP Endpoint...")
        await test_mcp_endpoint()
        print("✅ MCP Endpoint test passed!")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
