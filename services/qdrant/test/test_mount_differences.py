#!/usr/bin/env python3
"""Test cụ thể để chứng minh vấn đề FastAPI mount vs Starlette mount."""

import asyncio
import urllib.request
import urllib.parse
import json
import time
from multiprocessing import Process

def test_endpoint(url, description):
    """Test một endpoint cụ thể."""
    print(f"\n=== Testing {description} ===")
    print(f"URL: {url}")
    
    try:
        # Test initialize request
        data = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            },
            "id": 1
        }
        
        # Prepare request
        json_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=json_data)
        req.add_header('Content-Type', 'application/json')
        req.add_header('Accept', 'application/json, text/event-stream')
        
        # Make request
        response = urllib.request.urlopen(req, timeout=5)
        status_code = response.getcode()
        content = response.read().decode('utf-8')
        
        print(f"Status Code: {status_code}")
        
        if status_code == 200:
            print("✅ SUCCESS - MCP endpoint works")
            # Parse SSE response
            if "event: message" in content:
                print("✅ Valid SSE response format")
            else:
                print("❌ Invalid response format")
                print(f"Response: {content[:200]}...")
        else:
            print(f"❌ FAILED - HTTP {status_code}")
            print(f"Response: {content[:200]}...")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

def run_starlette_server():
    """Chạy Starlette server."""
    import sys
    import os
    sys.path.append(os.getcwd())
    from test_integration_comparison import test_starlette
    test_starlette()

def run_fastapi_server():
    """Chạy FastAPI server."""
    import sys
    import os
    sys.path.append(os.getcwd())
    from test_integration_comparison import test_fastapi
    test_fastapi()

def main():
    """Main test function."""
    print("🔍 Testing MCP endpoint mounting differences")
    
    # Test 1: Starlette approach
    print("\n" + "="*50)
    print("TEST 1: Pure Starlette Mount")
    print("="*50)
    
    starlette_process = Process(target=run_starlette_server)
    starlette_process.start()
    time.sleep(3)  # Wait for server to start
    
    test_endpoint("http://localhost:3003/mcp/", "Starlette /mcp/")
    
    starlette_process.terminate()
    starlette_process.join()
    time.sleep(1)
    
    # Test 2: FastAPI approach  
    print("\n" + "="*50)
    print("TEST 2: FastAPI Mount")
    print("="*50)
    
    fastapi_process = Process(target=run_fastapi_server)
    fastapi_process.start()
    time.sleep(3)  # Wait for server to start
    
    test_endpoint("http://localhost:3004/mcp/", "FastAPI /mcp/")
    
    fastapi_process.terminate()
    fastapi_process.join()
    
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print("This test shows the difference in behavior between:")
    print("1. Pure Starlette mount (working)")
    print("2. FastAPI mount (problematic)")

if __name__ == "__main__":
    main()
