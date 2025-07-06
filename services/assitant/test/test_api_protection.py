#!/usr/bin/env python3
"""
Test script to verify API key protection for MCP endpoint
"""

import requests
import json
import sys

def test_no_api_key():
    """Test access without API key - should be rejected"""
    print("Testing access without API key...")
    
    url = "http://localhost:8001/mcp"
    headers = {"Content-Type": "application/json"}
    data = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 403 or "Missing API key" in response.text:
            print("✅ PASS: Access correctly rejected without API key")
            return True
        else:
            print(f"❌ FAIL: Expected rejection but got: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_wrong_api_key():
    """Test access with wrong API key - should be rejected"""
    print("Testing access with wrong API key...")
    
    url = "http://localhost:8001/mcp"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "wrong-api-key"
    }
    data = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 403 or "Invalid API key" in response.text:
            print("✅ PASS: Access correctly rejected with wrong API key")
            return True
        else:
            print(f"❌ FAIL: Expected rejection but got: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_correct_api_key():
    """Test access with correct API key - should work"""
    print("Testing access with correct API key...")
    
    url = "http://localhost:8001/mcp"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "assistant-mcp-key-2025-super-secure-token"
    }
    data = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if "result" in result and "tools" in result["result"]:
                print("✅ PASS: Access granted with correct API key")
                print(f"   Found {len(result['result']['tools'])} tools")
                return True
            else:
                print(f"❌ FAIL: Unexpected response format: {result}")
                return False
        else:
            print(f"❌ FAIL: Expected 200 but got: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_mcp_tool_call():
    """Test calling a specific MCP tool"""
    print("Testing MCP tool call...")
    
    url = "http://localhost:8001/mcp"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "assistant-mcp-key-2025-super-secure-token"
    }
    data = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "server_status",
            "arguments": {}
        },
        "id": 2
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if "result" in result and "content" in result["result"]:
                print("✅ PASS: MCP tool call successful")
                print(f"   Response: {result['result']['content'][0]['text']}")
                return True
            else:
                print(f"❌ FAIL: Unexpected response format: {result}")
                return False
        else:
            print(f"❌ FAIL: Expected 200 but got: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MCP API KEY PROTECTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_no_api_key,
        test_wrong_api_key,
        test_correct_api_key,
        test_mcp_tool_call
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! API key protection is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Please check the configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
