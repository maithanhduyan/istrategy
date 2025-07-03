#!/usr/bin/env python3
"""Test script for new MCP tools."""

import asyncio
import aiohttp
import json

async def test_mcp_tool(tool_name: str, arguments: dict = None):
    """Test an MCP tool via HTTP request."""
    if arguments is None:
        arguments = {}
    
    url = "http://localhost:3004/mcp/tools/call"
    payload = {
        "name": tool_name,
        "arguments": arguments
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ {tool_name}: {result}")
                else:
                    text = await response.text()
                    print(f"❌ {tool_name}: HTTP {response.status} - {text}")
    except Exception as e:
        print(f"❌ {tool_name}: Error - {e}")

async def main():
    """Test all new tools."""
    print("Testing new MCP tools...")
    
    # Test list_tables
    await test_mcp_tool("postgres_list_tables", {
        "database": "polymind_db", 
        "schema": "app"
    })
    
    # Test list_tables with public schema
    await test_mcp_tool("postgres_list_tables", {
        "database": "polymind_db", 
        "schema": "public"
    })
    
    # Test table_structure 
    await test_mcp_tool("postgres_table_structure", {
        "database": "polymind_db",
        "table": "users",  # Assuming this table exists
        "schema": "public"
    })
    
    # Test table_data
    await test_mcp_tool("postgres_table_data", {
        "database": "polymind_db",
        "table": "users",  # Assuming this table exists
        "schema": "public",
        "limit": 5
    })

if __name__ == "__main__":
    asyncio.run(main())
