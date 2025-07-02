#!/usr/bin/env python3
"""So sánh Starlette mount vs FastAPI mount để hiểu vấn đề."""

import contextlib
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP
from typing import Dict
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from fastapi import FastAPI

# Create MCP server
mcp = FastMCP("test-mcp", stateless_http=True)

@mcp.tool()
async def echo(message: str) -> str:
    """Echo back the provided message."""
    return f"Echo: {message}"

# Test 1: Pure Starlette approach (working)
@contextlib.asynccontextmanager
async def starlette_lifespan(app: Starlette) -> AsyncIterator[None]:
    """Manage MCP server lifecycle."""
    async with mcp.session_manager.run():
        yield

starlette_app = Starlette(
    debug=True,
    routes=[
        Mount("/mcp", app=mcp.streamable_http_app())
    ],
    lifespan=starlette_lifespan,
)

# Test 2: FastAPI approach (có vấn đề)
@contextlib.asynccontextmanager
async def fastapi_lifespan(app: FastAPI):
    """Manage MCP server lifecycle."""
    async with mcp.session_manager.run():
        yield

fastapi_app = FastAPI(title="Test MCP Server", lifespan=fastapi_lifespan)
fastapi_app.mount("/mcp", mcp.streamable_http_app())

@fastapi_app.get("/health")
async def health():
    return {"status": "healthy"}

def test_starlette():
    """Test với Starlette approach."""
    print("Testing Starlette approach...")
    uvicorn.run(starlette_app, host="0.0.0.0", port=3003)

def test_fastapi():
    """Test với FastAPI approach."""
    print("Testing FastAPI approach...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=3004)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        test_fastapi()
    else:
        test_starlette()
