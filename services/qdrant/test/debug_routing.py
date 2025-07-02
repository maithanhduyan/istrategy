#!/usr/bin/env python3
"""Phân tích routing differences giữa Starlette và FastAPI."""

import contextlib
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from fastapi import FastAPI

# Create MCP server
mcp = FastMCP("debug-mcp", stateless_http=True)

@mcp.tool()
async def echo(message: str) -> str:
    return f"Echo: {message}"

# Debug function để check routes
def print_routes(app, name):
    print(f"\n=== {name} Routes ===")
    if hasattr(app, 'routes'):
        for route in app.routes:
            print(f"Route: {route}")
            if hasattr(route, 'path'):
                print(f"  Path: {route.path}")
            if hasattr(route, 'methods'):
                print(f"  Methods: {route.methods}")
    print("=" * 30)

@contextlib.asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:
    async with mcp.session_manager.run():
        yield

# Test 1: Starlette với debug routes
async def debug_handler(request):
    return JSONResponse({"message": "Debug endpoint", "path": str(request.url.path)})

starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/debug", debug_handler),
        Mount("/mcp", app=mcp.streamable_http_app())
    ],
    lifespan=lifespan,
)

# Test 2: FastAPI với debug routes  
fastapi_app = FastAPI(title="Debug MCP Server", lifespan=lifespan)

@fastapi_app.get("/debug")
async def debug_endpoint():
    return {"message": "Debug endpoint", "framework": "FastAPI"}

# Mount MCP app
fastapi_app.mount("/mcp", mcp.streamable_http_app())

def debug_starlette():
    print_routes(starlette_app, "Starlette")
    uvicorn.run(starlette_app, host="0.0.0.0", port=3005)

def debug_fastapi():
    print_routes(fastapi_app, "FastAPI")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=3006)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        debug_fastapi()
    else:
        debug_starlette()
