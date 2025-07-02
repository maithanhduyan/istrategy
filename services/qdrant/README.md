# Qdrant MCP Server

A Model Context Protocol (MCP) server for Qdrant integration with a single endpoint at `/mcp`.

## Features

- **Single Endpoint**: All MCP communication happens through `/mcp/` endpoint using streamable HTTP transport
- **Three Built-in Tools**:
  - `echo`: Echo back messages
  - `add`: Add two numbers
  - `get_server_status`: Get server status and available tools
- **Stateless HTTP**: Uses stateless streamable HTTP for better scalability
- **VSCode MCP Client Compatible**: Works seamlessly with VSCode MCP client

## Installation

```bash
cd services/qdrant
pip install -e .
```

## Usage

### Start the Server

```bash
python -m src.qdrant_mcp.main
```

The server will start on `http://localhost:3002/mcp/`

### Test with MCP Client

```bash
cd test
python run_tests.py
```

Or test individual components:

```bash
cd test
python test_mcp_endpoint.py
```

### VSCode Integration

Add to your `.vscode/mcp.json`:

```json
{
    "servers": {
        "qdrant-mcp": {
            "url": "http://localhost:3002/mcp/"
        }
    }
}
```

## Architecture

- Uses FastMCP with `stateless_http=True` for streamable HTTP transport
- Exposes single endpoint `/mcp/` (note the trailing slash)
- Runs on port 3002 to avoid conflicts
- All tools are accessible through the MCP protocol

## Project Structure

```
services/qdrant/
├── src/qdrant_mcp/
│   ├── __init__.py
│   ├── main.py              # Main MCP server entry point
│   └── tools/               # Tools modules
│       ├── __init__.py
│       ├── echo.py          # Echo tools
│       ├── time.py          # Time-related tools
│       ├── calculator.py    # Mathematical calculation tools
│       └── qdrant.py        # Qdrant vector database tools
├── test/                    # Test files
│   ├── __init__.py
│   ├── run_tests.py         # Test runner
│   ├── test_mcp_endpoint.py # MCP endpoint tests
│   └── ...                  # Other test files
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## Tools Available

### Echo Tools (echo.py)
- **echo(message: str) -> str**: Echoes back the provided message

### Time Tools (time.py)  
- **get_current_time() -> dict**: Get current date and time information
- **get_timestamp() -> int**: Get current Unix timestamp
- **format_timestamp(timestamp: int, format_string: str) -> str**: Format Unix timestamp

### Calculator Tools (calculator.py)
- **add(a: number, b: number) -> number**: Add two numbers
- **subtract(a: number, b: number) -> number**: Subtract second from first
- **multiply(a: number, b: number) -> number**: Multiply two numbers  
- **divide(a: number, b: number) -> number**: Divide first by second
- **power(base: number, exponent: number) -> number**: Calculate power
- **square_root(number: number) -> number**: Calculate square root
- **calculate_expression(expression: str) -> dict**: Safely evaluate math expressions

### Qdrant Tools (qdrant.py)
- **qdrant_status() -> dict**: Get Qdrant connection status
- **list_collections() -> list**: List all collections  
- **create_collection(name: str, vector_size: int) -> dict**: Create new collection
- **search_vectors(collection_name: str, query_vector: list, limit: int) -> list**: Search vectors
- **insert_vectors(collection_name: str, vectors: list, payloads: list) -> dict**: Insert vectors
- **get_collection_info(collection_name: str) -> dict**: Get collection information

### Legacy Tools
- **get_server_status() -> dict**: Get server status and configuration

## Development

The server is designed following MCP Python SDK best practices:
- Uses FastMCP for high-level server creation
- Implements streamable HTTP transport for production deployment
- Stateless mode for better scalability
- Single endpoint architecture for simplicity
