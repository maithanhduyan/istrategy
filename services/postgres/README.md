# PostgreSQL MCP Server

A Model Context Protocol (MCP) server for PostgreSQL database operations. This server provides a standardized interface for AI assistants to interact with PostgreSQL databases safely and efficiently.

## Features

- **Safe Query Execution**: Execute SELECT queries with built-in SQL validation
- **Database Commands**: Run INSERT, UPDATE, DELETE, and DDL commands
- **Schema Exploration**: List tables and inspect table structures
- **Database Statistics**: Get database metrics and health information
- **Connection Pooling**: Efficient connection management with asyncpg
- **Security**: SQL injection protection and query validation

## Installation

### Using pip

```bash
cd services/postgres
pip install -e .
```

### Using uv (recommended)

```bash
cd services/postgres
uv sync
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your PostgreSQL connection details:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=your_database
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_password
```

## Usage

### Starting the Server

```bash
# Using the installed script
postgres-mcp

# Or using Python module
python -m postgres_mcp.main

# Or using uv
uv run python -m postgres_mcp.main
```

### Available Tools

#### `execute_query`
Execute SELECT queries and return results.

**Parameters:**
- `query` (string): SQL SELECT query
- `params` (array, optional): Query parameters for parameterized queries

**Example:**
```json
{
  "name": "execute_query",
  "arguments": {
    "query": "SELECT * FROM users WHERE age > $1 LIMIT 10",
    "params": ["25"]
  }
}
```

#### `execute_command`
Execute SQL commands (INSERT, UPDATE, DELETE, CREATE, etc.).

**Parameters:**
- `command` (string): SQL command
- `params` (array, optional): Command parameters

**Example:**
```json
{
  "name": "execute_command",
  "arguments": {
    "command": "INSERT INTO users (name, email) VALUES ($1, $2)",
    "params": ["John Doe", "john@example.com"]
  }
}
```

#### `list_tables`
List all tables in the database.

**Example:**
```json
{
  "name": "list_tables",
  "arguments": {}
}
```

#### `describe_table`
Get detailed information about a table.

**Parameters:**
- `table_name` (string): Name of the table
- `schema_name` (string, optional): Schema name (default: "public")

**Example:**
```json
{
  "name": "describe_table",
  "arguments": {
    "table_name": "users",
    "schema_name": "public"
  }
}
```

#### `database_stats`
Get database statistics and information.

**Example:**
```json
{
  "name": "database_stats",
  "arguments": {}
}
```

#### `health_check`
Check database connection health.

**Example:**
```json
{
  "name": "health_check",
  "arguments": {}
}
```

## Development

### Setup Development Environment

```bash
cd services/postgres
uv sync --dev
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/ tests/
uv run isort src/ tests/
```

### Type Checking

```bash
uv run mypy src/
```

## Docker Support

The service includes Docker configuration for easy deployment:

```bash
# Start PostgreSQL with Docker Compose
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs postgres-mcp
```

## Security Considerations

- **Query Validation**: All SQL queries are parsed and validated before execution
- **Read-Only Queries**: SELECT queries are restricted to prevent data modification
- **Parameterized Queries**: Support for parameterized queries to prevent SQL injection
- **Connection Pooling**: Efficient connection management with configurable limits
- **Environment Variables**: Sensitive configuration via environment variables

## Architecture

```
postgres-mcp/
├── src/postgres_mcp/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # MCP server implementation
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection and operations
│   └── tools.py             # MCP tool definitions
├── tests/                   # Test suite
├── docker-compose.yml       # Docker configuration
├── pyproject.toml          # Project configuration
└── README.md               # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Contact the development team at team@istrategy.com
