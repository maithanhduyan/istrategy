from mcp.server.lowlevel import Server
from typing import Dict, List, TypedDict, Union
import mcp.types as types
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import contextlib
from typing import AsyncIterator
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from psycopg2 import pool

import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create MCP server using low-level Server for StreamableHTTP compatibility
app = Server("postgres-mcp")

# Global connection pool variable
postgres_pool = None


async def wait_for_postgres_and_pool(max_retries=10, delay=2):
    """Đợi kết nối với PostgreSQL, thử lại nhiều lần và khởi tạo connection pool."""
    import time
    from psycopg2 import pool

    for attempt in range(1, max_retries + 1):
        logger.info(f"Thử kết nối PostgreSQL (lần {attempt}/{max_retries})...")
        try:
            p = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host="localhost",
                port=5432,
                user="postgres",
                password="postgres#2025",
                database="postgres",
            )
            # Test lấy 1 connection
            conn = p.getconn()
            p.putconn(conn)
            logger.info("Kết nối PostgreSQL và khởi tạo pool thành công!")
            return p
        except Exception as e:
            logger.warning(f"Kết nối thất bại: {e}")
            time.sleep(delay)
    logger.error("Không thể kết nối PostgreSQL sau nhiều lần thử!")
    return None


def get_server_status() -> str:
    """Get the current server status."""
    return "Postgres MCP Server is running successfully!"


async def safe_execute_query(query: str, params=None):
    """Thực thi query an toàn với connection pool"""
    global postgres_pool
    if postgres_pool is None:
        raise RuntimeError("Postgres connection pool is not initialized.")
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            raise RuntimeError("No available connection in the pool.")
        # Kiểm tra xem query có phải là SELECT, WITH hoặc EXPLAIN không
        q = query.strip().upper()
        if not (
            q.startswith("SELECT") or q.startswith("WITH") or q.startswith("EXPLAIN")
        ):
            raise ValueError(
                "Chỉ cho phép thực thi các câu lệnh SELECT, WITH hoặc EXPLAIN."
            )
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        if cursor.description:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            return {"success": True, "data": rows, "columns": columns}
        else:
            cursor.close()
            return {"success": True, "message": "Query thực thi thành công"}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


async def postgres_count_databases():
    """Đếm số lượng databases trong PostgreSQL."""
    query = "SELECT COUNT(*) FROM pg_database WHERE datistemplate = false;"
    result = await safe_execute_query(query)
    if result["success"]:
        return {"success": True, "count": result["data"][0][0]}
    else:
        return {"success": False, "message": result["message"]}


async def postgres_list_databases():
    """Liệt kê tất cả databases trong PostgreSQL."""
    query = "SELECT datname FROM pg_database WHERE datistemplate = false;"
    result = await safe_execute_query(query)
    if result["success"]:
        return {"success": True, "databases": [row[0] for row in result["data"]]}
    else:
        return {"success": False, "message": result["message"]}


async def postgres_list_schemas(database: str):
    """Liệt kê tất cả schemas trong một database cụ thể."""
    global postgres_pool
    if postgres_pool is None:
        return {
            "success": False,
            "message": "Postgres connection pool is not initialized.",
        }
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            return {"success": False, "message": "No available connection in the pool."}
        # Đổi database nếu cần
        if conn.info.dbname != database:
            conn.set_isolation_level(0)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid();",
                (database,),
            )
            cursor.close()
            conn.close()
            conn = postgres_pool.getconn()
        cursor = conn.cursor()
        query = "SELECT schema_name FROM information_schema.schemata;"
        cursor.execute(query)
        rows = cursor.fetchall()
        schemas = [row[0] for row in rows]
        cursor.close()
        return {"success": True, "schemas": schemas}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


async def postgres_list_tables(database: str, schema: str = "public"):
    """Liệt kê tất cả tables trong một schema cụ thể của database."""
    global postgres_pool
    if postgres_pool is None:
        return {
            "success": False,
            "message": "Postgres connection pool is not initialized.",
        }
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            return {"success": False, "message": "No available connection in the pool."}
        cursor = conn.cursor()
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_catalog = %s
            ORDER BY table_name;
        """
        cursor.execute(query, (schema, database))
        rows = cursor.fetchall()
        tables = [row[0] for row in rows]
        cursor.close()
        return {"success": True, "tables": tables}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


async def postgres_table_structure(database: str, table: str, schema: str = "public"):
    """Lấy cấu trúc của một table cụ thể trong database."""
    global postgres_pool
    if postgres_pool is None:
        return {
            "success": False,
            "message": "Postgres connection pool is not initialized.",
        }
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            return {"success": False, "message": "No available connection in the pool."}
        cursor = conn.cursor()
        query = """
            SELECT column_name, 
                data_type, 
                is_nullable, 
                column_default,
                character_maximum_length 
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s AND table_catalog = %s
            ORDER BY ordinal_position;
        """
        cursor.execute(query, (schema, table, database))
        rows = cursor.fetchall()
        cursor.close()
        return {"success": True, "structure": rows}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


async def postgres_table_data(
    database: str, table: str, schema: str = "public", limit: int = 100
):
    """Lấy dữ liệu của một table cụ thể trong database."""
    global postgres_pool
    if postgres_pool is None:
        return {
            "success": False,
            "message": "Postgres connection pool is not initialized.",
        }
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            return {"success": False, "message": "No available connection in the pool."}
        from psycopg2 import sql

        cursor = conn.cursor()
        query = sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        cursor.close()
        return {"success": True, "data": rows, "columns": columns}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


async def postgres_execute_query(database: str, query: str, params: list = None):
    """Thực thi SQL query trong database được chỉ định."""
    global postgres_pool
    if postgres_pool is None:
        return {
            "success": False,
            "message": "Postgres connection pool is not initialized.",
        }
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            return {"success": False, "message": "No available connection in the pool."}
        cursor = conn.cursor()
        q = query.strip().upper()
        if not (
            q.startswith("SELECT") or q.startswith("WITH") or q.startswith("EXPLAIN")
        ):
            return {
                "success": False,
                "message": "Chỉ cho phép thực thi các câu lệnh SELECT, WITH hoặc EXPLAIN.",
            }
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        if cursor.description:
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            return {"success": True, "data": rows, "columns": columns}
        else:
            cursor.close()
            return {"success": True, "message": "Query thực thi thành công"}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


async def postgres_health_check(database: str = "postgres"):
    """Kiểm tra sức khỏe database và connection pool."""
    import psycopg2

    global postgres_pool

    health_info = {
        "database": database,
        "timestamp": time.time(),
        "pool_status": {},
        "database_status": {},
        "overall_status": "unknown",
    }

    try:
        # Kiểm tra connection pool
        if postgres_pool:
            # Lấy thông tin pool từ global pool
            health_info["pool_status"] = {
                "pool_available": True,
                "minconn": postgres_pool.minconn,
                "maxconn": postgres_pool.maxconn,
                "status": "healthy",
            }
        else:
            health_info["pool_status"] = {
                "pool_available": False,
                "status": "error",
                "message": "Connection pool not initialized",
            }

        # Kiểm tra kết nối database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres#2025",
            database=database,
        )
        cursor = conn.cursor()

        # Thực hiện các kiểm tra cơ bản
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]

        cursor.execute("SELECT current_database()")
        current_db = cursor.fetchone()[0]

        cursor.execute("SELECT NOW()")
        db_time = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'")
        active_connections = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        health_info["database_status"] = {
            "connection": "successful",
            "version": version,
            "current_database": current_db,
            "server_time": str(db_time),
            "active_connections": active_connections,
            "status": "healthy",
        }

        health_info["overall_status"] = "healthy"
        return {"success": True, "health_info": health_info}

    except Exception as e:
        health_info["database_status"] = {
            "connection": "failed",
            "error": str(e),
            "status": "error",
        }
        health_info["overall_status"] = "unhealthy"
        return {"success": False, "health_info": health_info, "message": str(e)}


async def postgres_database_stats(database: str):
    """Lấy thống kê chi tiết về database."""
    global postgres_pool
    if postgres_pool is None:
        return {
            "success": False,
            "message": "Postgres connection pool is not initialized.",
        }
    conn = None
    try:
        conn = postgres_pool.getconn()
        if conn is None:
            return {"success": False, "message": "No available connection in the pool."}
        cursor = conn.cursor()

        # Database basic info
        cursor.execute("SELECT current_database(), current_user, version()")
        db_info = cursor.fetchone()

        # Database size
        cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cursor.fetchone()[0]

        # Table statistics (dùng đúng cột tablename của pg_tables)
        cursor.execute(
            """
            SELECT 
                schemaname,
                COUNT(*) as table_count,
                SUM(pg_total_relation_size(schemaname||'.'||tablename)) as total_size
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            GROUP BY schemaname
        """
        )
        schema_stats = cursor.fetchall()

        # Connection statistics
        cursor.execute(
            """
            SELECT 
                state,
                COUNT(*) as count
            FROM pg_stat_activity 
            WHERE datname = current_database()
            GROUP BY state
        """
        )
        connection_stats = cursor.fetchall()

        # Index usage statistics (fix: dùng relname thay vì tablename nếu cần)
        cursor.execute(
            """
            SELECT 
                schemaname,
                relname as table,
                indexrelname as index,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes 
            ORDER BY idx_scan DESC
            LIMIT 10
        """
        )
        index_stats = cursor.fetchall()

        cursor.close()
        return {
            "success": True,
            "database_info": {
                "name": db_info[0],
                "user": db_info[1],
                "version": db_info[2],
                "size": db_size,
            },
            "schema_stats": [
                {
                    "schema": row[0],
                    "table_count": row[1],
                    "total_size": row[2] if row[2] else 0,
                }
                for row in schema_stats
            ],
            "connection_stats": [
                {"state": row[0] or "unknown", "count": row[1]}
                for row in connection_stats
            ],
            "top_indexes": [
                {
                    "schema": row[0],
                    "table": row[1],
                    "index": row[2],
                    "scans": row[3],
                    "tuples_read": row[4],
                    "tuples_fetched": row[5],
                }
                for row in index_stats
            ],
        }
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn is not None:
            postgres_pool.putconn(conn)


@app.call_tool()
async def handle_all_tools(name: str, arguments: dict) -> list[types.ContentBlock]:
    """Handle all tool calls in one unified handler."""
    match name:
        case "postgres_server_status":
            status_message = get_server_status()
            return [types.TextContent(type="text", text=status_message)]
        case "postgres_count_databases":
            result = await postgres_count_databases()
            if result["success"]:
                return [
                    types.TextContent(
                        type="text", text=f"Total databases: {result['count']}"
                    )
                ]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_list_databases":
            result = await postgres_list_databases()
            if result["success"]:
                dbs = ", ".join(result["databases"])
                return [types.TextContent(type="text", text=f"Databases: {dbs}")]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_list_schemas":
            db = arguments.get("database")
            if not db:
                return [
                    types.TextContent(type="text", text="Missing 'database' argument")
                ]
            result = await postgres_list_schemas(db)
            if result["success"]:
                schemas = ", ".join(result["schemas"])
                return [
                    types.TextContent(type="text", text=f"Schemas in {db}: {schemas}")
                ]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_list_tables":
            db = arguments.get("database")
            schema = arguments.get("schema", "public")
            if not db:
                return [
                    types.TextContent(type="text", text="Missing 'database' argument")
                ]
            result = await postgres_list_tables(db, schema)
            if result["success"]:
                tables = ", ".join(result["tables"])
                return [
                    types.TextContent(
                        type="text", text=f"Tables in {db}.{schema}: {tables}"
                    )
                ]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_table_structure":
            db = arguments.get("database")
            table = arguments.get("table")
            schema = arguments.get("schema", "public")
            if not db or not table:
                return [
                    types.TextContent(
                        type="text", text="Missing 'database' or 'table' argument"
                    )
                ]
            result = await postgres_table_structure(db, table, schema)
            if result["success"]:
                structure_info = []
                for row in result["structure"]:
                    col_name, data_type, is_nullable, default, max_length = row
                    info = f"{col_name}: {data_type}"
                    if max_length:
                        info += f"({max_length})"
                    if is_nullable == "NO":
                        info += " NOT NULL"
                    if default:
                        info += f" DEFAULT {default}"
                    structure_info.append(info)
                structure_text = "\n".join(structure_info)
                return [
                    types.TextContent(
                        type="text",
                        text=f"Structure of {db}.{schema}.{table}:\n{structure_text}",
                    )
                ]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_table_data":
            db = arguments.get("database")
            table = arguments.get("table")
            schema = arguments.get("schema", "public")
            limit = arguments.get("limit", 100)
            if not db or not table:
                return [
                    types.TextContent(
                        type="text", text="Missing 'database' or 'table' argument"
                    )
                ]
            result = await postgres_table_data(db, table, schema, limit)
            if result["success"]:
                if not result["data"]:
                    return [
                        types.TextContent(
                            type="text", text=f"Table {db}.{schema}.{table} is empty"
                        )
                    ]
                # Format data as simple text
                data_text = f"Data from {db}.{schema}.{table} (limit {limit}):\n"
                data_text += "Columns: " + ", ".join(result["columns"]) + "\n"
                data_text += f"Rows: {len(result['data'])}\n"
                for i, row in enumerate(result["data"][:10]):  # Show max 10 rows
                    data_text += f"Row {i+1}: {row}\n"
                if len(result["data"]) > 10:
                    data_text += f"... and {len(result['data']) - 10} more rows"
                return [types.TextContent(type="text", text=data_text)]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_execute_query":
            db = arguments.get("database")
            query = arguments.get("query")
            params = arguments.get("params", None)
            if not db or not query:
                return [
                    types.TextContent(
                        type="text", text="Missing 'database' or 'query' argument"
                    )
                ]
            result = await postgres_execute_query(db, query, params)
            if result["success"]:
                if result.get("data"):
                    # Format query results
                    data_text = f"Query Results:\n"
                    data_text += "Columns: " + ", ".join(result["columns"]) + "\n"
                    data_text += f"Rows: {len(result['data'])}\n"
                    for i, row in enumerate(result["data"][:20]):  # Show max 20 rows
                        data_text += f"Row {i+1}: {row}\n"
                    if len(result["data"]) > 20:
                        data_text += f"... and {len(result['data']) - 20} more rows"
                    return [types.TextContent(type="text", text=data_text)]
                else:
                    return [
                        types.TextContent(
                            type="text",
                            text=result.get("message", "Query executed successfully"),
                        )
                    ]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case "postgres_health_check":
            db = arguments.get("database", "postgres")
            result = await postgres_health_check(db)
            if result["success"]:
                health_info = result["health_info"]
                return [
                    types.TextContent(
                        type="text",
                        text=f"Health check successful!\nDatabase: {health_info['database']}\nStatus: {health_info['overall_status']}\n\nDatabase Status:\n- Connection: {health_info['database_status']['connection']}\n- Version: {health_info['database_status']['version']}\n- Current DB: {health_info['database_status']['current_database']}\n- Server Time: {health_info['database_status']['server_time']}\n- Active Connections: {health_info['database_status']['active_connections']}\n\nPool Status:\n- Available: {health_info['pool_status']['pool_available']}\n- Min Connections: {health_info['pool_status']['minconn']}\n- Max Connections: {health_info['pool_status']['maxconn']}",
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text", text=f"Health check failed: {result['message']}"
                    )
                ]
        case "postgres_database_stats":
            db = arguments.get("database")
            if not db:
                return [
                    types.TextContent(type="text", text="Missing 'database' argument")
                ]
            result = await postgres_database_stats(db)
            if result["success"]:
                db_info = result["database_info"]
                schema_stats = result["schema_stats"]
                connection_stats = result["connection_stats"]
                top_indexes = result["top_indexes"]

                # Format schema statistics
                schema_text = "\n".join(
                    [
                        f"Schema: {s['schema']}, Tables: {s['table_count']}, Size: {s['total_size']} bytes"
                        for s in schema_stats
                    ]
                )

                # Format connection statistics
                connection_text = "\n".join(
                    [
                        f"State: {c['state']}, Count: {c['count']}"
                        for c in connection_stats
                    ]
                )

                # Format index statistics
                index_text = "\n".join(
                    [
                        f"Schema: {i['schema']}, Table: {i['table']}, Index: {i['index']}, Scans: {i['scans']}"
                        for i in top_indexes
                    ]
                )

                return [
                    types.TextContent(
                        type="text",
                        text=f"Database Statistics for {db}:\n\n"
                        f"Basic Info:\n"
                        f"- Name: {db_info['name']}\n"
                        f"- User: {db_info['user']}\n"
                        f"- Version: {db_info['version']}\n"
                        f"- Size: {db_info['size']}\n\n"
                        f"Schema Statistics:\n{schema_text}\n\n"
                        f"Connection Statistics:\n{connection_text}\n\n"
                        f"Top Indexes:\n{index_text}",
                    )
                ]
            else:
                return [
                    types.TextContent(type="text", text=f"Error: {result['message']}")
                ]
        case _:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="postgres_server_status",
            description="Get the current status of the MCP server.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="postgres_count_databases",
            description="Count the number of databases in PostgreSQL.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="postgres_list_databases",
            description="List all databases in PostgreSQL.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="postgres_list_schemas",
            description="List all schemas in a specific database.",
            inputSchema={
                "type": "object",
                "properties": {"database": {"type": "string"}},
                "required": ["database"],
            },
        ),
        types.Tool(
            name="postgres_list_tables",
            description="List all tables in a specific schema of a database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "schema": {"type": "string", "default": "public"},
                },
                "required": ["database"],
            },
        ),
        types.Tool(
            name="postgres_table_structure",
            description="Get the structure (columns, types, constraints) of a specific table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "table": {"type": "string"},
                    "schema": {"type": "string", "default": "public"},
                },
                "required": ["database", "table"],
            },
        ),
        types.Tool(
            name="postgres_table_data",
            description="Get data from a specific table with optional limit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "table": {"type": "string"},
                    "schema": {"type": "string", "default": "public"},
                    "limit": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
                "required": ["database", "table"],
            },
        ),
        types.Tool(
            name="postgres_execute_query",
            description="Execute a SQL SELECT query in a specific database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "query": {"type": "string"},
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": None,
                    },
                },
                "required": ["database", "query"],
            },
        ),
        types.Tool(
            name="postgres_health_check",
            description="Check the health of the database and connection pool.",
            inputSchema={
                "type": "object",
                "properties": {"database": {"type": "string", "default": "postgres"}},
                "required": [],
            },
        ),
        types.Tool(
            name="postgres_database_stats",
            description="Get detailed statistics about the database.",
            inputSchema={
                "type": "object",
                "properties": {"database": {"type": "string"}},
                "required": ["database"],
            },
        ),
    ]


def main():
    """Main function to start the MCP server."""
    logger.info("Starting Postgres MCP server...")

    # Create session manager with stateless mode for better performance
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,  # No event store for stateless mode
        json_response=False,  # Use SSE streams
        stateless=True,  # Enable stateless mode
    )

    # ASGI handler for streamable HTTP connections
    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        # Khởi tạo pool khi app startup
        global postgres_pool
        postgres_pool = await wait_for_postgres_and_pool()
        if not postgres_pool:
            logger.error("Thoát do không kết nối được PostgreSQL!")
            raise RuntimeError("Không thể khởi tạo pool!")
        async with session_manager.run():
            logger.info(
                "Postgres MCP server started with StreamableHTTP session manager!"
            )
            try:
                yield
            finally:
                if postgres_pool:
                    postgres_pool.closeall()
                    logger.info("Postgres connection pool closed.")
                logger.info("Postgres MCP server shutting down...")

    # Create Starlette ASGI application
    starlette_app = Starlette(
        debug=False,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    # Run the Starlette application using Uvicorn
    import uvicorn

    try:
        logger.info("Starting uvicorn server with stateless MCP integration...")
        uvicorn.run(starlette_app, host="0.0.0.0", port=3004, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1


if __name__ == "__main__":
    main()
