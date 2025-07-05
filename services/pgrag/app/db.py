import asyncpg
from logger import get_logger
from config import DB_CONFIG
import psycopg2

logger = get_logger(__name__)

# Global asyncpg pool
_asyncpg_pool = None


async def get_pool():
    global _asyncpg_pool
    if _asyncpg_pool is None:
        _asyncpg_pool = await asyncpg.create_pool(
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["dbname"],
            min_size=1,
            max_size=10,
        )
    return _asyncpg_pool


async def close_pool():
    global _asyncpg_pool
    if _asyncpg_pool:
        await _asyncpg_pool.close()
        _asyncpg_pool = None


async def execute_query(sql: str, params=None) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            logger.info(f"[execute_query] SQL: {sql} | Params: {params}")
            await conn.execute(sql, *(params or []))
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise


async def fetch_one(sql: str, params=None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            logger.info(f"[fetch_one] SQL: {sql} | Params: {params}")
            row = await conn.fetchrow(sql, *(params or []))
            logger.info(f"[fetch_one] Result: {row}")
            return row
        except Exception as e:
            logger.error(f"Error fetching one: {e}")
            raise


async def fetch_all(sql: str, params=None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            logger.info(f"[fetch_all] SQL: {sql} | Params: {params}")
            rows = await conn.fetch(sql, *(params or []))
            logger.info(f"[fetch_all] Result: {rows}")
            return rows
        except Exception as e:
            logger.error(f"Error fetching all: {e}")
            raise


async def init_tables():
    # Create law_embed table
    sql = """
    CREATE TABLE IF NOT EXISTS law_embed (
        id SERIAL PRIMARY KEY,
        law_id INTEGER NOT NULL,
        embed_vector FLOAT8[] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    await execute_query(sql)
    
    # Create test_embeddings table for general embeddings
    sql_embeddings = """
    CREATE TABLE IF NOT EXISTS test_embeddings (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding vector(384) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    await execute_query(sql_embeddings)


async def install_pgvector_extension():
    sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    await execute_query(sql)


def create_database_if_not_exists():
    """Create a new database if it does not exist using config.py."""
    dbname = DB_CONFIG["dbname"]
    user = DB_CONFIG["user"]
    password = DB_CONFIG["password"]
    host = DB_CONFIG["host"]
    port = DB_CONFIG["port"]
    conn = None
    try:
        conn = psycopg2.connect(
            dbname="postgres", user=user, password=password, host=host, port=port
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE DATABASE {dbname};")
                logger.info(f"Database '{dbname}' created successfully.")
            else:
                logger.info(f"Database '{dbname}' already exists.")
    except Exception as e:
        logger.error(f"Error creating database '{dbname}': {e}")
    finally:
        if conn:
            conn.close()

# Không tự động gọi tạo database khi import - sẽ gọi trong startup event
# create_database_if_not_exists()

# Ví dụ sử dụng:
# conn = get_db_connection(db_pool)
# install_pgvector_extension(conn)
# put_db_connection(db_pool, conn)