"""Tests for PostgreSQL database operations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from postgres_mcp.database import PostgreSQLDatabase


class TestPostgreSQLDatabase:
    """Test PostgreSQL database operations."""
    
    @pytest.fixture
    def db(self):
        """Create database instance for testing."""
        return PostgreSQLDatabase()
    
    @pytest.mark.asyncio
    async def test_connect_success(self, db):
        """Test successful database connection."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await db.connect()
            
            assert db.pool == mock_pool
            mock_create_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, db):
        """Test database connection failure."""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                await db.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, db):
        """Test database disconnection."""
        mock_pool = AsyncMock()
        db.pool = mock_pool
        
        await db.disconnect()
        
        mock_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, db):
        """Test successful query execution."""
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        # Mock query result
        mock_row = MagicMock()
        mock_row.__iter__ = lambda self: iter([('id', 1), ('name', 'test')])
        mock_connection.fetch.return_value = [mock_row]
        
        db.pool = mock_pool
        
        result = await db.execute_query("SELECT * FROM test_table")
        
        assert isinstance(result, list)
        mock_connection.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_query_no_pool(self, db):
        """Test query execution without database connection."""
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db.execute_query("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_execute_query_invalid_sql(self, db):
        """Test query execution with invalid SQL."""
        db.pool = AsyncMock()
        
        with pytest.raises(ValueError, match="Invalid SQL query"):
            await db.execute_query("")
    
    @pytest.mark.asyncio
    async def test_execute_query_non_select(self, db):
        """Test query execution with non-SELECT statement."""
        db.pool = AsyncMock()
        
        with pytest.raises(ValueError, match="Only SELECT, WITH, and EXPLAIN queries are allowed"):
            await db.execute_query("INSERT INTO test VALUES (1)")
    
    @pytest.mark.asyncio
    async def test_execute_command_success(self, db):
        """Test successful command execution."""
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_connection.execute.return_value = "INSERT 0 1"
        
        db.pool = mock_pool
        
        result = await db.execute_command("INSERT INTO test VALUES (1)")
        
        assert result == "INSERT 0 1"
        mock_connection.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, db):
        """Test successful health check."""
        with patch.object(db, 'execute_query') as mock_execute:
            mock_execute.return_value = [{'health_check': 1}]
            
            result = await db.health_check()
            
            assert result is True
            mock_execute.assert_called_once_with("SELECT 1 as health_check")
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, db):
        """Test health check failure."""
        with patch.object(db, 'execute_query') as mock_execute:
            mock_execute.side_effect = Exception("Connection lost")
            
            result = await db.health_check()
            
            assert result is False
