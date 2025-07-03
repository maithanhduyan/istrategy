"""Tests for PostgreSQL MCP Server configuration."""

import pytest
import os
from postgres_mcp.config import PostgreSQLConfig, MCPConfig, Settings


class TestPostgreSQLConfig:
    """Test PostgreSQL configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PostgreSQLConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "postgres"
        assert config.username == "postgres"
        assert config.password == ""
        assert config.ssl_mode == "prefer"
        assert config.max_connections == 10
    
    def test_environment_variables(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("POSTGRES_HOST", "testhost")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_DATABASE", "testdb")
        monkeypatch.setenv("POSTGRES_USERNAME", "testuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
        
        config = PostgreSQLConfig()
        assert config.host == "testhost"
        assert config.port == 5433
        assert config.database == "testdb"
        assert config.username == "testuser"
        assert config.password == "testpass"


class TestMCPConfig:
    """Test MCP server configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MCPConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "info"
        assert config.debug is False
    
    def test_environment_variables(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("MCP_PORT", "9000")
        monkeypatch.setenv("MCP_LOG_LEVEL", "debug")
        monkeypatch.setenv("MCP_DEBUG", "true")
        
        config = MCPConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.log_level == "debug"
        assert config.debug is True


class TestSettings:
    """Test application settings."""
    
    def test_database_url(self):
        """Test database URL generation."""
        settings = Settings()
        settings.postgres.host = "localhost"
        settings.postgres.port = 5432
        settings.postgres.database = "testdb"
        settings.postgres.username = "testuser"
        settings.postgres.password = "testpass"
        
        expected_url = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert settings.database_url == expected_url
    
    def test_database_url_no_password(self):
        """Test database URL generation without password."""
        settings = Settings()
        settings.postgres.host = "localhost"
        settings.postgres.port = 5432
        settings.postgres.database = "testdb"
        settings.postgres.username = "testuser"
        settings.postgres.password = ""
        
        expected_url = "postgresql://testuser@localhost:5432/testdb"
        assert settings.database_url == expected_url
    
    def test_async_database_url(self):
        """Test async database URL generation."""
        settings = Settings()
        settings.postgres.host = "localhost"
        settings.postgres.port = 5432
        settings.postgres.database = "testdb"
        settings.postgres.username = "testuser"
        settings.postgres.password = "testpass"
        
        expected_url = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert settings.async_database_url == expected_url
