-- PostgreSQL Initialization Script
-- Tạo database và user riêng cho ứng dụng Polymind

-- Tạo user cho ứng dụng (không phải superuser)
CREATE USER polymind_user WITH 
    PASSWORD 'polymind_secure_2025!'
    CREATEDB 
    LOGIN;

-- Tạo database cho ứng dụng
CREATE DATABASE polymind_db 
    WITH 
    OWNER = polymind_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TEMPLATE = template0;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE polymind_db TO polymind_user;

-- Connect to the new database and set up schema
\c polymind_db polymind_user

-- Tạo schema cho các bảng chính
CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION polymind_user;
CREATE SCHEMA IF NOT EXISTS chat AUTHORIZATION polymind_user;
CREATE SCHEMA IF NOT EXISTS vector AUTHORIZATION polymind_user;

-- Set default search path
ALTER USER polymind_user SET search_path = app, chat, vector, public;

-- Tạo extension cần thiết cho vector operations (nếu cần)
-- CREATE EXTENSION IF NOT EXISTS vector;  -- Uncomment nếu sử dụng pgvector

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
    RAISE NOTICE 'Database: polymind_db';
    RAISE NOTICE 'User: polymind_user';
    RAISE NOTICE 'Schemas: app, chat, vector';
END $$;
