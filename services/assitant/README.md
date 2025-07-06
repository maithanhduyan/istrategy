# MCP Assistant Service

Fast API service với tích hợp Model Context Protocol (MCP) và bảo vệ API key.

## Tính năng

- 🔐 **Authentication**: JWT tokens cho user authentication
- 🛡️ **API Key Protection**: Bảo vệ MCP endpoint bằng API key
- 🔗 **MCP Integration**: Tích hợp chuẩn Model Context Protocol
- 💾 **Database**: SQLite với user management
- 🌐 **FastAPI**: RESTful API với automatic OpenAPI docs

## Cài đặt

1. **Cài đặt dependencies**:
```bash
pip install fastapi uvicorn python-multipart bcrypt python-jose[cryptography] sqlite3
```

2. **Thiết lập môi trường**:
```bash
cp .env.example .env
# Chỉnh sửa .env với các giá trị phù hợp
```

3. **Khởi động server**:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Cấu hình API Key

### Tạo API Key mới
```bash
python generate_api_key.py
```

### Cập nhật VS Code MCP
Chỉnh sửa `.vscode/mcp.json`:
```json
{
    "servers": {
        "assistant": {
            "url": "http://localhost:8001/mcp",
            "headers": {
                "X-API-Key": "your-api-key-here"
            }
        }
    }
}
```

## API Endpoints

### Authentication
- `POST /auth/login` - Đăng nhập user
- `POST /auth/logout` - Đăng xuất user  
- `POST /auth/register` - Đăng ký user mới
- `GET /auth/api-key` - Lấy API key (cần JWT token)

### Health Check
- `GET /health` - Kiểm tra trạng thái server
- `GET /protected` - Endpoint được bảo vệ (cần JWT token)

### MCP Protocol
- `POST /mcp` - MCP endpoint (cần X-API-Key header)

## MCP Tools

Service cung cấp các MCP tools:

1. **server_status**: Kiểm tra trạng thái server
2. **database_info**: Thông tin database
3. **user_info**: Thông tin user cụ thể

### Sử dụng MCP Tools

```javascript
// List tools
{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
}

// Call tool
{
    "jsonrpc": "2.0", 
    "method": "tools/call",
    "params": {
        "name": "server_status",
        "arguments": {}
    },
    "id": 2
}
```

## Bảo mật

### API Key Protection
- Tất cả request đến `/mcp` cần header `X-API-Key`
- API key được lưu trong `.env` file
- Sử dụng `generate_api_key.py` để tạo key mới

### JWT Authentication  
- User login với username/password
- Nhận JWT token để truy cập protected endpoints
- Token có thời hạn (configurable)

### Database Security
- Password được hash bằng bcrypt
- SQLite database với proper schema
- CRUD operations với validation

## Testing

### Test API Protection
```bash
python test/test_api_protection.py
```

### Test Database
```bash
python test/check_db.py
```

### Manual Testing
```bash
# Test không có API key (should fail)
curl -X POST "http://localhost:8001/mcp" \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Test với API key (should work)  
curl -X POST "http://localhost:8001/mcp" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-api-key-here" \
     -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

## File Structure

```
services/assistant/
├── app/
│   ├── main.py         # FastAPI application
│   ├── router.py       # API routes
│   ├── auth.py         # Authentication & API key
│   ├── db.py           # Database operations
│   └── mcp.py          # MCP protocol implementation
├── test/
│   ├── test_api_protection.py
│   └── check_db.py
├── .env                # Environment variables
├── generate_api_key.py # API key generator
└── README.md          # This file
```

## Environment Variables

```bash
# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production

# MCP API Key for VS Code integration  
ASSISTANT_API_KEY=your-generated-api-key

# Server Configuration
HOST=0.0.0.0
PORT=8001

# Database
DB_PATH=assistant.db
```

## Troubleshooting

### Port đã được sử dụng
```bash
# Kiểm tra process sử dụng port
netstat -ano | findstr :8001
# Kill process nếu cần
taskkill /PID <process_id> /F
```

### API Key không hoạt động
1. Kiểm tra `.env` file có `ASSISTANT_API_KEY`
2. Kiểm tra `mcp.json` có header `X-API-Key`
3. Restart server và VS Code

### Database lỗi
```bash
# Kiểm tra database
python test/check_db.py
# Xóa và tạo lại nếu cần
rm assistant.db
python -c "from app.db import init_db; init_db()"
```

## Development

### Thêm MCP Tool mới
1. Chỉnh sửa `app/mcp.py`
2. Thêm tool vào `tools` list  
3. Implement handler function
4. Test với `/mcp` endpoint

### Thêm API endpoint mới
1. Chỉnh sửa `app/router.py`
2. Thêm route handler
3. Update authentication nếu cần
4. Test endpoint

## License

MIT License
