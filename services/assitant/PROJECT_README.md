# 🚀 Assistant Service - Complete Project

Dự án hoàn chỉnh bao gồm FastAPI backend với MCP integration và React frontend dashboard.

## 📁 Cấu trúc dự án

```
services/assistant/
├── app/                    # Backend FastAPI
│   ├── main.py            # FastAPI application
│   ├── router.py          # API routes
│   ├── auth.py            # Authentication & JWT
│   ├── db.py              # Database operations
│   └── mcp.py             # MCP protocol implementation
├── ui/                    # Frontend React
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   └── types/         # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── test/                  # Tests
├── .env                   # Backend environment
├── start.ps1              # Startup script
└── README.md              # This file
```

## 🚀 Quick Start

### Tự động (Khuyến nghị)
```powershell
# Khởi động cả backend và frontend
.\start.ps1
```

### Thủ công

#### 1. Backend (FastAPI)
```bash
cd services/assistant
pip install fastapi uvicorn python-multipart python-jose[cryptography]
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend (React + Vite)
```bash
cd services/assistant/ui
pnpm install
pnpm run dev
```

## 🌐 Endpoints

### Frontend
- **Dashboard**: http://localhost:3000
- **Login**: Username: `admin`, Password: `admin123`

### Backend API
- **Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **MCP Endpoint**: http://localhost:8000/mcp (requires X-API-Key)

## 🔐 Authentication & API Keys

### User Authentication (JWT)
- Login: `POST /api/auth/login`
- Logout: `POST /api/auth/logout` 
- Get API Key: `GET /api/auth/api-key`

### MCP Protection
- Endpoint: `/mcp`
- Header: `X-API-Key: assistant-mcp-key-2025-super-secure-token`
- Tools: `server_status`, `database_info`, `user_info`

## 📊 Dashboard Features

### 🔑 X-API-Key Display
- Hiển thị API key để copy/paste
- Hướng dẫn cấu hình VS Code MCP
- One-click copy to clipboard

### 👥 User Management  
- Danh sách tất cả users
- Thông tin: ID, username, ngày tạo
- Real-time data từ database

### 🖥️ Server Status
- Trạng thái server (running/stopped)
- Health check status
- Database connection info

### 📈 System Info
- Database type và tables
- User count
- Collections info

## 🔧 VS Code MCP Configuration

Thêm vào `.vscode/mcp.json`:

```json
{
  "servers": {
    "assistant": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-API-Key": "assistant-mcp-key-2025-super-secure-token"
      }
    }
  }
}
```

## 🛠️ Development

### Backend Hot Reload
```bash
cd services/assistant
python -m uvicorn app.main:app --reload
```

### Frontend Hot Reload
```bash
cd services/assistant/ui  
pnpm run dev
```

### Build Production
```bash
cd services/assistant/ui
pnpm run build
```

## 🧪 Testing

### API Protection Tests
```bash
cd services/assistant
python test/test_api_protection.py
```

### Database Check
```bash
cd services/assistant
python test/check_db.py
```

### Manual API Testing
```powershell
# Health check
Invoke-WebRequest -Uri "http://localhost:8000/health"

# Login
$formData = @{username='admin'; password='admin123'}
Invoke-WebRequest -Uri 'http://localhost:8000/api/auth/login' -Method POST -Body $formData

# MCP with API key
Invoke-WebRequest -Uri "http://localhost:8000/mcp" -Method POST -Headers @{"Content-Type"="application/json"; "X-API-Key"="assistant-mcp-key-2025-super-secure-token"} -Body '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

## 🚨 Troubleshooting

### Port Conflicts
```powershell
# Check what's using ports
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# Kill processes if needed
taskkill /PID <process_id> /F
```

### API Key Issues
1. Check `.env` file has `ASSISTANT_API_KEY`
2. Restart backend after changing API key
3. Update VS Code `mcp.json` with new key
4. Restart VS Code to reload MCP config

### Database Issues
```bash
# Check database
python test/check_db.py

# Reset database
rm assistant.db
python -c "from app.db import init_database; init_database()"
```

### Frontend Build Issues
```bash
cd ui
rm -rf node_modules
pnpm install
pnpm run dev
```

## 📝 Features Implemented

### ✅ Backend
- [x] FastAPI with proper routing
- [x] JWT Authentication
- [x] SQLite database with users
- [x] MCP protocol implementation
- [x] API key protection
- [x] CORS middleware
- [x] Request/Response logging
- [x] Error handling
- [x] Health checks

### ✅ Frontend  
- [x] React 18 + TypeScript
- [x] Vite build system
- [x] TailwindCSS styling
- [x] Login/logout flow
- [x] Dashboard with stats
- [x] API key display
- [x] User list management
- [x] Responsive design
- [x] Error handling
- [x] Loading states

### ✅ Integration
- [x] API proxy configuration
- [x] Authentication flow
- [x] Real-time data fetching
- [x] VS Code MCP integration
- [x] Cross-origin requests
- [x] Development workflow

## 🎯 Next Steps

### Potential Enhancements
- [ ] User role management
- [ ] API rate limiting  
- [ ] Real-time WebSocket updates
- [ ] File upload capabilities
- [ ] Advanced MCP tools
- [ ] Metrics and analytics
- [ ] Docker deployment
- [ ] Production optimization

## 🔒 Security

- JWT tokens for authentication
- Password hashing (SHA256)
- API key protection for MCP
- CORS properly configured
- Input validation
- Error sanitization

## 📄 License

MIT License - See LICENSE file for details

---

**🎉 Dự án đã hoàn thành với đầy đủ tính năng theo yêu cầu:**
- ✅ TypeScript + Vite + TailwindCSS
- ✅ Dashboard hiển thị danh sách users
- ✅ Hiển thị và copy X-API-Key
- ✅ Tích hợp với FastAPI backend
- ✅ Authentication flow hoàn chỉnh
- ✅ MCP protocol integration
