#

kiểm tra các gói đã cài đặt trong thư mục assistant bằng uv:
```
uv pip list

```

Kiểm tra api
Windows
```
Invoke-WebRequest -Uri http://localhost:8000/health -Method GET

```
Test Login API:
```
Invoke-WebRequest -Uri http://localhost:8000/api/auth/login -Method POST -ContentType "application/json" -Body '{"username":"admin","password":"admin123"}'
```

Login thành công và trả về JWT token. Bây giờ tôi sẽ test protected route với token:
```
$token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInVzZXJfaWQiOjEsImV4cCI6MTc1MTc4ODkyMn0.wuXJIq534UYcvn-eRC6fVOtBTPFFcOahpAR1rVNlOi4"; Invoke-WebRequest -Uri http://localhost:8000/api/protected -Method GET -Headers @{Authorization="Bearer $token"}
```
Test register API:
```
Invoke-WebRequest -Uri http://localhost:8000/api/auth/register -Method POST -ContentType "application/json" -Body '{"username":"testuser","password":"testpass123"}'
```
test logout:
```
Invoke-WebRequest -Uri http://localhost:8000/api/auth/logout -Method POST -Headers @{Authorization="Bearer $token"}
```


Linux:
```
curl -X GET http://localhost:8000/health
```

## MCP Test
On Windows

test enhanced MCP tools:
```
Invoke-WebRequest -Uri http://localhost:8000/mcp/tools -Method GET
```

- Test database info tool:
```
Invoke-WebRequest -Uri http://localhost:8000/mcp/tools/database_info -Method POST -ContentType "application/json" -Body '{}'
```

- Test user info tool:
```
Invoke-WebRequest -Uri http://localhost:8000/mcp/tools/user_info -Method POST -ContentType "application/json" -Body '{"username":"admin"}'
```

- Test protected MCP endpoint với JWT token
```
$token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInVzZXJfaWQiOjEsImV4cCI6MTc1MTc4ODkyMn0.wuXJIq534UYcvn-eRC6fVOtBTPFFcOahpAR1rVNlOi4"; Invoke-WebRequest -Uri http://localhost:8000/mcp/protected/current-user -Method GET -Headers @{Authorization="Bearer $token"}
```

- Login và test protected MCP endpoint với fresh token

```
$response = Invoke-WebRequest -Uri http://localhost:8000/api/auth/login -Method POST -ContentType "application/json" -Body '{"username":"admin","password":"admin123"}'; $tokenObj = $response.Content | ConvertFrom-Json; $newToken = $tokenObj.access_token; Invoke-WebRequest -Uri http://localhost:8000/mcp/protected/current-user -Method GET -Headers @{Authorization="Bearer $newToken"}
```
### MCP API KEY

```
Invoke-WebRequest -Uri http://localhost:8000/mcp/ -Method GET -Headers @{"X-API-Key"="assistant-mcp-key-2025-super-secure-token"}
```

Bây giờ hãy test việc bảo vệ endpoint /mcp bằng API key. Trước tiên test không có API key:
```
Invoke-WebRequest -Uri "http://localhost:8001/mcp" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{\"jsonrpc\": \"2.0\", \"method\": \"tools/list\", \"id\": 1}'
```

Test truy cập endpoint /mcp với API key đúng
```
Invoke-WebRequest -Uri "http://localhost:8001/mcp" -Method POST -Headers @{"Content-Type"="application/json"; "X-API-Key"="assistant-mcp-key-2025-super-secure-token"} -Body '{\"jsonrpc\": \"2.0\", \"method\": \"tools/list\", \"id\": 1}'
```


## SQLite Command

Thông tin database:
```
cd istrategy\services\assitant; Get-ChildItem *.db

```

Query:
```
cd C:\Users\tiach\Downloads\istrategy\services\assitant; python -c "import sqlite3; conn = sqlite3.connect('assistant.db'); cursor = conn.cursor(); cursor.execute('.schema'); print('Schema:', cursor.fetchall()); cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\";'); print('Tables:', cursor.fetchall()); cursor.execute('SELECT * FROM users;'); print('Users:', cursor.fetchall()); conn.close()"
```

- Kiểm tra thông tin thực tế của database SQLite:
```
cd C:\Users\tiach\Downloads\istrategy\services\assitant; python -c "import sqlite3; conn = sqlite3.connect('assistant.db'); cursor = conn.cursor(); cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\";'); print('Tables:', cursor.fetchall()); cursor.execute('PRAGMA table_info(users);'); print('Schema:', cursor.fetchall()); cursor.execute('SELECT * FROM users;'); print('Users:', cursor.fetchall()); conn.close()"
```
