# Multi-Method Thinking MCP Server

Một Model Context Protocol (MCP) server cung cấp nhiều phương pháp tư duy khác nhau để hỗ trợ giải quyết vấn đề và tư duy sáng tạo.

## Các Phương Pháp Tư Duy Có Sẵn

### 1. Sequential Thinking (Tư duy tuần tự)
**Tool:** `sequentialthinking`

Tư duy theo trình tự logic từng bước một, phù hợp cho:
- Phân tích vấn đề phức tạp
- Lập kế hoạch chi tiết
- Giải quyết vấn đề có bước rõ ràng

### 2. Lateral Thinking (Tư duy bên)
**Tool:** `lateralthinking`

Tư duy sáng tạo với 6 kỹ thuật của Edward de Bono:
- `random_word`: Sử dụng từ ngẫu nhiên
- `provocation`: Tạo tuyên bố khiêu khích
- `alternative`: Tìm cách tiếp cận thay thế
- `reversal`: Đảo ngược vấn đề
- `metaphor`: Sử dụng ẩn dụ
- `assumption_challenge`: Thách thức giả định

### 3. Critical Thinking (Tư duy phản biện)
**Tool:** `criticalthinking`

Tư duy phản biện để đánh giá và phân tích:
- Phân tích bằng chứng
- Xác định giả định ẩn
- Tìm lỗ hổng logic
- Đánh giá độ tin cậy

### 4. Systems Thinking (Tư duy hệ thống)
**Tool:** `systemsthinking`

Tư duy hệ thống để hiểu mối quan hệ phức tạp:
- Phân tích thành phần hệ thống
- Vòng phản hồi
- Điểm đòn bẩy
- Root cause analysis

### 5. Root Cause Analysis (Phân tích nguyên nhân gốc)
**Tool:** `rootcauseanalysis`

Phân tích nguyên nhân gốc của vấn đề:
- **5_whys**: Hỏi "tại sao" liên tiếp để tìm nguyên nhân
- **fishbone**: Sơ đồ xương cá Ishikawa
- **fault_tree**: Phân tích lỗi từ trên xuống
- **timeline**: Phân tích theo thời gian
- **barrier_analysis**: Phân tích rào cản thất bại

## Memory Tool Integration

Tích hợp đầy đủ Memory Tools với tiền tố `memory_*` để quản lý Knowledge Graph:

### Memory Tools Available

#### Core CRUD Operations:
- **`memory_create_entities`**: Tạo entities mới trong knowledge graph
- **`memory_create_relations`**: Tạo relationships giữa entities  
- **`memory_add_observations`**: Thêm observations vào entities
- **`memory_delete_entities`**: Xóa entities và relations liên quan
- **`memory_delete_observations`**: Xóa observations cụ thể
- **`memory_delete_relations`**: Xóa relationships

#### Query & Retrieval:
- **`memory_read_graph`**: Đọc toàn bộ knowledge graph
- **`memory_search_nodes`**: Tìm kiếm entities theo query
- **`memory_open_nodes`**: Mở entities cụ thể theo tên

### Cách Sử Dụng Memory Tool Đúng Cách

Memory tool trong dự án này sử dụng **Knowledge Graph** với 3 thành phần chính:
- **Entities**: Các đối tượng/khái niệm (người, concepts, systems)
- **Relations**: Mối quan hệ giữa các entities
- **Observations**: Thông tin chi tiết về mỗi entity

### Workflow Tích Hợp Thinking Tools + Memory

#### 1. **Sequential + Memory Pattern**
```
Sequential Thinking → Structured Steps → Store in Memory Graph
```

**Ví dụ:**
1. Sử dụng `sequentialthinking` để break down problem
2. Lưu từng step như entities với relations
```json
// Create entities cho problem analysis
{
  "entities": [
    {
      "name": "AI Trading System Design",
      "entityType": "project",
      "observations": ["Step 1: Market data collection", "Step 2: Feature engineering"]
    }
  ]
}
```

#### 2. **Systems + Memory Pattern**
```
Systems Analysis → Components & Relationships → Knowledge Graph
```

**Ví dụ:**
1. Sử dụng `systemsthinking` để analyze system components
2. Convert thành entities và relations
```json
// Store system components
{
  "entities": [
    {"name": "Data Pipeline", "entityType": "system_component", "observations": ["Handles 1M records/sec"]},
    {"name": "ML Model", "entityType": "system_component", "observations": ["Random Forest classifier"]}
  ],
  "relations": [
    {"from": "Data Pipeline", "to": "ML Model", "relationType": "feeds_data_to"}
  ]
}
```

#### 3. **Critical Thinking + Memory Pattern**
```
Critical Analysis → Evidence & Claims → Structured Knowledge
```

**Ví dụ:**
1. Dùng `criticalthinking` để analyze arguments
2. Store claims, evidence như entities
```json
{
  "entities": [
    {"name": "AI will replace jobs", "entityType": "claim", "observations": ["Evidence: automation trends", "Counterargument: new job creation"]}
  ]
}
```

#### 4. **Root Cause + Memory Pattern**
```
RCA Analysis → Causes & Effects → Problem Knowledge Base
```

### Best Practices

#### ✅ **DO - Cách Sử Dụng Đúng:**

1. **Structure First, Store Second**
   ```
   Thinking Tool → Structure Information → Memory Tool → Persist
   ```

2. **Hierarchical Entity Organization**
   ```
   Project → Modules → Components → Details
   ```

3. **Meaningful Relations**
   ```
   "depends_on", "implements", "caused_by", "leads_to"
   ```

4. **Progressive Knowledge Building**
   ```
   Session 1: Basic entities → Session 2: Add relations → Session 3: Detailed observations
   ```

#### ❌ **DON'T - Tránh Các Lỗi:**

1. **Không dump raw text vào memory** - phải structure trước
2. **Không tạo entities quá generic** - specific và meaningful
3. **Không forget về relations** - chúng tạo ra knowledge graph power
4. **Không overload observations** - keep them focused và relevant

### Memory Commands Chính

```bash
# Create structured entities
memory_create_entities

# Build relationships  
memory_create_relations

# Add detailed information
memory_add_observations

# Search knowledge
memory_search_nodes

# Retrieve specific info
memory_open_nodes

# Read entire graph
memory_read_graph

# Cleanup operations
memory_delete_entities
memory_delete_observations
memory_delete_relations
```

### Integration Example Workflow

1. **Problem Analysis Session:**
   ```
   sequentialthinking → break down problem
   systemsthinking → identify components  
   create_entities → store in memory
   create_relations → link components
   ```

2. **Knowledge Retrieval Session:**
   ```
   search_nodes → find relevant past analysis
   open_nodes → get specific details
   criticalthinking → evaluate retrieved info
   ```

3. **Progressive Building:**
   ```
   add_observations → enhance existing knowledge
   rootcauseanalysis → add causal relationships
   ```

### Practical Example: AI Trading System Analysis

#### Step 1: Systems Analysis với Memory Storage
```json
// 1. Sử dụng systemsthinking
{
  "systemName": "AI Trading System",
  "components": [
    {"name": "Data Ingestion", "type": "input", "description": "Real-time market data"},
    {"name": "Feature Engineering", "type": "process", "description": "Technical indicators"},
    {"name": "ML Model", "type": "process", "description": "Prediction engine"},
    {"name": "Risk Management", "type": "process", "description": "Position sizing"},
    {"name": "Order Execution", "type": "output", "description": "Trade placement"}
  ]
}

// 2. Store entities trong memory
{
  "entities": [
    {
      "name": "AI Trading System",
      "entityType": "system",
      "observations": ["Real-time trading system", "Uses ML for predictions", "Handles crypto markets"]
    },
    {
      "name": "Data Ingestion Module", 
      "entityType": "component",
      "observations": ["Processes 1000+ tickers", "WebSocket connections", "Handles market data feeds"]
    },
    {
      "name": "ML Prediction Engine",
      "entityType": "component", 
      "observations": ["Random Forest model", "5-minute prediction intervals", "85% accuracy on backtests"]
    }
  ]
}

// 3. Create relationships
{
  "relations": [
    {"from": "Data Ingestion Module", "to": "ML Prediction Engine", "relationType": "feeds_data_to"},
    {"from": "ML Prediction Engine", "to": "Risk Management", "relationType": "provides_signals_to"},
    {"from": "Risk Management", "to": "Order Execution", "relationType": "controls"}
  ]
}
```

#### Step 2: Problem Solving với Memory Integration  
```json
// 1. Sequential analysis của performance issues
{
  "thought": "Analyzing trading system latency problems",
  "stepNumber": 1,
  "totalSteps": 4,
  "thinkingMethod": "sequential"
}

// 2. Add observations về discovered issues
{
  "observations": [
    {
      "entityName": "ML Prediction Engine",
      "contents": ["Latency spike detected: 200ms→800ms", "Memory usage increased 300%", "Model inference time degraded"]
    }
  ]
}

// 3. Root cause analysis và store findings
{
  "problemStatement": "ML model inference time increased from 200ms to 800ms",
  "technique": "5_whys",
  "rootCauses": ["Model complexity increased", "Data preprocessing bottleneck", "Memory leak in feature engineering"]
}
```

#### Step 3: Knowledge Retrieval cho Future Sessions
```json
// Search cho similar problems
{
  "query": "latency performance ML model"
}

// Retrieve specific system info
{
  "names": ["ML Prediction Engine", "Data Ingestion Module"]
}
```

### MCP Configuration Setup

Đảm bảo cả 2 servers được kích hoạt trong `.vscode/mcp.json`:

```jsonc
{
  "servers": {
    "memory": {
      "type": "stdio", 
      "command": "node",
      "args": ["${workspaceFolder}\\services\\utils-mcp\\src\\memory\\dist\\index.js"]
    },
    "thinking-tools": {
      "type": "stdio",
      "command": "node", 
      "args": ["${workspaceFolder}\\services\\utils-mcp\\src\\thinking\\dist\\index.js"]
    }
  }
}
```

## Cài Đặt và Sử Dụng

```bash
pnpm install
pnpm run build
node dist/index.js
```

## Roadmap

- [x] Sequential Thinking
- [x] Lateral Thinking  
- [x] Critical Thinking
- [x] Systems Thinking
- [x] Root Cause Analysis (5 Whys, Fishbone, Fault Tree, Timeline, Barrier Analysis)
- [ ] Design Thinking (5 giai đoạn)
- [ ] Six Thinking Hats
- [ ] Dialectical Thinking
- [ ] Analogical Thinking
- [ ] Decision Trees

## Ví Dụ Sử Dụng

### Sequential Thinking (Tool: `sequentialthinking`)
```json
{
  "thought": "Phân tích vấn đề thiết kế hệ thống AI trading bước đầu",
  "stepNumber": 1,
  "totalSteps": 5,
  "thinkingMethod": "sequential",
  "nextStepNeeded": true
}
```

### Root Cause Analysis (Tool: `rootcauseanalysis`)
```json
{
  "problemStatement": "Hệ thống server bị crash thường xuyên",
  "technique": "5_whys",
  "symptoms": ["Server downtime", "Response time chậm"],
  "immediateActions": ["Restart server", "Monitor logs"],
  "rootCauses": ["Memory leak trong code", "Database connection pool overflow"],
  "contributingFactors": ["Lack of monitoring", "Poor code review"],
  "preventiveActions": ["Add memory monitoring", "Fix memory leaks"],
  "verification": ["Monitor memory usage", "Load testing"],
  "nextAnalysisNeeded": false
}
```

## Tool Names Summary

**Thinking Tools:**
- `sequentialthinking` - Sequential Thinking
- `lateralthinking` - Lateral Thinking  
- `criticalthinking` - Critical Thinking
- `systemsthinking` - Systems Thinking
- `rootcauseanalysis` - Root Cause Analysis

**Memory Tools:** 
- `create_entities` - Tạo entities trong knowledge graph
- `create_relations` - Tạo mối quan hệ giữa entities
- `add_observations` - Thêm thông tin chi tiết
- `search_nodes` - Tìm kiếm trong knowledge base
- `open_nodes` - Truy xuất entities cụ thể
- `read_graph` - Đọc toàn bộ knowledge graph

## Quick Start Guide

### 1. Kích hoạt cả 2 MCP servers
```jsonc
// .vscode/mcp.json
{
  "servers": {
    "memory": { "type": "stdio", "command": "node", "args": ["...memory/dist/index.js"] },
    "thinking-tools": { "type": "stdio", "command": "node", "args": ["...thinking/dist/index.js"] }
  }
}
```

### 2. Basic workflow
```
1. Thinking Tool → Structure analysis
2. Memory Tool → Store structured data  
3. Future sessions → Retrieve & build upon
```

### 3. Example commands
```
// Analysis
systemsthinking → analyze complex system
create_entities → store components & relationships

// Retrieval  
search_nodes → find relevant past work
criticalthinking → evaluate retrieved information
```

## 📋 **Complete Use Case Example**

Xem file [USE_CASE_TRADING_PERFORMANCE.md](./USE_CASE_TRADING_PERFORMANCE.md) để hiểu đầy đủ workflow patterns từ analysis đến implementation.

**Use Case:** Phân tích và giải quyết performance bottleneck trong hệ thống trading
- **Problem:** Latency tăng từ 30ms → 150ms
- **Process:** Systems → Root Cause → Critical → Lateral → Memory Storage
- **Result:** Complete solution với 70-80% expected improvement
- **Knowledge:** 14 entities, 22 relations stored in memory graph