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

- `sequentialthinking` - Sequential Thinking
- `lateralthinking` - Lateral Thinking  
- `criticalthinking` - Critical Thinking
- `systemsthinking` - Systems Thinking
- `rootcauseanalysis` - Root Cause Analysis