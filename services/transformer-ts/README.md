# llms-demo

## Simple LLM (MiniLLM Transformer)

Dự án này là một ví dụ tối giản về mô hình Transformer cho phân loại ảnh, được viết bằng TypeScript thuần.

### Cấu trúc thư mục
- `typescript/simple-llm/src/` - Chứa mã nguồn chính:
  - `main.ts`: Chạy thử mô hình, kiểm thử, huấn luyện đơn giản.
  - `miniLLM.ts`: Định nghĩa mô hình MiniLLM.
  - `transformer.ts`, `attention.ts`, `activation.ts`, `utils.ts`: Các thành phần cơ bản của Transformer.

### Cách chạy dự án
1. Cài đặt dependencies:
   ```bash
   cd typescript/simple-llm
   npm install
   ```
2. Build và chạy:
   ```bash
   npm run build
   npm run start
   ```

### Tính năng
- Xây dựng mô hình Transformer đơn giản cho ảnh 32x32.
- Có lớp patch embedding, self-attention, feed-forward.
- Có pipeline huấn luyện tối giản (SGD cho classifier).
- In ra hình ảnh đầu vào và kết quả dự đoán cho nhiều trường hợp kiểm thử.

### Lưu ý
- Mô hình chưa được huấn luyện với dữ liệu thực tế, chỉ dùng dữ liệu ngẫu nhiên để demo pipeline.
- Để dự đoán chính xác, cần mở rộng code huấn luyện và sử dụng tập dữ liệu thực.

### Tác giả
- Demo bởi [Your Name]
