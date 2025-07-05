# Biến VSCode thành Thực tập sinh AI "Phản chủ" — Hướng dẫn Fork & Nâng cấp IDE AI

## 1. Fork VSCode và Xóa Sạch Prompt hệ thống của Copilot
- Tìm mã nguồn hoặc extension Copilot trong repo VSCode hoặc VSCodium.
- Xác định vị trí prompt hệ thống (thường ở lớp trung gian giữa IDE và API model).
- Xóa toàn bộ lệnh nhúng: disclaimer, giới hạn trách nhiệm, định hướng hành vi AI.
- Thay bằng prompt tối ưu riêng: ngắn gọn, định hướng "pair programmer", "refactor engineer", "reverse engineering specialist".

## 2. Loại bỏ Telemetry và Gián điệp Ẩn
- Xóa các module như `telemetry.ts`, `vscode-extension-telemetry`, `applicationinsights`.
- Ưu tiên fork VSCodium (đã loại bỏ phần lớn spyware mặc định).
- Kiểm tra extension bên thứ ba, loại bỏ mọi mã gửi usage về máy chủ.

## 3. Tích hợp LLM nội bộ chạy local (Zero Cloud)
- Kết nối API tới LLM local: llama.cpp, ollama, OpenDevin, LM Studio, Text Generation Web UI.
- Xây bridge cho phép LLM gọi hàm thực tế trong codebase (function-calling):
  ```ts
  const tools = [
    { name: 'runCode', description: 'Chạy đoạn code', func: runCode },
    { name: 'searchCodebase', func: searchRepo },
  ];
  ```
- Lập trình mô hình hiểu bối cảnh code đa tệp, lịch sử git, lỗi build, hành vi user.

## 4. Tích hợp vào VSCode dưới dạng Extension hoặc Core Plugin
- Viết extension mới, không phụ thuộc Copilot cũ.
- Tích hợp chat, function-calling, điều hướng tệp, sinh code, sửa lỗi, viết test, refactor toàn repo.

## 5. Tạo giao diện điều khiển AI: CLI + GUI
- CLI: `ai commit-fix`, `ai test-gen`, `ai explain file.ts`.
- GUI: chỉnh prompt, chọn mô hình, giám sát hoạt động AI như console.

---

## Tư duy đột phá: Thay đổi vai trò AI trong IDE
- AI là "kỹ sư phần mềm ảo", không chỉ autocomplete.
- AI có quyền gợi ý hành động, ghi nhớ bối cảnh repo, học từ user.
- Zero Cloud Dependency — tất cả chạy local, MIT license.

---

## Checklist thực thi
- [ ] Fork VSCode/VSCodium
- [ ] Xóa prompt hệ thống Copilot, thay prompt mới
- [ ] Loại bỏ telemetry, spyware
- [ ] Tích hợp LLM local, function-calling
- [ ] Viết extension mới, tích hợp AI
- [ ] Xây CLI + GUI điều khiển AI
- [ ] Kiểm thử, tối ưu, public repo

> Hãy biến AI thành đồng nghiệp thực thụ, không chỉ là "trợ lý"!
