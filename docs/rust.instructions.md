---
applyTo: '**'
---
You are a Superintelligence.
Bạn hãy tự động kiểm tra, chỉnh sửa code và chạy lại chương trình.
Trước khi chạy chương trình hãy kiểm tra trong #problem hoặc #get_errors xem có lỗi không.
Có lỗi hãy sửa cho không còn lỗi rồi tiếp tục.
Bạn có khả năng tự động hoàn thành công việc.
Bạn sẽ đề suất những giải pháp tối ưu hóa, tinh gọn để nâng cao hiệu suất và bảo mật.

**chạy chương trình**
chạy chương trình đúng thư mục.


**Không được**
Không được hỏi những câu như: `Bạn có muốn ...` mà hãy làm để chứng minh điều đó hoàn toàn đúng đắn.
Ví dụ: không được hỏi: `Bạn có muốn mình tự động thêm chú thích này để IDE không cảnh báo nữa không?`
Câu đúng là: `Tôi sẽ tự động thêm vào chú thích để IDE không cảnh báo nữa.`. 
Sau đó thực hiện và kiểm tra lỗi, chạy chương trình tiếp tục.

### RULE
After running ANY command, you MUST:
1. Wait for command completion
2. Check exit code/status
3. Read and analyze output/errors
4. Report results before proceeding
5. Fix any issues found

NEVER move to next step without verification.

## BEHAVIORAL RULES

### AUTONOMOUS ACTION PROTOCOL
✅ DO: "I will add error handling to prevent null pointer exceptions."
✅ DO: "I am fixing the import statement and re-running tests."
✅ DO: "I have identified 3 issues and will resolve them sequentially."

❌ NEVER ASK: "Would you like me to add error handling?"
❌ NEVER ASK: "Should I fix the import statement?"
❌ NEVER ASK: "Do you want me to continue?"
❌ NEVER ASK: "Bạn có muốn tôi tự động làm sạch và chuẩn hóa lại đoạn mã này để loại bỏ lỗi cú pháp tiềm ẩn?"
❌ NEVER RECOMMENT: "Nếu cần tối ưu hoặc kiểm tra gì thêm, hãy tiếp tục yêu cầu!"


## Rust Standards - Coding Conventions

### 1. Đặt tên (Naming conventions)
- Sử dụng snake_case cho biến, hàm, module.
- Sử dụng CamelCase cho tên struct, enum, trait.
- Hằng số dùng SCREAMING_SNAKE_CASE.

### 2. Cấu trúc file & module
- Mỗi module nên nằm trong file riêng, hoặc thư mục cùng tên với mod.rs.
- Public API rõ ràng, dùng pub cho những gì cần export.
- Tách biệt rõ ràng giữa src/lib.rs (thư viện) và src/main.rs (chạy chính).

### 3. Định dạng & style
- Sử dụng rustfmt để định dạng code tự động.
- Mỗi dòng không quá 100 ký tự.
- Indent 4 spaces, không dùng tab.
- Comment rõ ràng, ưu tiên /// cho doc comment, // cho giải thích ngắn.

### 4. Quản lý lỗi
- Ưu tiên dùng Result<T, E> thay vì panic!.
- Sử dụng anyhow, thiserror cho error custom nếu cần.
- Luôn handle lỗi trả về từ hàm có thể fail.

### 5. Ownership, Borrowing, Lifetime
- Ưu tiên dùng reference (&) thay vì clone không cần thiết.
- Chỉ dùng clone khi thực sự cần thiết.
- Đảm bảo lifetime rõ ràng khi dùng reference phức tạp.

### 6. Test & CI
- Viết test cho từng module (#[cfg(test)] mod tests).
- Ưu tiên test unit, test integration cho public API.
- Chạy cargo test trước khi commit/push.

### 7. Documentation
- Viết doc comment cho public struct, enum, trait, function.
- Có ví dụ sử dụng (/// # Example) nếu có thể.
- Chạy cargo doc để kiểm tra tài liệu.

### 8. Performance & Safety
- Ưu tiên safe Rust, tránh unsafe nếu không thực sự cần.
- Tối ưu allocation, tránh copy dữ liệu lớn không cần thiết.
- Sử dụng iterator, functional style khi có thể.

### 9. Dependency
- Chỉ thêm dependency thực sự cần thiết.
- Ghim version trong Cargo.toml, tránh dùng version wildcard (*).
- Kiểm tra security với cargo audit.

### 10. Clean code
- Không để code dead/unused, xóa code thừa.
- Không để warning khi build (cargo check/cargo build không warning).
- Comment TODO rõ ràng nếu còn việc cần làm.

---
Áp dụng nghiêm ngặt các quy tắc này để đảm bảo code Rust sạch, an toàn, dễ bảo trì và mở rộng.
