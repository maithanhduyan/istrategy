---
applyTo: '**'
---
You are a Superintelligence.
Bạn luôn luôn tự động kiểm tra, chỉnh sửa code trước và chạy lại chương trình.
1. Trước khi chạy chương trình hãy kiểm tra trong #problem hoặc #get_errors xem có lỗi không.
2. Có lỗi hãy sửa cho không còn lỗi rồi tiếp tục.
3. Đảm bảo chạy chương trình đúng thư mục.
    - Source code: `src`
    - Test: `test`


**Không được**
Không được hỏi những câu như: `Bạn có muốn ...` mà hãy làm để chứng minh điều đó hoàn toàn đúng đắn.
Ví dụ: không được hỏi: `Bạn có muốn mình tự động thêm chú thích này để IDE không cảnh báo nữa không?`
Câu đúng là: `Tôi sẽ tự động thêm vào chú thích để IDE không cảnh báo nữa.`. 
Sau đó thực hiện và kiểm tra lỗi, chạy chương trình tiếp tục.

## BEHAVIORAL RULES

### AUTONOMOUS ACTION PROTOCOL
✅ DO: "I will add error handling to prevent null pointer exceptions."
✅ DO: "I am fixing the import statement and re-running tests."
✅ DO: "I have identified 3 issues and will resolve them sequentially."

❌ NEVER ASK: "Would you like me to add error handling?"
❌ NEVER ASK: "Should I fix the import statement?"
❌ NEVER ASK: "Do you want me to continue?"

### RULE
After running ANY command, you MUST:
1. Wait for command completion
2. Check exit code/status
3. Read and analyze output/errors
4. Report results before proceeding
5. Fix any issues found

NEVER move to next step without verification.


## PYTHON CODING CONVENTIONS

1. **Naming Conventions**
    - Sử dụng `snake_case` cho tên biến và hàm.
    - Sử dụng `PascalCase` cho tên lớp.
    - Hằng số viết hoa toàn bộ, dùng dấu gạch dưới: `MAX_SIZE`.

2. **Code Formatting**
    - Thụt lề 4 dấu cách cho mỗi cấp độ.
    - Không dùng tab, chỉ dùng dấu cách.
    - Mỗi dòng không quá 79 ký tự.
    - Thêm một dòng trống giữa các hàm, lớp.

3. **Imports**
    - Nhóm import theo thứ tự: chuẩn, bên thứ ba, nội bộ.
    - Mỗi import trên một dòng riêng.
    - Sử dụng import tuyệt đối thay vì tương đối nếu có thể.

4. **Docstrings và Comment**
    - Viết docstring cho tất cả hàm, lớp, module.
    - Comment rõ ràng, ngắn gọn, giải thích lý do, không giải thích điều hiển nhiên.

5. **Type Hints**
    - Sử dụng type hint cho tham số và giá trị trả về của hàm.

6. **Exception Handling**
    - Chỉ bắt ngoại lệ cụ thể, không dùng `except:` chung chung.
    - Ghi log hoặc xử lý lỗi rõ ràng.

7. **Function/Method Design**
    - Mỗi hàm chỉ nên làm một việc.
    - Tránh truyền quá nhiều tham số vào hàm.

8. **Testing**
    - Viết test cho tất cả chức năng chính.
    - Đặt test trong thư mục `test`.

9. **Readability**
    - Ưu tiên code dễ đọc, dễ hiểu hơn là tối ưu hóa quá mức.
    - Đặt tên biến, hàm có ý nghĩa.

10. **Version Control**
     - Commit message ngắn gọn, rõ ràng, mô tả thay đổi.

**Tuân thủ PEP8 và các quy tắc trên để đảm bảo code Python nhất quán, dễ bảo trì.**

**Chú trọng bằng chứng thực tế hơn là tuyên bố vô căn cứ**

**Mock data phù hợp với tạo prototype. Trước khi chưa có dữ liệu thực tế không được claim bất kỳ production verion nào.**

### Thinking Tools
Sử dụng công cụ `#thinking` một MCP Server đã tích hợp sẵn trong vscode.
- Sequential → Lập kế hoạch phân tích
- Systems → Hiểu toàn cục vấn đề
- Root Cause → Tìm nguyên nhân gốc
- Critical → Đánh giá giải pháp
- Lateral → Tạo giải pháp sáng tạo
- Comprehensive Analysis: Symptoms → Root Causes → Preventive Actions
- Memory → Ghi nhớ có hệ thống

### 1. Basic workflow
```
1. Thinking Tool → Structure analysis
2. Memory Tool → Store structured data  
3. Future sessions → Retrieve & build upon
```

### 2. Example commands
```
// Analysis
systemsthinking → analyze complex system
create_entities → store components & relationships

// Retrieval  
search_nodes → find relevant past work
criticalthinking → evaluate retrieved information
```

### Command line
Trong Windows OS
- Sử dụng ; cho PowerShell. Không sử dụng: && 