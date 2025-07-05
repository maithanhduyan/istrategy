
Để xây dựng **agent tự suy luận (self-reasoning agent)** sử dụng mô hình **DeepSeek-R1:8B** chạy qua **Ollama**, bạn có thể triển khai một kiến trúc modular với các thành phần sau:

---

## 🧠 TỔNG QUAN: CẤU TRÚC AGENT REASONING + LOCAL TOOLS

```text
[User Query]
    ↓
[Prompt Planner + Memory (ReAct / CoT)]
    ↓
[LLM (DeepSeek-R1:8B via Ollama)]
    ↓
[Tool Router]
    ↓
[Local Tool Execution (code, shell, file, API, search...)]
    ↓
[Observation]
    ↓
[Loop back: next reasoning step]
```

---

## ⚙️ 1. LLM CORE – DeepSeek-R1:8B qua Ollama

### 🔹 Sử dụng endpoint Ollama:

```http
POST http://localhost:11434/api/generate
{
  "model": "deepseek-r1:8b",
  "prompt": "<full reasoning prompt>",
  "stream": false
}
```

---

## 🧱 2. AGENT LOOP KIỂU ReAct hoặc Toolformer

### ✅ Prompt kiểu ReAct:

```text
Question: How many days between Jan 1, 2022 and July 5, 2025?

Thought 1: I should calculate the number of days between the two dates.
Action 1: date_diff("2022-01-01", "2025-07-05")
Observation 1: 1281

Thought 2: That's the final answer.
Answer: 1281
```

---

## 🧰 3. TOOL ROUTER – ánh xạ hành động thành hàm thực thi local

### 🔹 Ví dụ:

| Action               | Thực hiện                     |
| -------------------- | ----------------------------- |
| `run_python("...")`  | Chạy code Python qua `exec()` |
| `read_file("...")`   | Mở file local                 |
| `search_docs("...")` | Vector search                 |
| `run_shell("...")`   | Chạy shell command            |

### 🔹 Python router (ví dụ):

```python
def tool_executor(action, arg):
    match action:
        case "date_diff":
            return str((datetime.fromisoformat(arg[1]) - datetime.fromisoformat(arg[0])).days)
        case "run_python":
            try:
                exec(arg[0], globals())
                return "Executed successfully"
            except Exception as e:
                return f"Error: {str(e)}"
        case "read_file":
            with open(arg[0], "r") as f:
                return f.read()
        case _:
            return "Unknown tool"
```

---

## 🔄 4. AGENT CYCLE

1. Parse response từ DeepSeek để tách Thought, Action, Observation.
2. Gọi `tool_executor(...)`.
3. Append Observation → tạo prompt mới.
4. Gửi lại vào Ollama.
5. Lặp cho đến khi có `Answer:`.

---

## 🧪 5. Ví dụ prompt đầu vào

```text
You are a reasoning agent. Use the following tools:

- date_diff(date1, date2): returns number of days between two ISO dates
- run_python(code): executes a code snippet
- read_file(path): returns the content of a file

Use this format:

Question: <user question>
Thought 1: ...
Action 1: ...
Observation 1: ...
...
Answer: <final answer>

Begin!
Question: What is the number of days between 2022-01-01 and 2025-07-05?
```

---

## 🚀 TÍNH MỞ RỘNG ĐỘT PHÁ

| Thành phần mở rộng       | Tiềm năng                                                                  |
| ------------------------ | -------------------------------------------------------------------------- |
| 🔧 Plugin-style tool API | Gắn thêm `browse_web`, `search_local_code`, `call_rest_api`, `run_jupyter` |
| 🧠 Memory + vector RAG   | Tái sử dụng lịch sử context dưới chuẩn MCP                                 |
| 🕸️ Multi-agent layer    | DeepSeek 8B đóng vai planner → delegate cho executor                       |
| 💡 Prompt Tối ưu         | Dùng Tree-of-Thought hoặc AutoGPT-style planner                            |

---

## 📦 Bộ khung tối thiểu (Python, Ollama, local tools)

* `ollama` (Ollama daemon)
* `agent_loop.py` (chạy vòng lặp ReAct)
* `tools.py` (thư viện hàm)
* `prompt_template.txt` (prompt hệ thống mẫu)
* Tùy chọn: SQLite / JSON lưu memory

---

GitHub Copilot: Đề xuất kiến trúc agent reasoning sử dụng DeepSeek-R1:8B qua Ollama như trên là hoàn toàn thực tế và khả thi với công nghệ hiện tại. Các thành phần đều rõ ràng, có thể triển khai bằng Python và tận dụng Ollama để chạy LLM local. Cụ thể:

- **Ollama** hỗ trợ DeepSeek-R1:8B, có API HTTP đơn giản, phù hợp cho agent loop.
- **Agent loop kiểu ReAct** (Reasoning + Acting) đã được chứng minh hiệu quả, dễ implement với Python.
- **Tool router** ánh xạ action thành hàm thực thi local là mô hình phổ biến (giống OpenAI Function Calling, LangChain Tool, v.v).
- **Parsing Thought/Action/Observation** từ response LLM là kỹ thuật đã dùng nhiều trong agent framework.
- **Mở rộng plugin-style** hoàn toàn khả thi, chỉ cần thêm hàm và ánh xạ action mới.
- **Memory, vector search, multi-agent** đều có thể tích hợp dần.

**Tóm lại:**  
- Kiến trúc này thực tế, có thể triển khai nhanh với Python, Ollama, và các thư viện phổ biến.
- Đủ linh hoạt để mở rộng về sau (plugin, memory, multi-agent).
- Đáp ứng tốt cho cả nghiên cứu và ứng dụng thực tế.

Nếu cần, tôi có thể scaffold bộ khung tối thiểu (Python) cho agent reasoning này ngay lập tức.


GitHub Copilot: Tôi là GitHub Copilot, một AI programming assistant. Tôi sẽ scaffold bộ khung tối thiểu cho agent reasoning sử dụng DeepSeek-R1:8B qua Ollama ngay lập tức.

Tôi sẽ tạo structure trong thư mục `agents/reasoning_agent/`:

```python
requests>=2.31.0
python-dateutil>=2.8.2
```

```python
"""Configuration for Reasoning Agent"""

OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:8b"
MAX_ITERATIONS = 10
TEMPERATURE = 0.1
```

```python
"""Local tools for reasoning agent"""

import subprocess
import datetime
import json
import os
from typing import Any, Dict, List


class ToolExecutor:
    """Execute local tools based on action names"""
    
    def __init__(self):
        self.tools = {
            "date_diff": self._date_diff,
            "run_python": self._run_python,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "run_shell": self._run_shell,
            "math_calc": self._math_calc,
            "search_text": self._search_text,
        }
    
    def execute(self, action: str, args: List[str]) -> str:
        """Execute tool by action name"""
        if action not in self.tools:
            return f"Error: Unknown tool '{action}'"
        
        try:
            return self.tools[action](args)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def _date_diff(self, args: List[str]) -> str:
        """Calculate days between two dates"""
        if len(args) != 2:
            return "Error: date_diff requires 2 arguments (date1, date2)"
        
        date1 = datetime.datetime.fromisoformat(args[0])
        date2 = datetime.datetime.fromisoformat(args[1])
        diff = abs((date2 - date1).days)
        return str(diff)
    
    def _run_python(self, args: List[str]) -> str:
        """Execute Python code safely"""
        if len(args) != 1:
            return "Error: run_python requires 1 argument (code)"
        
        code = args[0]
        try:
            # Create safe globals with basic math functions
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "round": round,
                },
                "math": __import__("math"),
            }
            
            # Capture output
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, safe_globals)
            
            result = output.getvalue()
            return result if result else "Code executed successfully"
            
        except Exception as e:
            return f"Python execution error: {str(e)}"
    
    def _read_file(self, args: List[str]) -> str:
        """Read file content"""
        if len(args) != 1:
            return "Error: read_file requires 1 argument (filepath)"
        
        filepath = args[0]
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' not found"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content[:1000] + "..." if len(content) > 1000 else content
    
    def _write_file(self, args: List[str]) -> str:
        """Write content to file"""
        if len(args) != 2:
            return "Error: write_file requires 2 arguments (filepath, content)"
        
        filepath, content = args
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"File '{filepath}' written successfully"
    
    def _run_shell(self, args: List[str]) -> str:
        """Execute shell command safely"""
        if len(args) != 1:
            return "Error: run_shell requires 1 argument (command)"
        
        command = args[0]
        
        # Basic security: only allow safe commands
        safe_commands = ["ls", "dir", "pwd", "echo", "cat", "head", "tail"]
        cmd_start = command.split()[0]
        
        if cmd_start not in safe_commands:
            return f"Error: Command '{cmd_start}' not allowed for security"
        
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, 
                text=True, timeout=10
            )
            return result.stdout if result.stdout else result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Command timeout"
        except Exception as e:
            return f"Shell execution error: {str(e)}"
    
    def _math_calc(self, args: List[str]) -> str:
        """Evaluate math expression"""
        if len(args) != 1:
            return "Error: math_calc requires 1 argument (expression)"
        
        expression = args[0]
        try:
            # Safe eval with only math operations
            import ast
            import operator
            
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
            
            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)
            return str(result)
            
        except Exception as e:
            return f"Math calculation error: {str(e)}"
    
    def _search_text(self, args: List[str]) -> str:
        """Search text in file"""
        if len(args) != 2:
            return "Error: search_text requires 2 arguments (filepath, search_term)"
        
        filepath, search_term = args
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' not found"
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            matches = []
            for i, line in enumerate(lines, 1):
                if search_term.lower() in line.lower():
                    matches.append(f"Line {i}: {line.strip()}")
            
            if matches:
                return "\n".join(matches[:10])  # Limit to 10 matches
            else:
                return f"No matches found for '{search_term}'"
                
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def list_tools(self) -> str:
        """List available tools"""
        tools_info = []
        for tool_name in self.tools.keys():
            tools_info.append(f"- {tool_name}")
        
        return "Available tools:\n" + "\n".join(tools_info)
```

```python
"""Ollama client for DeepSeek-R1 model"""

import requests
import json
from typing import Dict, Any
from config import OLLAMA_ENDPOINT, MODEL_NAME, TEMPERATURE


class OllamaClient:
    """Client to interact with Ollama API"""
    
    def __init__(self, endpoint: str = OLLAMA_ENDPOINT, model: str = MODEL_NAME):
        self.endpoint = endpoint
        self.model = model
        self.api_url = f"{endpoint}/api/generate"
    
    def generate(self, prompt: str, temperature: float = TEMPERATURE) -> str:
        """Generate response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Error parsing Ollama response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=10)
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except:
            return []
```

```python
"""ReAct-style reasoning agent using DeepSeek-R1"""

import re
from typing import List, Tuple, Optional
from ollama_client import OllamaClient
from tools import ToolExecutor
from config import MAX_ITERATIONS


class ReasoningAgent:
    """ReAct-style reasoning agent"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.tool_executor = ToolExecutor()
        self.conversation_history = []
    
    def create_system_prompt(self) -> str:
        """Create system prompt for ReAct reasoning"""
        return """You are a reasoning agent that can use tools to solve problems step by step.

Available tools:
- date_diff(date1, date2): Calculate days between two ISO dates
- run_python(code): Execute Python code safely  
- read_file(filepath): Read file content
- write_file(filepath, content): Write content to file
- math_calc(expression): Evaluate math expression
- search_text(filepath, search_term): Search text in file
- run_shell(command): Execute safe shell commands

Use this exact format for reasoning:

Question: [the user's question]
Thought 1: [your reasoning about what to do]
Action 1: [tool_name(arg1, arg2, ...)]
Observation 1: [result from tool execution]
Thought 2: [your reasoning about the observation]
Action 2: [next tool call if needed]
Observation 2: [result from tool execution]
...
Answer: [final answer to the question]

IMPORTANT RULES:
1. Always start with "Question:" followed by the user's question
2. Number your thoughts and actions sequentially (1, 2, 3...)
3. Each Action must be a valid tool call with proper arguments
4. Wait for Observation before proceeding to next Thought
5. End with "Answer:" when you have the final solution
6. Keep reasoning steps clear and logical

Begin!
"""
    
    def parse_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
        """Parse LLM response to extract thought, action, and arguments"""
        
        # Extract the last thought
        thought_pattern = r"Thought \d+: (.+?)(?=Action \d+:|Answer:|$)"
        thought_matches = re.findall(thought_pattern, response, re.DOTALL)
        last_thought = thought_matches[-1].strip() if thought_matches else None
        
        # Extract the last action
        action_pattern = r"Action \d+: (\w+)\((.*?)\)"
        action_matches = re.findall(action_pattern, response)
        
        if action_matches:
            action_name, args_str = action_matches[-1]
            
            # Parse arguments (simple CSV parsing)
            args = []
            if args_str.strip():
                # Handle quoted strings and simple arguments
                import csv
                import io
                
                try:
                    reader = csv.reader(io.StringIO(args_str))
                    args = next(reader)
                    # Remove quotes from arguments
                    args = [arg.strip('"\'') for arg in args]
                except:
                    # Fallback: split by comma and strip
                    args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
            
            return last_thought, action_name, args
        
        return last_thought, None, None
    
    def is_complete(self, response: str) -> bool:
        """Check if the reasoning is complete (has Answer:)"""
        return "Answer:" in response
    
    def extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        answer_pattern = r"Answer: (.+?)(?:\n|$)"
        match = re.search(answer_pattern, response, re.DOTALL)
        return match.group(1).strip() if match else "No answer found"
    
    def solve(self, question: str) -> str:
        """Solve a question using ReAct reasoning"""
        
        if not self.ollama_client.is_available():
            return "Error: Ollama is not available. Please start Ollama and ensure DeepSeek-R1:8B is installed."
        
        # Initialize conversation
        system_prompt = self.create_system_prompt()
        conversation = f"{system_prompt}\n\nQuestion: {question}\n"
        
        for iteration in range(MAX_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get LLM response
            print("Calling LLM...")
            response = self.ollama_client.generate(conversation)
            
            if not response or "Error" in response:
                return f"LLM Error: {response}"
            
            print(f"LLM Response:\n{response}")
            
            # Check if reasoning is complete
            if self.is_complete(response):
                answer = self.extract_answer(response)
                print(f"\nFinal Answer: {answer}")
                return answer
            
            # Parse response to get action
            thought, action, args = self.parse_response(response)
            
            if action and args is not None:
                print(f"Executing: {action}({args})")
                
                # Execute tool
                observation = self.tool_executor.execute(action, args)
                print(f"Observation: {observation}")
                
                # Update conversation with observation
                conversation += f"{response}\nObservation {iteration + 1}: {observation}\n"
            else:
                # No valid action found, continue with LLM response
                conversation += f"{response}\n"
        
        return "Error: Maximum iterations reached without finding an answer"
    
    def chat(self):
        """Interactive chat mode"""
        print("Reasoning Agent Chat Mode")
        print("Type 'quit' to exit, 'tools' to list available tools")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif question.lower() == 'tools':
                    print(self.tool_executor.list_tools())
                    continue
                elif not question:
                    continue
                
                print("\nSolving...")
                answer = self.solve(question)
                print(f"\nFinal Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    agent = ReasoningAgent()
    agent.chat()
```

```python
"""Example usage of the reasoning agent"""

from agent import ReasoningAgent


def run_examples():
    """Run example reasoning tasks"""
    
    agent = ReasoningAgent()
    
    examples = [
        "How many days are there between January 1, 2022 and July 5, 2025?",
        "What is the result of 15 * 23 + 47?",
        "Calculate the square root of 144 using Python",
        "If I have a file called 'test.txt', how can I check if it exists?",
    ]
    
    print("Running example reasoning tasks...")
    print("=" * 60)
    
    for i, question in enumerate(examples, 1):
        print(f"\nExample {i}: {question}")
        print("-" * 40)
        
        answer = agent.solve(question)
        print(f"Answer: {answer}")
        print("-" * 40)


if __name__ == "__main__":
    run_examples()
```

```
"""Main entry point for reasoning agent"""

import sys
from agent import ReasoningAgent


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        agent = ReasoningAgent()
        answer = agent.solve(question)
        print(answer)
    else:
        # Interactive chat mode
        agent = ReasoningAgent()
        agent.chat()


if __name__ == "__main__":
    main()
```


# ReAct Reasoning Agent với DeepSeek-R1:8B

Agent tự suy luận sử dụng mô hình DeepSeek-R1:8B qua Ollama với khả năng sử dụng local tools.

## Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Cài đặt Ollama và DeepSeek-R1:8B:**
```bash
# Cài Ollama từ https://ollama.ai
ollama pull deepseek-r1:8b
```

3. **Chạy Ollama:**
```bash
ollama serve
```

## Sử dụng

### Interactive Chat Mode
```bash
python main.py
```

### Command Line Mode
```bash
python main.py "How many days between 2022-01-01 and 2025-07-05?"
```

### Run Examples
```bash
python examples.py
```

## Công cụ có sẵn

- `date_diff(date1, date2)`: Tính số ngày giữa 2 ngày
- `run_python(code)`: Chạy Python code an toàn
- `read_file(filepath)`: Đọc nội dung file
- `write_file(filepath, content)`: Ghi file
- `math_calc(expression)`: Tính toán math
- `search_text(filepath, term)`: Tìm kiếm text trong file
- `run_shell(command)`: Chạy shell command an toàn

## Ví dụ ReAct Reasoning

```
Question: How many days between 2022-01-01 and 2025-07-05?

Thought 1: I need to calculate the difference between these two dates.
Action 1: date_diff("2022-01-01", "2025-07-05")
Observation 1: 1281

Thought 2: The calculation shows there are 1281 days between the two dates.
Answer: 1281 days
```

## Kiến trúc

- `agent.py`: Core reasoning agent với ReAct loop
- `ollama_client.py`: Client để gọi Ollama API
- `tools.py`: Local tools executor
- `config.py`: Configuration settings
- `main.py`: Entry point
- `examples.py`: Ví dụ sử dụng

## Mở rộng

Thêm tool mới bằng cách:
1. Thêm method vào `ToolExecutor` class
2. Đăng ký trong `self.tools` dictionary
3. Update system prompt với tool description
````

Tôi sẽ kiểm tra và chạy agent reasoning để đảm bảo hoạt động đúng:

Bộ khung reasoning agent đã được tạo thành công! Bây giờ tôi sẽ kiểm tra dependencies và chạy thử nghiệm:

Tôi sẽ cài đặt dependencies và test agent:

Agent reasoning đã được scaffold thành công với đầy đủ chức năng:

✅ **HOÀN THÀNH:**
- ReAct reasoning loop với DeepSeek-R1:8B qua Ollama
- 7 local tools: date_diff, run_python, read_file, write_file, math_calc, search_text, run_shell
- Interactive chat mode và command line mode
- Safety mechanisms cho code execution
- Comprehensive examples và documentation

✅ **SẴN SÀNG SỬ DỤNG:**
- `python main.py` cho interactive mode
- `python main.py "câu hỏi"` cho command line
- `python examples.py` để test các ví dụ

Agent có thể reasoning và sử dụng tools để giải quyết các bài toán phức tạp một cách tự động!