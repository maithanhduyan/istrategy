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
        """Calculate days between two dates with precision"""
        if len(args) != 2:
            return "Error: date_diff requires 2 arguments (date1, date2)"

        try:
            # Parse dates more precisely
            date1_str, date2_str = args[0], args[1]

            # Handle different date formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    date1 = datetime.datetime.strptime(date1_str, fmt)
                    date2 = datetime.datetime.strptime(date2_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                # Fallback to ISO format
                date1 = datetime.datetime.fromisoformat(date1_str)
                date2 = datetime.datetime.fromisoformat(date2_str)

            # Calculate difference with proper handling
            diff = (date2 - date1).days
            return str(abs(diff))

        except ValueError as e:
            return f"Error: Invalid date format - {str(e)}"
        except Exception as e:
            return f"Error calculating date difference: {str(e)}"

    def _run_python(self, args: List[str]) -> str:
        """Execute Python code safely"""
        if len(args) != 1:
            return "Error: run_python requires 1 argument (code)"

        code = args[0]
        try:
            # Create safe globals with basic math functions
            import math

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
                    "__import__": __import__,  # Allow imports
                },
                "math": math,  # Pre-import math module
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

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return content[:1000] + "..." if len(content) > 1000 else content

    def _write_file(self, args: List[str]) -> str:
        """Write content to file"""
        if len(args) != 2:
            return "Error: write_file requires 2 arguments (filepath, content)"

        filepath, content = args
        with open(filepath, "w", encoding="utf-8") as f:
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
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.stdout else result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Command timeout"
        except Exception as e:
            return f"Shell execution error: {str(e)}"

    def _math_calc(self, args: List[str]) -> str:
        """Evaluate math expression with function support"""
        if len(args) != 1:
            return "Error: math_calc requires 1 argument (expression)"

        expression = args[0]
        try:
            # Safe eval with math operations and functions
            import ast
            import operator
            import math

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            # Supported functions
            functions = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "ln": math.log,
                "abs": abs,
                "round": round,
                "floor": math.floor,
                "ceil": math.ceil,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }

            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Name):
                    # Handle constants like pi, e
                    if node.id in functions:
                        return functions[node.id]
                    else:
                        raise ValueError(f"Unknown variable: {node.id}")
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](
                        eval_expr(node.left), eval_expr(node.right)
                    )
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                elif isinstance(node, ast.Call):
                    # Handle function calls
                    if isinstance(node.func, ast.Name) and node.func.id in functions:
                        func = functions[node.func.id]
                        args = [eval_expr(arg) for arg in node.args]
                        return func(*args)
                    else:
                        func_name = (
                            node.func.id
                            if isinstance(node.func, ast.Name)
                            else "complex"
                        )
                        raise ValueError(f"Unknown function: {func_name}")
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")

            tree = ast.parse(expression, mode="eval")
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
            with open(filepath, "r", encoding="utf-8") as f:
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
