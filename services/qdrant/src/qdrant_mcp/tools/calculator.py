"""Calculator tools for MCP server."""

from mcp.server.fastmcp import FastMCP
from typing import Dict, Union
import math


def register_calculator_tools(mcp_server: FastMCP):
    """Register calculator tools with MCP server."""
    
    @mcp_server.tool()
    async def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers together."""
        return a + b
    
    @mcp_server.tool()
    async def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Subtract second number from first number."""
        return a - b
    
    @mcp_server.tool()
    async def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiply two numbers."""
        return a * b
    
    @mcp_server.tool()
    async def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Divide first number by second number."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    @mcp_server.tool()
    async def power(base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]:
        """Calculate base raised to the power of exponent."""
        return math.pow(base, exponent)
    
    @mcp_server.tool()
    async def square_root(number: Union[int, float]) -> float:
        """Calculate square root of a number."""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(number)
    
    @mcp_server.tool()
    async def calculate_expression(expression: str) -> Dict:
        """Safely evaluate a mathematical expression."""
        try:
            # Chỉ cho phép các operations an toàn
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            
            # Evaluate expression một cách an toàn
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "result": None
            }


# Alias để dễ import
calculator_tools = register_calculator_tools
