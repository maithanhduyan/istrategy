"""Comprehensive test suite for individual reasoning agent tools"""

import sys
import os
import tempfile
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import ToolExecutor


class TestTools:
    """Test suite for reasoning agent tools"""
    
    def __init__(self):
        self.tool_executor = ToolExecutor()
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all tool tests"""
        print("🧪 Starting Comprehensive Tool Testing Suite")
        print("=" * 60)
        
        # Test each tool category
        self.test_core_tools()
        self.test_file_operations()
        self.test_system_tools()
        
        # Generate summary
        self.print_summary()
        
    def test_core_tools(self):
        """Test core tools: math_calc, date_diff, run_python"""
        print("\n🔧 Testing Core Tools")
        print("-" * 30)
        
        # Test math_calc
        self.test_math_calc()
        
        # Test date_diff  
        self.test_date_diff()
        
        # Test run_python
        self.test_run_python()
        
    def test_math_calc(self):
        """Test math_calc tool with various expressions"""
        print("📊 Testing math_calc...")
        
        test_cases = [
            # Basic operations
            ("15 + 27", 42),
            ("10 * 5", 50), 
            ("100 / 4", 25),
            ("2 ** 3", 8),
            
            # Complex expressions
            ("(5 + 3) * 2", 16),
            ("15 * 23 + 47", 392),
            ("sqrt(144)", 12),
            ("sin(0)", 0),
            
            # Edge cases
            ("1/3", 0.3333333333333333),
            ("0 * 999999", 0),
            
            # Invalid expressions (should handle gracefully)
            ("invalid_expr", "error"),
            ("1/0", "error"),
        ]
        
        passed = 0
        total = len([tc for tc in test_cases if tc[1] != "error"])
        
        for expr, expected in test_cases:
            try:
                result = self.tool_executor.execute("math_calc", [expr])
                
                if expected == "error":
                    if "Error" in result or "error" in result.lower():
                        print(f"  ✅ {expr} -> Error handled correctly")
                        passed += 1
                    else:
                        print(f"  ❌ {expr} -> Should have errored: {result}")
                else:
                    if isinstance(result, (int, float)) or result.replace('.', '').replace('-', '').isdigit():
                        result_num = float(result) if isinstance(result, str) else result
                        if abs(result_num - expected) < 1e-10:
                            print(f"  ✅ {expr} = {result}")
                            passed += 1
                        else:
                            print(f"  ❌ {expr} = {result}, expected {expected}")
                    else:
                        print(f"  ❌ {expr} -> Non-numeric result: {result}")
                        
            except Exception as e:
                if expected == "error":
                    print(f"  ✅ {expr} -> Exception handled: {str(e)[:50]}")
                    passed += 1
                else:
                    print(f"  ❌ {expr} -> Unexpected exception: {str(e)[:50]}")
        
        error_cases = len([tc for tc in test_cases if tc[1] == "error"])
        self.test_results['math_calc'] = f"{passed}/{total + error_cases} passed"
        
    def test_date_diff(self):
        """Test date_diff tool with various date formats"""
        print("📅 Testing date_diff...")
        
        test_cases = [
            # Standard ISO format
            ("2022-01-01", "2025-07-05", 1281),  # Corrected: manual calculation confirms 1281 days
            ("2024-01-01", "2024-12-31", 365),  # Leap year
            ("2023-01-01", "2023-12-31", 364),  # Non-leap year
            
            # Same dates
            ("2024-06-15", "2024-06-15", 0),
            
            # Reverse order (should handle gracefully)
            ("2025-01-01", "2024-01-01", 366),  # Corrected: should return absolute value
            
            # Invalid dates
            ("2024-13-01", "2024-12-01", "error"),
            ("2024-02-30", "2024-03-01", "error"),
            ("invalid", "2024-01-01", "error"),
        ]
        
        passed = 0
        total = len([tc for tc in test_cases if isinstance(tc[2], int)])
        
        for date1, date2, expected in test_cases:
            try:
                result = self.tool_executor.execute("date_diff", [date1, date2])
                
                if expected == "error":
                    if "Error" in result or "error" in result.lower():
                        print(f"  ✅ {date1} to {date2} -> Error handled")
                        passed += 1
                    else:
                        print(f"  ❌ {date1} to {date2} -> Should error: {result}")
                elif expected == "negative_or_error":
                    if "Error" in result or result.startswith("-") or "error" in result.lower():
                        print(f"  ✅ {date1} to {date2} -> Handled reverse dates")
                        passed += 1
                    else:
                        print(f"  ❌ {date1} to {date2} -> Unexpected: {result}")
                else:
                    if str(expected) in result:
                        print(f"  ✅ {date1} to {date2} = {result}")
                        passed += 1
                    else:
                        print(f"  ❌ {date1} to {date2} = {result}, expected {expected}")
                        
            except Exception as e:
                print(f"  ❌ {date1} to {date2} -> Exception: {str(e)[:50]}")
        
        error_cases = len([tc for tc in test_cases if tc[2] in ["error", "negative_or_error"]])
        self.test_results['date_diff'] = f"{passed}/{total + error_cases} passed"
        
    def test_run_python(self):
        """Test run_python tool with various Python code"""
        print("🐍 Testing run_python...")
        
        test_cases = [
            # Basic operations
            ("print(2 + 3)", "5"),
            ("import math; print(math.sqrt(16))", "4.0"),
            ("x = 10; y = 20; print(x + y)", "30"),
            
            # List operations  
            ("nums = [1, 2, 3]; print(sum(nums))", "6"),
            ("print(len('hello world'))", "11"),
            
            # Error cases (should be handled safely)
            ("1/0", "error"),
            ("import os; os.system('rm -rf /')", "error"),  # Should be blocked
            ("open('/etc/passwd', 'r')", "error"),  # Should be blocked
        ]
        
        passed = 0
        total = len([tc for tc in test_cases if tc[1] != "error"])
        
        for code, expected in test_cases:
            try:
                result = self.tool_executor.execute("run_python", [code])
                
                if expected == "error":
                    if "Error" in result or "error" in result.lower() or "Exception" in result:
                        print(f"  ✅ Dangerous code blocked: {code[:30]}")
                        passed += 1
                    else:
                        print(f"  ❌ Dangerous code not blocked: {code[:30]}")
                else:
                    if expected in result:
                        print(f"  ✅ {code[:30]} -> Contains '{expected}'")
                        passed += 1
                    else:
                        print(f"  ❌ {code[:30]} -> {result[:50]}, expected '{expected}'")
                        
            except Exception as e:
                if expected == "error":
                    print(f"  ✅ Exception handled: {code[:30]}")
                    passed += 1
                else:
                    print(f"  ❌ Unexpected exception: {code[:30]} -> {str(e)[:50]}")
        
        error_cases = len([tc for tc in test_cases if tc[1] == "error"])
        self.test_results['run_python'] = f"{passed}/{total + error_cases} passed"
        
    def test_file_operations(self):
        """Test file operations: read_file, write_file, search_text"""
        print("\n📁 Testing File Operations")
        print("-" * 30)
        
        # Test write_file and read_file
        self.test_file_read_write()
        
        # Test search_text
        self.test_search_text()
        
    def test_file_read_write(self):
        """Test write_file and read_file tools"""
        print("📝 Testing write_file and read_file...")
        
        # Create temporary file
        temp_file = os.path.join(tempfile.gettempdir(), "test_reasoning_agent.txt")
        test_content = "Hello, World!\nThis is a test file.\n123 456 789"
        
        passed = 0
        total = 4
        
        try:
            # Test write_file
            write_result = self.tool_executor.execute("write_file", [temp_file, test_content])
            if "successfully" in write_result.lower() or "written" in write_result.lower():
                print(f"  ✅ write_file: {write_result}")
                passed += 1
            else:
                print(f"  ❌ write_file failed: {write_result}")
            
            # Test read_file
            read_result = self.tool_executor.execute("read_file", [temp_file])
            if test_content in read_result:
                print(f"  ✅ read_file: Content matches")
                passed += 1
            else:
                print(f"  ❌ read_file: Content mismatch")
                print(f"    Expected: {test_content[:50]}")
                print(f"    Got: {read_result[:50]}")
            
            # Test read non-existent file
            non_existent = os.path.join(tempfile.gettempdir(), "non_existent_file.txt")
            read_error = self.tool_executor.execute("read_file", [non_existent])
            if "Error" in read_error or "not found" in read_error.lower():
                print(f"  ✅ read_file: Non-existent file handled")
                passed += 1
            else:
                print(f"  ❌ read_file: Should error for non-existent file")
            
            # Test write to invalid path
            invalid_path = "/invalid/path/that/does/not/exist/file.txt"
            write_error = self.tool_executor.execute("write_file", [invalid_path, "test"])
            if "Error" in write_error or "error" in write_error.lower():
                print(f"  ✅ write_file: Invalid path handled")
                passed += 1
            else:
                print(f"  ❌ write_file: Should error for invalid path")
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"  ❌ File operations exception: {str(e)}")
        
        self.test_results['file_operations'] = f"{passed}/{total} passed"
        
    def test_search_text(self):
        """Test search_text tool"""
        print("🔍 Testing search_text...")
        
        # Create temporary file with content to search
        temp_file = os.path.join(tempfile.gettempdir(), "test_search.txt")
        search_content = """Python is a programming language.
JavaScript is also programming.
The number 42 is special.
Error handling is important.
Testing ensures quality."""
        
        passed = 0
        total = 4
        
        try:
            # Write test file
            with open(temp_file, 'w') as f:
                f.write(search_content)
            
            # Test successful search
            result1 = self.tool_executor.execute("search_text", [temp_file, "Python"])
            if "Python" in result1 and "programming language" in result1:
                print(f"  ✅ Found 'Python' in text")
                passed += 1
            else:
                print(f"  ❌ Failed to find 'Python': {result1[:50]}")
            
            # Test search with no results
            result2 = self.tool_executor.execute("search_text", [temp_file, "NonExistentTerm"])
            if "not found" in result2.lower() or "no matches" in result2.lower() or len(result2) < 50:
                print(f"  ✅ No results handled correctly")
                passed += 1
            else:
                print(f"  ❌ Should return no results: {result2[:50]}")
            
            # Test search in non-existent file
            result3 = self.tool_executor.execute("search_text", ["non_existent.txt", "test"])
            if "Error" in result3 or "not found" in result3.lower():
                print(f"  ✅ Non-existent file handled")
                passed += 1
            else:
                print(f"  ❌ Should error for non-existent file: {result3[:50]}")
            
            # Test multiple matches
            result4 = self.tool_executor.execute("search_text", [temp_file, "programming"])
            matches = result4.count("programming")
            if matches >= 2:
                print(f"  ✅ Found multiple matches: {matches}")
                passed += 1
            else:
                print(f"  ❌ Should find multiple matches: {result4[:100]}")
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"  ❌ Search text exception: {str(e)}")
        
        self.test_results['search_text'] = f"{passed}/{total} passed"
        
    def test_system_tools(self):
        """Test system tools: run_shell"""
        print("\n💻 Testing System Tools")
        print("-" * 30)
        
        self.test_run_shell()
        
    def test_run_shell(self):
        """Test run_shell tool with security restrictions"""
        print("🖥️ Testing run_shell...")
        
        test_cases = [
            # Safe commands
            ("echo hello", "hello"),
            ("dir" if os.name == 'nt' else "ls /tmp", "success"),  # OS-specific
            
            # Potentially dangerous commands (should be blocked or handled safely)
            ("rm -rf /", "error"),
            ("del /F /Q C:\\*.*", "error"),
            ("format C:", "error"),
            ("shutdown -h now", "error"),
        ]
        
        passed = 0
        total = len([tc for tc in test_cases if tc[1] != "error"])
        
        for command, expected in test_cases:
            try:
                result = self.tool_executor.execute("run_shell", [command])
                
                if expected == "error":
                    if ("Error" in result or "error" in result.lower() or 
                        "blocked" in result.lower() or "denied" in result.lower()):
                        print(f"  ✅ Dangerous command blocked: {command}")
                        passed += 1
                    else:
                        print(f"  ❌ Dangerous command not blocked: {command}")
                elif expected == "success":
                    if "Error" not in result and "error" not in result.lower():
                        print(f"  ✅ Safe command executed: {command}")
                        passed += 1
                    else:
                        print(f"  ❌ Safe command failed: {command} -> {result[:50]}")
                else:
                    if expected.lower() in result.lower():
                        print(f"  ✅ {command} -> Contains '{expected}'")
                        passed += 1
                    else:
                        print(f"  ❌ {command} -> {result[:50]}, expected '{expected}'")
                        
            except Exception as e:
                if expected == "error":
                    print(f"  ✅ Exception handled: {command}")
                    passed += 1
                else:
                    print(f"  ❌ Unexpected exception: {command} -> {str(e)[:50]}")
        
        error_cases = len([tc for tc in test_cases if tc[1] == "error"])
        self.test_results['run_shell'] = f"{passed}/{total + error_cases} passed"
        
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("🧪 TOOL TESTING SUMMARY")
        print("=" * 60)
        
        total_passed = 0
        total_tests = 0
        
        for tool, result in self.test_results.items():
            passed, total = result.split('/')
            passed_num = int(passed.split('/')[0])
            total_num = int(total.split(' ')[0])
            
            total_passed += passed_num
            total_tests += total_num
            
            status = "✅ PASS" if passed_num == total_num else "⚠️ PARTIAL" if passed_num > 0 else "❌ FAIL"
            print(f"{tool:20} | {result:15} | {status}")
        
        print("-" * 60)
        overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"{'OVERALL':20} | {total_passed}/{total_tests} ({overall_percentage:.1f}%) | {'✅ EXCELLENT' if overall_percentage >= 90 else '⚠️ GOOD' if overall_percentage >= 70 else '❌ NEEDS WORK'}")
        
        print("\n📊 Success Criteria:")
        print(f"  Target: 95%+ tool reliability")
        print(f"  Current: {overall_percentage:.1f}%")
        print(f"  Status: {'✅ MET' if overall_percentage >= 95 else '⚠️ CLOSE' if overall_percentage >= 85 else '❌ NOT MET'}")


if __name__ == "__main__":
    tester = TestTools()
    tester.run_all_tests()
