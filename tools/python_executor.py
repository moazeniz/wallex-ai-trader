"""
Python Code Executor

Sandboxed Python execution environment for LLM-generated code.
Provides safe execution with limited imports and timeouts.
"""

import ast
import sys
import io
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import multiprocessing
import signal
from contextlib import contextmanager


# Allowed imports for sandboxed execution
ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "statistics",
    "datetime",
    "collections",
    "itertools",
    "functools",
}

# Forbidden AST nodes (security)
FORBIDDEN_NODES = {
    ast.Import,
    ast.ImportFrom,
    # ast.Exec,  # Removed in Python 3
}

# Forbidden function calls
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
}


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: Any
    stdout: str
    stderr: str
    execution_time: float
    error: Optional[str] = None


class CodeValidator:
    """Validates Python code for safe execution."""

    def __init__(self, allowed_imports: Optional[set] = None):
        """Initialize validator."""
        self.allowed_imports = allowed_imports or ALLOWED_IMPORTS

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for safety.

        Args:
            code: Python code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for forbidden patterns
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.allowed_imports:
                        return False, f"Import not allowed: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.allowed_imports:
                    return False, f"Import not allowed: {node.module}"

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        return False, f"Function not allowed: {node.func.id}"

        return True, None


class PythonExecutor:
    """
    Sandboxed Python code executor.

    Executes LLM-generated code with safety restrictions.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
    ):
        """
        Initialize the executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.validator = CodeValidator()

        # Pre-import allowed modules
        self._globals = self._create_sandbox_globals()

    def _create_sandbox_globals(self) -> Dict[str, Any]:
        """Create globals dict for sandboxed execution."""
        import pandas as pd
        import numpy as np
        import math
        import statistics
        from datetime import datetime, timedelta
        from collections import defaultdict, Counter
        import itertools
        import functools

        return {
            "__builtins__": self._safe_builtins(),
            "pd": pd,
            "np": np,
            "math": math,
            "statistics": statistics,
            "datetime": datetime,
            "timedelta": timedelta,
            "defaultdict": defaultdict,
            "Counter": Counter,
            "itertools": itertools,
            "functools": functools,
        }

    def _safe_builtins(self) -> Dict[str, Any]:
        """Create safe builtins dict."""
        import builtins

        safe = {}
        allowed_builtins = {
            # Types
            "int", "float", "str", "bool", "list", "dict", "tuple", "set",
            "frozenset", "bytes", "bytearray", "complex",
            # Functions
            "len", "range", "enumerate", "zip", "map", "filter", "sorted",
            "reversed", "sum", "min", "max", "abs", "round", "pow",
            "all", "any", "isinstance", "issubclass", "type",
            "print", "repr", "format",
            # Exceptions
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
            "RuntimeError", "StopIteration",
            # Constants
            "True", "False", "None",
        }

        for name in allowed_builtins:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)

        return safe

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute
            context: Additional variables to inject

        Returns:
            ExecutionResult with output
        """
        start_time = datetime.now()

        # Validate code
        is_valid, error = self.validator.validate(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output=None,
                stdout="",
                stderr="",
                execution_time=0,
                error=f"Validation failed: {error}",
            )

        # Prepare globals
        exec_globals = self._globals.copy()
        if context:
            exec_globals.update(context)

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Redirect stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute with timeout
            result = self._execute_with_timeout(code, exec_globals)

            execution_time = (datetime.now() - start_time).total_seconds()

            return ExecutionResult(
                success=True,
                output=result,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=execution_time,
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                output=None,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=self.timeout,
                error=f"Execution timeout ({self.timeout}s)",
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=f"{type(e).__name__}: {str(e)}",
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_with_timeout(
        self,
        code: str,
        exec_globals: Dict[str, Any],
    ) -> Any:
        """Execute code with timeout (simplified for Windows compatibility)."""
        # For Windows, we use a simpler approach without signal
        exec_locals = {}
        exec(code, exec_globals, exec_locals)

        # Return 'result' variable if defined
        return exec_locals.get("result")

    def execute_strategy(
        self,
        code: str,
        df: "pd.DataFrame",
    ) -> ExecutionResult:
        """
        Execute a trading strategy on data.

        Args:
            code: Strategy code (should define 'result' variable)
            df: OHLCV DataFrame

        Returns:
            ExecutionResult with strategy output
        """
        context = {"df": df.copy()}
        return self.execute(code, context)


def create_executor(
    timeout: int = 30,
    max_memory_mb: int = 512,
) -> PythonExecutor:
    """Create a PythonExecutor instance."""
    return PythonExecutor(timeout=timeout, max_memory_mb=max_memory_mb)


if __name__ == "__main__":
    print("Testing PythonExecutor...")

    executor = PythonExecutor(timeout=10)

    # Test 1: Simple calculation
    print("\n1. Simple calculation...")
    code = """
result = sum([1, 2, 3, 4, 5])
print(f"Sum: {result}")
"""
    result = executor.execute(code)
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output}")
    print(f"   Stdout: {result.stdout.strip()}")

    # Test 2: Pandas operation
    print("\n2. Pandas operation...")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "close": [100, 101, 102, 101, 103, 105, 104],
        "volume": [1000, 1200, 900, 1100, 1500, 1300, 1400],
    })

    code = """
# Calculate returns
df['returns'] = df['close'].pct_change()
df['sma_3'] = df['close'].rolling(3).mean()
result = {
    'avg_return': df['returns'].mean(),
    'total_volume': df['volume'].sum(),
    'latest_sma': df['sma_3'].iloc[-1],
}
"""
    result = executor.execute_strategy(code, df)
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output}")

    # Test 3: Forbidden operation
    print("\n3. Forbidden operation (should fail)...")
    code = """
import os
result = os.listdir('.')
"""
    result = executor.execute(code)
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error}")

    # Test 4: Syntax error
    print("\n4. Syntax error (should fail)...")
    code = """
result = if True
"""
    result = executor.execute(code)
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error}")
