import subprocess
import tempfile
import os
import re


ALLOWED_IMPORTS = {
    "os", "sys", "re", "json", "math", "random", "datetime", "time",
    "collections", "itertools", "functools", "string", "pathlib",
    "urllib", "http", "csv", "io", "copy", "typing", "dataclasses",
    "enum", "abc", "contextlib", "traceback", "textwrap", "hashlib",
    "base64", "uuid", "struct", "array", "heapq", "bisect", "queue",
    "threading", "multiprocessing", "subprocess", "shutil", "glob",
    "sqlite3", "xml", "html", "urllib.parse", "statistics", "decimal",
}

BLOCKED_PATTERNS = [
    r"import\s+requests",
    r"import\s+httpx",
    r"import\s+aiohttp",
    r"__import__\s*\(",
    r"open\s*\([^)]*['\"]w['\"]",  # file write
    r"os\.system",
    r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
    r"eval\s*\(",
    r"exec\s*\(",
    r"rm\s+-rf",
    r"shutil\.rmtree",
]


def is_safe_code(code: str) -> tuple[bool, str]:
    """Check if code is safe to execute."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Blocked pattern detected: {pattern}"
    return True, "OK"


def execute_python(code: str, timeout: int = 10) -> dict:
    """Safely execute Python code in a subprocess."""
    safe, reason = is_safe_code(code)
    if not safe:
        return {"success": False, "output": "", "error": f"Safety check failed: {reason}", "executed": False}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout[:3000],
            "error": result.stderr[:1000] if result.stderr else "",
            "executed": True,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Execution timed out (10s limit)", "executed": True}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e), "executed": True}
    finally:
        os.unlink(tmp_path)
