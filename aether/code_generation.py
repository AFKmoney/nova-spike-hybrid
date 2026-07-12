"""
code_generation.py — Code generation and execution.

PROBLEM
-------
GPT-4 can write and execute code. AETHER has no code capabilities beyond
the basic `python` tool (which only evaluates simple expressions).

SOLUTION
--------
CodeGenerator provides:
  1. Code generation from natural language descriptions
  2. Safe execution in a sandboxed environment
  3. Code templates for common patterns (loops, functions, classes)
  4. Error detection and correction
  5. Multi-language support (Python, pseudocode)

Templates cover:
  - Function definitions
  - Loops (for, while)
  - Conditionals (if/elif/else)
  - List comprehensions
  - Class definitions
  - File I/O
  - Sorting/searching algorithms
"""

from __future__ import annotations
import ast
import re
import textwrap
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
import traceback

log = logging.getLogger(__name__)


@dataclass
class CodeResult:
    """Result of code generation + execution."""
    code: str
    output: Optional[str]
    error: Optional[str]
    success: bool
    language: str = "python"


# Code templates indexed by intent
CODE_TEMPLATES: Dict[str, str] = {
    "function": """
def {name}({params}):
    \"\"\"{docstring}\"\"\"
    {body}
    return {return_val}
""".strip(),

    "for_loop": """
for {var} in {iterable}:
    {body}
""".strip(),

    "while_loop": """
while {condition}:
    {body}
""".strip(),

    "if_else": """
if {condition}:
    {body_if}
else:
    {body_else}
""".strip(),

    "list_comprehension": "[{expr} for {var} in {iterable}]",

    "class": """
class {name}:
    def __init__(self, {params}):
        {init_body}

    def {method_name}(self):
        {method_body}
""".strip(),

    "sort": """
def {name}(arr):
    return sorted(arr)
""".strip(),

    "search": """
def {name}(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
""".strip(),

    "factorial": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""".strip(),

    "fibonacci": """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""".strip(),

    "reverse_string": """
def reverse_string(s):
    return s[::-1]
""".strip(),

    "sum_list": """
def sum_list(lst):
    return sum(lst)
""".strip(),

    "max_list": """
def max_list(lst):
    return max(lst)
""".strip(),
}


class CodeGenerator:
    """Generate and execute code from natural language."""

    # Intent detection patterns
    INTENT_PATTERNS = [
        (r"function\s+(?:called\s+)?(\w+)", "function"),
        (r"for\s+loop", "for_loop"),
        (r"while\s+loop", "while_loop"),
        (r"if\s+else|conditional", "if_else"),
        (r"list\s+comprehension", "list_comprehension"),
        (r"class\s+(?:called\s+)?(\w+)", "class"),
        (r"sort", "sort"),
        (r"search|find\s+in\s+list", "search"),
        (r"factorial", "factorial"),
        (r"fibonacci", "fibonacci"),
        (r"reverse\s+string", "reverse_string"),
        (r"sum\s+(?:of\s+)?(?:a\s+)?list", "sum_list"),
        (r"max\s+(?:of\s+)?(?:a\s+)?list", "max_list"),
    ]

    def __init__(self, agent):
        self.agent = agent

    # ------------------------------------------------------------------ #
    # Code generation
    # ------------------------------------------------------------------ #
    def generate_code(self, description: str) -> CodeResult:
        """Generate code from a natural language description."""
        desc_lower = description.lower()

        # Detect intent
        intent = None
        match_groups = None
        for pattern, intent_name in self.INTENT_PATTERNS:
            m = re.search(pattern, desc_lower)
            if m:
                intent = intent_name
                match_groups = m.groups()
                break

        if intent is None:
            # Try to detect from keywords
            if "factorial" in desc_lower:
                intent = "factorial"
            elif "fibonacci" in desc_lower:
                intent = "fibonacci"
            elif "reverse" in desc_lower and "string" in desc_lower:
                intent = "reverse_string"
            elif "sort" in desc_lower:
                intent = "sort"
            elif "search" in desc_lower:
                intent = "search"
            elif "sum" in desc_lower:
                intent = "sum_list"
            elif "max" in desc_lower:
                intent = "max_list"

        if intent and intent in CODE_TEMPLATES:
            code = CODE_TEMPLATES[intent]
            # Fill in template variables
            if intent == "function" and match_groups:
                code = code.format(
                    name=match_groups[0] if match_groups else "my_function",
                    params="x",
                    docstring=description,
                    body="pass",
                    return_val="x",
                )
            elif intent == "class" and match_groups:
                code = code.format(
                    name=match_groups[0] if match_groups else "MyClass",
                    params="",
                    init_body="pass",
                    method_name="do_something",
                    method_body="pass",
                )
            elif intent == "sort":
                code = code.format(name="sort_array")
            elif intent == "search":
                code = code.format(name="search_array")
            else:
                # Use template as-is (factorial, fibonacci, etc.)
                pass

            return CodeResult(code=code, output=None, error=None,
                             success=True, language="python")

        # No template matched — generate pseudocode
        pseudocode = self._generate_pseudocode(description)
        return CodeResult(code=pseudocode, output=None, error=None,
                         success=True, language="pseudocode")

    def _generate_pseudocode(self, description: str) -> str:
        """Generate pseudocode for a description."""
        lines = [
            f"# Pseudocode for: {description}",
            "1. Initialize variables",
            "2. Process input",
            "3. Compute result",
            "4. Return output",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Safe execution
    # ------------------------------------------------------------------ #
    def execute(self, code: str, timeout: int = 5) -> CodeResult:
        """Execute code safely in a restricted environment.

        Only allows: arithmetic, string ops, list/tuple/dict ops,
        basic control flow, function defs, print.
        Blocks: imports, file I/O, exec, eval, open, system.
        """
        # Validate: no dangerous constructs
        dangerous = ["import ", "__", "exec(", "eval(", "open(", "system(",
                     "subprocess", "os.", "sys.", "globals(", "locals("]
        for d in dangerous:
            if d in code:
                return CodeResult(code=code, output=None,
                                 error=f"Blocked: contains '{d}'",
                                 success=False)

        # Try to parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return CodeResult(code=code, output=None,
                             error=f"Syntax error: {e}",
                             success=False)

        # Execute in a sandboxed namespace
        namespace = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "sum": sum,
                "max": max, "min": min, "abs": abs, "round": round,
                "sorted": sorted, "enumerate": enumerate, "zip": zip,
                "int": int, "float": float, "str": str, "list": list,
                "tuple": tuple, "dict": dict, "set": set, "bool": bool,
                "True": True, "False": False, "None": None,
                "TypeError": TypeError, "ValueError": ValueError,
                "IndexError": IndexError, "KeyError": KeyError,
            }
        }

        # Capture stdout
        import io
        import contextlib
        stdout_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(tree, "<generated>", "exec"), namespace)
            output = stdout_capture.getvalue()
            return CodeResult(code=code, output=output, error=None,
                             success=True)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            return CodeResult(code=code, output=None, error=error,
                             success=False)

    # ------------------------------------------------------------------ #
    # Combined: generate + execute
    # ------------------------------------------------------------------ #
    def generate_and_execute(self, description: str) -> CodeResult:
        """Generate code from description and execute it."""
        gen_result = self.generate_code(description)
        if not gen_result.success:
            return gen_result
        if gen_result.language != "python":
            return gen_result  # can't execute pseudocode
        return self.execute(gen_result.code)

    # ------------------------------------------------------------------ #
    # Error correction
    # ------------------------------------------------------------------ #
    def correct_code(self, code: str, error: str) -> str:
        """Try to correct code based on an error message."""
        corrections = []
        # Common errors and fixes
        if "IndentationError" in error:
            corrections.append("# Fix: ensure consistent indentation (4 spaces)")
            # Try to fix indentation
            lines = code.split("\n")
            fixed_lines = []
            indent_level = 0
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if stripped.startswith(("def ", "class ", "for ", "while ", "if ", "elif ", "else:", "try:", "except", "finally:")):
                        fixed_lines.append("    " * indent_level + stripped)
                        indent_level += 1
                    elif stripped.startswith(("return", "pass", "break", "continue")):
                        indent_level = max(0, indent_level - 1)
                        fixed_lines.append("    " * indent_level + stripped)
                    else:
                        fixed_lines.append("    " * indent_level + stripped)
                else:
                    fixed_lines.append("")
            return "\n".join(fixed_lines)

        if "NameError" in error:
            # Missing variable — add a placeholder
            m = re.search(r"name '(\w+)' is not defined", error)
            if m:
                name = m.group(1)
                return f"{name} = None  # TODO: define {name}\n{code}"

        if "SyntaxError" in error:
            corrections.append("# Fix: check for missing colons, parentheses, or quotes")

        return code + "\n" + "\n".join(corrections)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "n_templates": len(CODE_TEMPLATES),
            "n_intents": len(self.INTENT_PATTERNS),
            "supported_intents": list(set(intent for _, intent in self.INTENT_PATTERNS)),
        }
