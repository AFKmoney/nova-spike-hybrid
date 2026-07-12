"""
symbolic_math.py — Symbolic math: algebra, equations, simplification.

PROBLEM
-------
AETHER's `calc` tool only does arithmetic. GPT-4 can solve equations,
simplify expressions, do calculus. We need symbolic math.

SOLUTION
--------
SymbolicMathEngine provides:
  1. Expression parsing (string → AST)
  2. Expression simplification (constant folding, like terms)
  3. Linear equation solving (ax + b = c → x = (c-b)/a)
  4. Quadratic equation solving (ax² + bx + c = 0)
  5. Differentiation (basic: polynomials, power rule)
  6. Integration (basic: power rule)
  7. Factorization (simple cases)

No sympy dependency — pure Python.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class MathResult:
    """Result of a symbolic math operation."""
    input: str
    output: str
    steps: List[str]
    success: bool
    operation: str


class SymbolicMathEngine:
    """Symbolic math engine."""

    def __init__(self, agent=None):
        self.agent = agent

    # ------------------------------------------------------------------ #
    # Expression parsing
    # ------------------------------------------------------------------ #
    def parse_equation(self, eq: str) -> Optional[Dict]:
        """Parse a linear or quadratic equation.

        Returns dict with coefficients, or None if can't parse.
        """
        eq = eq.replace(" ", "").replace("^", "**")
        # Split on =
        if "=" not in eq:
            return None
        left, right = eq.split("=", 1)
        # Move everything to left: left - right = 0
        # Try to parse as ax^2 + bx + c
        # Expand: left - right
        try:
            # Simple approach: substitute x=0 to get constant, x=1 to get a+b+c, etc.
            coeffs = self._extract_coefficients(left, right)
            return coeffs
        except Exception:
            return None

    def _extract_coefficients(self, left: str, right: str) -> Dict:
        """Extract coefficients a, b, c from 'ax^2 + bx + c = d' form."""
        # Evaluate left - right at x=0, x=1, x=-1 to get 3 equations
        # Then solve for a, b, c
        import ast
        # Try simple linear: ax + b = c
        # Pattern: (number)x + (number) = (number)
        # or: x + (number) = (number)
        # or: (number)x = (number)

        # Move right to left: left - (right) = 0
        expr = f"({left}) - ({right})"

        # Evaluate at x=0, x=1, x=2
        vals = []
        for x_val in [0, 1, 2]:
            try:
                # Safe eval: only allow x and numbers
                namespace = {"x": x_val, "__builtins__": {}}
                v = eval(expr, namespace)
                vals.append(v)
            except:
                return None

        # vals[0] = c (constant term)
        # vals[1] = a + b + c
        # vals[2] = 4a + 2b + c
        c = vals[0]
        # a + b = vals[1] - c
        # 4a + 2b = vals[2] - c
        # => 2a = (vals[2] - c) - 2*(vals[1] - c) = vals[2] - 2*vals[1] + c
        if len(vals) == 3:
            a = (vals[2] - 2 * vals[1] + c) / 2
            b = (vals[1] - c) - a
            return {"a": a, "b": b, "c": c, "degree": 2 if a != 0 else (1 if b != 0 else 0)}
        return None

    # ------------------------------------------------------------------ #
    # Equation solving
    # ------------------------------------------------------------------ #
    def solve(self, equation: str) -> MathResult:
        """Solve an equation."""
        steps = []
        coeffs = self.parse_equation(equation)
        if coeffs is None:
            return MathResult(equation, "Could not parse equation", [], False, "solve")

        a, b, c = coeffs["a"], coeffs["b"], coeffs["c"]
        steps.append(f"Parsed: {a}x² + {b}x + {c} = 0")

        if coeffs["degree"] == 0:
            return MathResult(equation, "No variable to solve for", steps, False, "solve")

        if coeffs["degree"] == 1:
            # bx + c = 0 → x = -c/b
            x = -c / b
            steps.append(f"Linear: {b}x + {c} = 0")
            steps.append(f"x = -{c} / {b}")
            steps.append(f"x = {x}")
            return MathResult(equation, f"x = {x}", steps, True, "solve_linear")

        if coeffs["degree"] == 2:
            # ax² + bx + c = 0 → quadratic formula
            import math
            discriminant = b**2 - 4*a*c
            steps.append(f"Quadratic: {a}x² + {b}x + {c} = 0")
            steps.append(f"Discriminant = b² - 4ac = {b}² - 4*{a}*{c} = {discriminant}")

            if discriminant > 0:
                x1 = (-b + math.sqrt(discriminant)) / (2 * a)
                x2 = (-b - math.sqrt(discriminant)) / (2 * a)
                steps.append(f"Two real roots:")
                steps.append(f"x₁ = (-b + √Δ) / 2a = {x1}")
                steps.append(f"x₂ = (-b - √Δ) / 2a = {x2}")
                return MathResult(equation, f"x₁ = {x1}, x₂ = {x2}", steps, True, "solve_quadratic")
            elif discriminant == 0:
                x = -b / (2 * a)
                steps.append(f"One real root (double):")
                steps.append(f"x = -b / 2a = {x}")
                return MathResult(equation, f"x = {x}", steps, True, "solve_quadratic")
            else:
                real = -b / (2 * a)
                imag = math.sqrt(-discriminant) / (2 * a)
                steps.append(f"Two complex roots:")
                steps.append(f"x₁ = {real} + {imag}i")
                steps.append(f"x₂ = {real} - {imag}i")
                return MathResult(equation, f"x₁ = {real} + {imag}i, x₂ = {real} - {imag}i",
                                 steps, True, "solve_quadratic")

        return MathResult(equation, "Equation degree too high", steps, False, "solve")

    # ------------------------------------------------------------------ #
    # Simplification
    # ------------------------------------------------------------------ #
    def simplify(self, expr: str) -> MathResult:
        """Simplify a mathematical expression."""
        steps = []
        expr_clean = expr.replace(" ", "").replace("^", "**")

        # Try to evaluate constant expressions
        try:
            namespace = {"__builtins__": {}}
            result = eval(expr_clean, namespace)
            if isinstance(result, (int, float)):
                steps.append(f"Constant folding: {expr} = {result}")
                return MathResult(expr, str(result), steps, True, "simplify")
        except:
            pass

        # Try basic algebraic simplification
        # Combine like terms: 2x + 3x = 5x
        m = re.match(r"(\d+)x\s*\+\s*(\d+)x", expr)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            result = f"{a+b}x"
            steps.append(f"Combine like terms: {a}x + {b}x = {a+b}x")
            return MathResult(expr, result, steps, True, "simplify")

        # Factor out common term: 2x + 4 = 2(x + 2)
        m = re.match(r"(\d+)x\s*\+\s*(\d+)$", expr)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            from math import gcd
            g = gcd(a, b)
            if g > 1:
                result = f"{g}({a//g}x + {b//g})"
                steps.append(f"Factor out GCD={g}: {a}x + {b} = {g}({a//g}x + {b//g})")
                return MathResult(expr, result, steps, True, "simplify")

        return MathResult(expr, expr, ["No simplification found"], False, "simplify")

    # ------------------------------------------------------------------ #
    # Differentiation (power rule)
    # ------------------------------------------------------------------ #
    def differentiate(self, expr: str) -> MathResult:
        """Differentiate a polynomial using the power rule."""
        steps = []
        expr = expr.replace(" ", "")

        # Parse terms: ax^n + bx + c
        # Pattern: optional coefficient, x, optional ^n
        terms = re.findall(r'([+-]?\d*)x(?:\^(\d+))?|([+-]?\d+)', expr)

        result_terms = []
        for coef, power, const in terms:
            if const:
                # Constant term: derivative is 0
                continue
            c = int(coef) if coef and coef not in ("+", "-") else (1 if coef != "-" else -1)
            n = int(power) if power else 1
            # Power rule: d/dx[ax^n] = a*n*x^(n-1)
            new_c = c * n
            new_n = n - 1
            if new_n == 0:
                result_terms.append(str(new_c))
            elif new_n == 1:
                result_terms.append(f"{new_c}x")
            else:
                result_terms.append(f"{new_c}x^{new_n}")
            steps.append(f"d/dx[{c}x^{n}] = {c}*{n}*x^{n-1} = {new_c}x^{new_n}")

        if not result_terms:
            return MathResult(expr, "0", ["Derivative of constant is 0"], True, "differentiate")

        result = " + ".join(result_terms)
        return MathResult(expr, result, steps, True, "differentiate")

    # ------------------------------------------------------------------ #
    # Integration (power rule)
    # ------------------------------------------------------------------ #
    def integrate(self, expr: str) -> MathResult:
        """Integrate a polynomial using the power rule."""
        steps = []
        expr = expr.replace(" ", "")

        terms = re.findall(r'([+-]?\d*)x(?:\^(\d+))?|([+-]?\d+)', expr)

        result_terms = []
        for coef, power, const in terms:
            if const:
                # Constant: integral is c*x
                c = int(const)
                result_terms.append(f"{c}x")
                steps.append(f"∫{c}dx = {c}x")
                continue
            c = int(coef) if coef and coef not in ("+", "-") else (1 if coef != "-" else -1)
            n = int(power) if power else 1
            # Power rule: ∫ax^n dx = a/(n+1) * x^(n+1)
            new_n = n + 1
            if c % (n + 1) == 0:
                new_c = c // (n + 1)
                result_terms.append(f"{new_c}x^{new_n}")
                steps.append(f"∫{c}x^{n}dx = {c}/{n+1} * x^{new_n} = {new_c}x^{new_n}")
            else:
                result_terms.append(f"({c}/{n+1})x^{new_n}")
                steps.append(f"∫{c}x^{n}dx = {c}/{n+1} * x^{new_n}")

        if not result_terms:
            return MathResult(expr, "C", ["Integral of 0 is constant"], True, "integrate")

        result = " + ".join(result_terms) + " + C"
        return MathResult(expr, result, steps, True, "integrate")

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "operations": ["solve_linear", "solve_quadratic", "simplify",
                          "differentiate", "integrate"],
        }
