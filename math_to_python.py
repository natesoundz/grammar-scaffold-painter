"""
math_to_python.py — Using math scaffold to write Python code.

The same content/scaffold split found in English definitions exists
natively in mathematical notation:

    content  = operands: numbers, variables, named objects
    scaffold = operators & quantifiers: + - * / = < > in subset forall
               exists -> sum prod lim sqrt not and or

And it exists natively in Python syntax too:

    content  = identifiers & literals
    scaffold = keywords & operators: if for while return def = == and or in

Because both domains encode the SAME relational content (a universal
quantifier is a universal quantifier whether written "for all x in S"
or "for x in S: assert ..."), a math scaffold can be transformed directly
into a Python scaffold while preserving content words/operands — exactly
the same operation as scaffold_transform.py's device-to-device rewrites,
but cross-domain instead of cross-rhetoric.

Each transform below is a regex-recognized math pattern paired with a
Python code generator. Content (variable/function names, numbers) passes
through unchanged; only the scaffold (notation) changes.

Usage:
    python math_to_python.py "forall x in S, x > 0"
    python math_to_python.py "sum_{i=1}^{n} a_i"
    python math_to_python.py "f(x) = x^2 + 1"
    python math_to_python.py --list
"""

import re
import argparse

MATH_TO_PY = []  # registered (name, description, example, regex, generator)


def _to_py_expr(expr):
    """Normalize a math sub-expression into Python syntax. Scaffold-only
    substitution — content (identifiers, numbers) is never touched."""
    expr = expr.strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("<=", "<=").replace(">=", ">=")
    expr = expr.replace("!=", "!=")
    expr = re.sub(r'\bnot\b', 'not', expr)
    # a_i style subscripts -> a[i]
    expr = re.sub(r'([A-Za-z]\w*)_(\w+)', r'\1[\2]', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    return expr


def _register(name, description, example, pattern):
    def deco(fn):
        MATH_TO_PY.append({
            "name": name, "description": description, "example": example,
            "regex": re.compile(pattern), "generate": fn,
        })
        return fn
    return deco


@_register(
    "universal_quantifier",
    "Universal quantifier -> all() over a domain",
    '"forall x in S, x > 0"  ->  "all(x > 0 for x in S)"',
    r'^\s*forall\s+(?P<var>\w+)\s*(?:in\s+(?P<domain>\w+))?\s*,\s*(?P<body>.+)$',
)
def _gen_forall(m):
    var, domain, body = m.group("var"), m.group("domain") or "DOMAIN", _to_py_expr(m.group("body"))
    return f"all({body} for {var} in {domain})"


@_register(
    "existential_quantifier",
    "Existential quantifier -> any() over a domain",
    '"exists x in S, x < 0"  ->  "any(x < 0 for x in S)"',
    r'^\s*exists\s+(?P<var>\w+)\s*(?:in\s+(?P<domain>\w+))?\s*,\s*(?P<body>.+)$',
)
def _gen_exists(m):
    var, domain, body = m.group("var"), m.group("domain") or "DOMAIN", _to_py_expr(m.group("body"))
    return f"any({body} for {var} in {domain})"


@_register(
    "summation",
    "Summation (sum) -> sum() over a range",
    '"sum_{i=1}^{n} a_i"  ->  "sum(a[i] for i in range(1, n + 1))"',
    r'^\s*sum_\{?(?P<var>\w+)\s*=\s*(?P<lo>\w+)\}?\^\{?(?P<hi>\w+)\}?\s*(?P<body>.+)$',
)
def _gen_sum(m):
    var, lo, hi, body = m.group("var"), m.group("lo"), m.group("hi"), _to_py_expr(m.group("body"))
    return f"sum({body} for {var} in range({lo}, {hi} + 1))"


@_register(
    "product",
    "Product (prod) -> math.prod() over a range",
    '"prod_{i=1}^{n} a_i"  ->  "math.prod(a[i] for i in range(1, n + 1))"',
    r'^\s*prod_\{?(?P<var>\w+)\s*=\s*(?P<lo>\w+)\}?\^\{?(?P<hi>\w+)\}?\s*(?P<body>.+)$',
)
def _gen_prod(m):
    var, lo, hi, body = m.group("var"), m.group("lo"), m.group("hi"), _to_py_expr(m.group("body"))
    return f"math.prod({body} for {var} in range({lo}, {hi} + 1))"


@_register(
    "piecewise_function",
    "Piecewise definition -> Python conditional expression",
    '"f(x) = a if x > 0 otherwise b"  ->  "def f(x):\\n    return a if x > 0 else b"',
    r'^\s*(?P<name>\w+)\((?P<args>[\w,\s]+)\)\s*=\s*(?P<a>.+?)\s+if\s+(?P<cond>.+?)\s+(?:otherwise|else)\s+(?P<b>.+)$',
)
def _gen_piecewise(m):
    name, args = m.group("name"), m.group("args")
    a, cond, b = _to_py_expr(m.group("a")), _to_py_expr(m.group("cond")), _to_py_expr(m.group("b"))
    return f"def {name}({args}):\n    return {a} if {cond} else {b}"


@_register(
    "function_definition",
    "Function definition f(x) = expr -> def f(x): return expr",
    '"f(x) = x^2 + 1"  ->  "def f(x):\\n    return x**2 + 1"',
    r'^\s*(?P<name>\w+)\((?P<args>[\w,\s]+)\)\s*=\s*(?P<body>.+)$',
)
def _gen_funcdef(m):
    name, args, body = m.group("name"), m.group("args"), _to_py_expr(m.group("body"))
    return f"def {name}({args}):\n    return {body}"


@_register(
    "implication",
    "Implication -> if statement",
    '"x > 0 -> y = 1"  ->  "if x > 0:\\n    y = 1"',
    r'^\s*(?P<cond>.+?)\s*->\s*(?P<body>.+)$',
)
def _gen_implication(m):
    cond, body = _to_py_expr(m.group("cond")), _to_py_expr(m.group("body"))
    return f"if {cond}:\n    {body}"


@_register(
    "assignment",
    "Assignment (:=) -> Python =",
    '"x := x + 1"  ->  "x = x + 1"',
    r'^\s*(?P<var>\w+)\s*:=\s*(?P<body>.+)$',
)
def _gen_assign(m):
    var, body = m.group("var"), _to_py_expr(m.group("body"))
    return f"{var} = {body}"


@_register(
    "set_builder",
    "Set-builder notation -> list comprehension",
    '"{x | x in S, x > 0}"  ->  "[x for x in S if x > 0]"',
    r'^\s*\{\s*(?P<expr>\w+)\s*(?:\||:)\s*(?P<var>\w+)\s+in\s+(?P<domain>\w+)\s*(?:,\s*(?P<cond>[^}]+))?\}\s*$',
)
def _gen_setbuilder(m):
    expr, var, domain, cond = m.group("expr"), m.group("var"), m.group("domain"), m.group("cond")
    if cond:
        return f"[{expr} for {var} in {domain} if {_to_py_expr(cond)}]"
    return f"[{expr} for {var} in {domain}]"


@_register(
    "set_membership",
    "Set membership (in) -> Python in",
    '"x in S"  ->  "x in S"',
    r'^\s*(?P<elem>\w+)\s+in\s+(?P<domain>\w+)\s*$',
)
def _gen_membership(m):
    return f"{m.group('elem')} in {m.group('domain')}"


def convert(expr):
    """Try each registered math->Python transform in order; first match wins."""
    expr = expr.strip()
    for t in MATH_TO_PY:
        m = t["regex"].match(expr)
        if m:
            return t["generate"](m), t["name"], t["description"]
    return None, None, None


def main():
    ap = argparse.ArgumentParser(description="Math scaffold -> Python scaffold compiler")
    ap.add_argument("expr", nargs="?", default="forall x in S, x > 0")
    ap.add_argument("--list", action="store_true", help="List all transforms")
    args = ap.parse_args()

    if args.list:
        print("=" * 70)
        print("MATH -> PYTHON SCAFFOLD TRANSFORMS")
        print("=" * 70)
        print()
        for t in MATH_TO_PY:
            print(f"  {t['name']}")
            print(f"    {t['description']}")
            print(f"    {t['example']}")
            print()
        return

    code, name, desc = convert(args.expr)
    print(f"Math:   {args.expr}")
    if code:
        print(f"Device: {name}  ({desc})")
        print("Python:")
        for line in code.split("\n"):
            print(f"  {line}")
    else:
        print("No matching transform found.")
        print("Available patterns:")
        for t in MATH_TO_PY:
            print(f"  {t['name']}: {t['example']}")


if __name__ == "__main__":
    main()
