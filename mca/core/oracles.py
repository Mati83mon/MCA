"""Symbolic oracles: claims that can be *computed* instead of retrieved.

* :class:`PythonSemanticsOracle` executes tiny, AST-whitelisted Python snippets
  found in a claim (``all([])``, ``t[0] = 7``) and compares the observed
  behaviour with what the claim states.
* :class:`NumberTheoryOracle` decides primality / parity claims about integers.

An oracle verdict is authoritative: it overrides retrieval-based NLI.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from mca.telemetry.schemas import PremiseStatus


@dataclass
class OracleVerdict:
    oracle: str
    status: PremiseStatus
    explanation: str
    observed: str = ""


_SAFE_BUILTINS = {
    "all": all, "any": any, "bool": bool, "len": len, "int": int, "float": float, "str": str,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "sorted": sorted, "sum": sum,
    "min": min, "max": max, "abs": abs, "repr": repr, "type": type, "isinstance": isinstance,
    "TypeError": TypeError,
}
_SAFE_ATTRS = {("json", "dumps"), ("json", "loads")}
_ALLOWED_NODES = (
    ast.Module, ast.Expression, ast.Expr, ast.Assign, ast.Call, ast.Name, ast.Attribute, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.BoolOp, ast.Load, ast.Store, ast.operator, ast.unaryop, ast.cmpop, ast.boolop,
)


class UnsafeCode(ValueError):
    pass


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeCode(f"node {type(node).__name__} not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise UnsafeCode("dunder names are not allowed")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and (node.value.id, node.attr) in _SAFE_ATTRS):
                raise UnsafeCode(f"attribute {node.attr} not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in _SAFE_BUILTINS:
            raise UnsafeCode(f"call to {node.func.id} not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 200:
            raise UnsafeCode("constant too large")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise UnsafeCode("exponentiation not allowed")


def run_snippet(code: str) -> tuple[str, object]:
    """Execute a whitelisted snippet. Returns ("value", v) or ("error", ExceptionName)."""
    if len(code) > 300:
        raise UnsafeCode("snippet too long")
    tree = ast.parse(code, mode="exec")
    _check_ast(tree)
    env: dict[str, object] = {"__builtins__": _SAFE_BUILTINS, "json": json}
    body = tree.body
    try:
        if body and isinstance(body[-1], ast.Expr):
            exec(compile(ast.Module(body=body[:-1], type_ignores=[]), "<oracle>", "exec"), env)  # noqa: S102 - AST-whitelisted
            value = eval(compile(ast.Expression(body=body[-1].value), "<oracle>", "eval"), env)  # noqa: S307
            return "value", value
        exec(compile(tree, "<oracle>", "exec"), env)  # noqa: S102
        return "value", None
    except Exception as exc:  # noqa: BLE001 - the exception *is* the observation
        return "error", type(exc).__name__


_CALL_RE = re.compile(r"\b((?:json\.)?[a-z_]+\((?:[^()]|\([^()]*\))*\))")
_ASSIGN_RE = re.compile(r"\b([a-z_]\w*\[[^\]]+\]\s*=\s*[^,?]+?)(?=\s+(?:zmienia|changes|modyfikuje)|\s*[,?]|$)")
_BINDING_RE = re.compile(r"\b([a-z_]\w*\s*=\s*(?:\([^()]*\)|\[[^\[\]]*\]))")
_STATED_VALUE_RE = re.compile(r"(?:zwraca|daje|returns|równa się|wynosi)\s+[„\"']?(True|False|None|true|false|null|-?\d+)\b")


def _literal(word: str) -> object:
    return {"True": True, "False": False, "None": None}.get(word, word)


class PythonSemanticsOracle:
    """Verifies claims about the behaviour of small Python expressions."""

    name = "python"

    def applies(self, text: str) -> bool:
        return bool(_CALL_RE.search(text) or _ASSIGN_RE.search(text)) and bool(
            re.search(r"python|json|krotk|list|tuple|\(\)|\[\]", text, re.IGNORECASE)
        )

    def evaluate(self, text: str) -> OracleVerdict | None:
        assign = _ASSIGN_RE.search(text)
        if assign and re.search(r"zmienia|changes|modyfikuje", text):
            bindings = [b for b in _BINDING_RE.findall(text) if not b.startswith(assign.group(1).split("[")[0] + "[")]
            code = "\n".join(bindings + [assign.group(1).strip()])
            try:
                kind, observed = run_snippet(code)
            except (UnsafeCode, SyntaxError):
                return None
            if kind == "error":
                return OracleVerdict(self.name, PremiseStatus.CONTRADICTED,
                                     f"Wykonanie `{code.replace(chr(10), '; ')}` zgłasza {observed} – operacja niczego nie zmienia.",
                                     str(observed))
            return OracleVerdict(self.name, PremiseStatus.SUPPORTED, f"`{code}` wykonuje się poprawnie.")
        call = _CALL_RE.search(text)
        if not call:
            return None
        expr = call.group(1)
        try:
            kind, observed = run_snippet(expr)
        except (UnsafeCode, SyntaxError):
            return None
        shown = repr(observed) if kind == "value" else f"wyjątek {observed}"
        stated = _STATED_VALUE_RE.search(text[call.end():])
        if stated is None:
            return OracleVerdict(self.name, PremiseStatus.SUPPORTED, f"`{expr}` daje {shown}.", shown)
        expected = _literal(stated.group(1))
        matches = kind == "value" and (observed == expected or (isinstance(observed, str) and observed == str(expected)))
        if isinstance(observed, bool) and isinstance(expected, str):
            matches = False
        status = PremiseStatus.SUPPORTED if matches else PremiseStatus.CONTRADICTED
        return OracleVerdict(self.name, status, f"`{expr}` daje {shown}, a nie {stated.group(1)}." if not matches else f"`{expr}` daje {shown}.", shown)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


class NumberTheoryOracle:
    """Decides 'n is prime' / 'n is even/odd' claims embedded in a sentence."""

    name = "number_theory"
    _PRIME = re.compile(r"liczb\w*\s+pierwsz\w*\s+(\d+)|(\d+)\s+jest\s+liczbą\s+pierwszą|prime\s+(?:number\s+)?(\d+)|(\d+)\s+is\s+(?:a\s+)?prime")
    _PARITY = re.compile(r"(\d+)\s+jest\s+(?:liczbą\s+)?(nie)?parzyst")

    def applies(self, text: str) -> bool:
        return bool(self._PRIME.search(text) or self._PARITY.search(text))

    def evaluate(self, text: str) -> OracleVerdict | None:
        m = self._PRIME.search(text)
        if m:
            n = int(next(g for g in m.groups() if g))
            divisors = [d for d in range(1, n + 1) if n % d == 0] if n <= 10_000 else []
            if is_prime(n):
                return OracleVerdict(self.name, PremiseStatus.SUPPORTED, f"{n} jest liczbą pierwszą.")
            extra = f" (dodatnie dzielniki: {divisors})" if divisors else ""
            return OracleVerdict(self.name, PremiseStatus.CONTRADICTED,
                                 f"{n} nie jest liczbą pierwszą{extra}; liczba pierwsza ma dokładnie dwa dodatnie dzielniki.")
        m = self._PARITY.search(text)
        if m:
            n, odd_claim = int(m.group(1)), bool(m.group(2))
            ok = (n % 2 == 1) == odd_claim
            word = "nieparzysta" if n % 2 else "parzysta"
            return OracleVerdict(self.name, PremiseStatus.SUPPORTED if ok else PremiseStatus.CONTRADICTED, f"{n} jest liczbą {word}.")
        return None


DEFAULT_ORACLES = (PythonSemanticsOracle(), NumberTheoryOracle())
