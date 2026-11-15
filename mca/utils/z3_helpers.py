"""Safe propositional/arithmetic DSL -> Z3 compiler and solver utilities.

Formal specs are produced either by the rule-based formalizer or by an LLM
(JSON). They are *never* passed to ``eval``: expressions are parsed with
:mod:`ast` and compiled node-by-node against a whitelist.

Spec format::

    {
      "sorts": ["Obj"],                                   # optional uninterpreted sorts
      "vars": {"x": "Real", "n": "Int", "p": "Bool", "b": "Obj"},
      "functions": {"S": ["Obj", "Obj", "Bool"]},         # domain..., range
      "assertions": [{"id": "a1", "expr": "x*x == -9", "text": "x² = -9"}],
      "target": "x"                                        # optional, for determinacy
    }

Every Z3 object is created in a private :class:`z3.Context`, so independent
checks are safe to run from worker threads.
"""

from __future__ import annotations

import ast
import re
import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import z3
from pydantic import BaseModel, Field

_SORTS = {"Bool", "Int", "Real"}

_INFIX_FIXES = (
    (re.compile(r"(?<=[\w)\]])\s+And\s+"), " and "),  # infix only; And(a, b) calls stay intact
    (re.compile(r"(?<=[\w)\]])\s+Or\s+"), " or "),
    (re.compile(r"\bNot\s+(?=[^\s(])"), "not "),
    (re.compile(r"->|=>"), ">>"),
)


def normalize_expr(expr: str) -> str:
    """Accept the infix spelling small LLMs often emit (``a And b``, ``p -> q``)."""
    for pattern, repl in _INFIX_FIXES:
        expr = pattern.sub(repl, expr)
    return expr.strip()



class Assertion(BaseModel):
    id: str
    expr: str
    text: str = ""


class FormalSpec(BaseModel):
    """Declarative formalization of a set of claims."""

    sorts: list[str] = Field(default_factory=list)
    vars: dict[str, str] = Field(default_factory=dict)
    functions: dict[str, list[str]] = Field(default_factory=dict)
    assertions: list[Assertion] = Field(default_factory=list)
    target: str | None = None
    target_description: dict[str, str] = Field(default_factory=dict, description="free var -> human description")
    self_reference: dict[str, str] = Field(default_factory=dict, description="Bool var -> defining formula (P := f(P,...))")
    entailment_goal: str | None = Field(None, description="Conclusion claimed to follow from the assertions.")
    label: str = ""


class DSLError(ValueError):
    """Raised when an expression uses syntax outside the whitelist."""


@dataclass
class Compiled:
    ctx: z3.Context
    symbols: dict[str, Any]
    assertions: list[tuple[str, z3.BoolRef, str]]
    sorts: dict[str, z3.SortRef] = field(default_factory=dict)


def _const(value: Any, ctx: z3.Context) -> z3.ExprRef:
    if isinstance(value, bool):
        return z3.BoolVal(value, ctx)
    if isinstance(value, int):
        return z3.IntVal(value, ctx)
    if isinstance(value, float):
        frac = Fraction(str(value))
        return z3.RealVal(f"{frac.numerator}/{frac.denominator}", ctx)
    raise DSLError(f"unsupported constant {value!r}")


class _Compiler:
    def __init__(self, symbols: dict[str, Any], ctx: z3.Context) -> None:
        self.symbols = symbols
        self.ctx = ctx
        self.calls: dict[str, Callable[..., z3.ExprRef]] = {
            "And": lambda *a: z3.And(*a) if a else z3.BoolVal(True, ctx),
            "Or": lambda *a: z3.Or(*a) if a else z3.BoolVal(False, ctx),
            "Not": z3.Not,
            "Implies": z3.Implies,
            "Iff": lambda a, b: a == b,
            "Xor": z3.Xor,
            "If": z3.If,
            "Distinct": z3.Distinct,
            "Abs": lambda a: z3.If(a >= 0, a, -a),
        }

    def compile(self, expr: str) -> z3.ExprRef:
        try:
            tree = ast.parse(normalize_expr(expr), mode="eval")
        except SyntaxError as exc:
            raise DSLError(f"syntax error in {expr!r}: {exc}") from exc
        return self._node(tree.body)

    def _node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return _const(node.value, self.ctx)
        if isinstance(node, ast.Name):
            if node.id in self.symbols:
                return self.symbols[node.id]
            if node.id in {"True", "False"}:
                return z3.BoolVal(node.id == "True", self.ctx)
            raise DSLError(f"undeclared symbol {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            operand = self._node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return z3.Not(operand)
        if isinstance(node, ast.BoolOp):
            values = [self._node(v) for v in node.values]
            return z3.And(*values) if isinstance(node.op, ast.And) else z3.Or(*values)
        if isinstance(node, ast.BinOp):
            left, right = self._node(node.left), self._node(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if z3.is_int(left) and z3.is_int(right):
                    left = z3.ToReal(left)
                return left / right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and 0 <= node.right.value <= 8):
                    raise DSLError("only small non-negative integer exponents are allowed")
                result = _const(1, self.ctx) if z3.is_int(left) else _const(1.0, self.ctx)
                for _ in range(node.right.value):
                    result = result * left
                return result
            if isinstance(op, ast.BitAnd):
                return z3.And(left, right)
            if isinstance(op, ast.BitOr):
                return z3.Or(left, right)
            if isinstance(op, ast.RShift):
                return z3.Implies(left, right)
        if isinstance(node, ast.Compare):
            parts = []
            left = self._node(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self._node(comp)
                if isinstance(op, ast.Eq):
                    parts.append(left == right)
                elif isinstance(op, ast.NotEq):
                    parts.append(left != right)
                elif isinstance(op, ast.Lt):
                    parts.append(left < right)
                elif isinstance(op, ast.LtE):
                    parts.append(left <= right)
                elif isinstance(op, ast.Gt):
                    parts.append(left > right)
                elif isinstance(op, ast.GtE):
                    parts.append(left >= right)
                else:
                    raise DSLError(f"unsupported comparison {type(op).__name__}")
                left = right
            return parts[0] if len(parts) == 1 else z3.And(*parts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if node.keywords:
                raise DSLError("keyword arguments are not allowed")
            if fname in {"ForAll", "Exists"}:
                *bound, body = node.args
                bound_vars = [self._node(b) for b in bound]
                quant = z3.ForAll if fname == "ForAll" else z3.Exists
                return quant(bound_vars, self._node(body))
            args = [self._node(a) for a in node.args]
            if fname in self.calls:
                return self.calls[fname](*args)
            if fname in self.symbols and isinstance(self.symbols[fname], z3.FuncDeclRef):
                return self.symbols[fname](*args)
            raise DSLError(f"unknown function {fname!r}")
        raise DSLError(f"unsupported syntax: {ast.dump(node)[:80]}")


def compile_spec(spec: FormalSpec) -> Compiled:
    """Declare symbols in a fresh context and compile all assertions."""
    ctx = z3.Context()
    sorts: dict[str, z3.SortRef] = {"Bool": z3.BoolSort(ctx), "Int": z3.IntSort(ctx), "Real": z3.RealSort(ctx)}
    for s in spec.sorts:
        if s in _SORTS:
            raise DSLError(f"sort name {s!r} is reserved")
        sorts[s] = z3.DeclareSort(s, ctx)
    symbols: dict[str, Any] = {}
    for name, sort in spec.vars.items():
        if sort not in sorts:
            raise DSLError(f"unknown sort {sort!r} for {name!r}")
        symbols[name] = z3.Const(name, sorts[sort])
    for name, signature in spec.functions.items():
        if len(signature) < 2 or any(s not in sorts for s in signature):
            raise DSLError(f"bad signature for function {name!r}: {signature}")
        symbols[name] = z3.Function(name, *[sorts[s] for s in signature])
    compiler = _Compiler(symbols, ctx)
    compiled = [(a.id, compiler.compile(a.expr), a.text or a.expr) for a in spec.assertions]
    for aid, expr, _ in compiled:
        if not z3.is_bool(expr):
            raise DSLError(f"assertion {aid!r} is not boolean")
    return Compiled(ctx=ctx, symbols=symbols, assertions=compiled, sorts=sorts)


@dataclass
class SolverOutcome:
    status: str  # "sat" | "unsat" | "unknown"
    core: list[str] = field(default_factory=list)
    model: dict[str, str] = field(default_factory=dict)


def _model_dict(model: z3.ModelRef, symbols: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, sym in symbols.items():
        if isinstance(sym, z3.FuncDeclRef):
            continue
        val = model.eval(sym, model_completion=False)
        if not z3.eq(val, sym):
            out[name] = str(val)
    return out


def check_consistency(spec: FormalSpec, timeout_ms: int = 2000) -> SolverOutcome:
    """SAT check with tracked assertions; on UNSAT returns a minimised unsat core."""
    comp = compile_spec(spec)
    solver = z3.Solver(ctx=comp.ctx)
    solver.set("timeout", timeout_ms)
    solver.set("unsat_core", True)
    trackers: dict[str, z3.BoolRef] = {}
    for aid, expr, _ in comp.assertions:
        tracker = z3.Bool(f"__track_{aid}", comp.ctx)
        trackers[aid] = tracker
        solver.assert_and_track(expr, tracker)
    result = solver.check()
    if result == z3.sat:
        return SolverOutcome("sat", model=_model_dict(solver.model(), comp.symbols))
    if result == z3.unknown:
        return SolverOutcome("unknown")
    core_ids = [str(c).removeprefix("__track_") for c in solver.unsat_core()]
    return SolverOutcome("unsat", core=_minimise_core(comp, core_ids, timeout_ms))


def _minimise_core(comp: Compiled, core_ids: list[str], timeout_ms: int) -> list[str]:
    """Deletion-based minimisation: drop each assertion if the rest stays UNSAT."""
    by_id = {aid: expr for aid, expr, _ in comp.assertions}
    core = list(core_ids)
    for aid in list(core):
        trial = [c for c in core if c != aid]
        if not trial:
            continue
        solver = z3.Solver(ctx=comp.ctx)
        solver.set("timeout", timeout_ms)
        solver.add(*[by_id[c] for c in trial])
        if solver.check() == z3.unsat:
            core = trial
    return core


@dataclass
class Determinacy:
    status: str  # "determined" | "underdetermined" | "inconsistent" | "unknown"
    values: list[str] = field(default_factory=list)


def check_determinacy(spec: FormalSpec, timeout_ms: int = 2000) -> Determinacy:
    """Is ``spec.target`` uniquely fixed by the assertions?

    Finds a model, then asks for a second model with a different target value.
    """
    if not spec.target:
        raise ValueError("spec has no target")
    comp = compile_spec(spec)
    target = _Compiler(comp.symbols, comp.ctx).compile(spec.target)
    solver = z3.Solver(ctx=comp.ctx)
    solver.set("timeout", timeout_ms)
    solver.add(*[e for _, e, _ in comp.assertions])
    first = solver.check()
    if first == z3.unsat:
        return Determinacy("inconsistent")
    if first == z3.unknown:
        return Determinacy("unknown")
    v1 = solver.model().eval(target, model_completion=True)
    solver.add(target != v1)
    second = solver.check()
    if second == z3.unsat:
        return Determinacy("determined", [str(v1)])
    if second == z3.unknown:
        return Determinacy("unknown", [str(v1)])
    v2 = solver.model().eval(target, model_completion=True)
    return Determinacy("underdetermined", [str(v1), str(v2)])


def check_entailment(spec: FormalSpec, timeout_ms: int = 2000) -> tuple[bool, dict[str, str]]:
    """Does ``spec.entailment_goal`` follow from the assertions? Returns (entailed, countermodel)."""
    if not spec.entailment_goal:
        raise ValueError("spec has no entailment_goal")
    comp = compile_spec(spec)
    goal = _Compiler(comp.symbols, comp.ctx).compile(spec.entailment_goal)
    solver = z3.Solver(ctx=comp.ctx)
    solver.set("timeout", timeout_ms)
    solver.add(*[e for _, e, _ in comp.assertions])
    solver.add(z3.Not(goal))
    result = solver.check()
    if result == z3.unsat:
        return True, {}
    if result == z3.sat:
        model = solver.model()
        counter = {str(d.name()): str(model[d]) for d in model.decls() if not str(d.name()).startswith("__")}
        return False, counter
    return False, {}


# ---------------------------------------------------------------------------
# Three-valued logics for self-referential definitions
# ---------------------------------------------------------------------------
T, U, F = 1.0, 0.5, 0.0
_TV_NAME = {T: "T", U: "U", F: "F"}


def _eval3(node: ast.AST, env: dict[str, float], logic: str) -> float:
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in {"True", "False"}:
            return T if node.id == "True" else F
        raise DSLError(f"undeclared symbol {node.id!r}")
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return T if node.value else F
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        return 1.0 - _eval3(node.operand, env, logic)
    if isinstance(node, ast.BoolOp):
        vals = [_eval3(v, env, logic) for v in node.values]
        return min(vals) if isinstance(node.op, ast.And) else max(vals)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_eval3(a, env, logic) for a in node.args]
        name = node.func.id
        if name == "Not":
            return 1.0 - args[0]
        if name == "And":
            return min(args)
        if name == "Or":
            return max(args)
        if name == "Implies":
            a, b = args
            return min(1.0, 1.0 - a + b) if logic == "lukasiewicz" else max(1.0 - a, b)
        if name == "Iff":
            a, b = args
            return 1.0 - abs(a - b) if logic == "lukasiewicz" else min(max(1.0 - a, b), max(1.0 - b, a))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        a = _eval3(node.left, env, logic)
        b = _eval3(node.comparators[0], env, logic)
        iff = 1.0 - abs(a - b) if logic == "lukasiewicz" else min(max(1.0 - a, b), max(1.0 - b, a))
        return iff if isinstance(node.ops[0], ast.Eq) else 1.0 - iff
    raise DSLError(f"unsupported three-valued construct: {ast.dump(node)[:80]}")


@dataclass
class FixedPointAnalysis:
    classical: list[dict[str, str]]
    three_valued: list[dict[str, str]]

    @property
    def is_paradox(self) -> bool:
        """No classical fixed point, but a gap (U) fixed point exists."""
        return not self.classical and bool(self.three_valued)


def kleene_fixed_points(definitions: dict[str, str], logic: str = "kleene") -> FixedPointAnalysis:
    """Enumerate fixed points of ``P := f(...)`` definitions in 2- and 3-valued logic.

    ``logic`` is ``"kleene"`` (strong Kleene K3) or ``"lukasiewicz"`` (Ł3).
    """
    if logic not in {"kleene", "lukasiewicz"}:
        raise ValueError("logic must be 'kleene' or 'lukasiewicz'")
    names = list(definitions)
    trees = {n: ast.parse(normalize_expr(expr), mode="eval").body for n, expr in definitions.items()}
    classical: list[dict[str, str]] = []
    three: list[dict[str, str]] = []
    for values in itertools.product((T, U, F), repeat=len(names)):
        env = dict(zip(names, values))
        if all(abs(_eval3(trees[n], env, logic) - env[n]) < 1e-9 for n in names):
            named = {n: _TV_NAME[v] for n, v in env.items()}
            (classical if U not in values else three).append(named)
    return FixedPointAnalysis(classical=classical, three_valued=three)
