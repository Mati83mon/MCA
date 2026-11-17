"""Z3 / SMT layer: DSL safety, consistency, unsat cores, determinacy, entailment, 3-valued logic."""

from __future__ import annotations

import pytest

from mca.config import MCAConfig
from mca.core.formalizer import RuleBasedFormalizer
from mca.core.logic_engine import LogicEngine
from mca.telemetry.schemas import LogicStatus
from mca.utils.z3_helpers import (
    Assertion,
    DSLError,
    FormalSpec,
    check_consistency,
    check_determinacy,
    check_entailment,
    compile_spec,
    kleene_fixed_points,
)


def spec(vars, *exprs, **kw):
    return FormalSpec(vars=vars, assertions=[Assertion(id=f"a{i}", expr=e) for i, e in enumerate(exprs)], **kw)


def test_sat_returns_model():
    out = check_consistency(spec({"x": "Int"}, "x > 2", "x < 4"))
    assert out.status == "sat" and out.model["x"] == "3"


def test_unsat_core_is_minimal():
    s = spec({"x": "Real", "y": "Real"}, "y == 1", "x > 0", "x < 0", "y > 0")
    out = check_consistency(s)
    assert out.status == "unsat"
    assert sorted(out.core) == ["a1", "a2"]


@pytest.mark.parametrize("expr", ["x * x == -9", "And(x > 0, x < 0)", "x * 0 == 7"])
def test_real_arithmetic_contradictions(expr):
    assert check_consistency(spec({"x": "Real"}, expr)).status == "unsat"


def test_no_largest_natural_number():
    s = spec({"n": "Int", "m": "Int"}, "n >= 0", "ForAll(m, Implies(m >= 0, m <= n))")
    assert check_consistency(s).status == "unsat"


def test_determinacy():
    under = check_determinacy(spec({"x": "Real", "y": "Real"}, "x + y == 10", target="x"))
    assert under.status == "underdetermined" and len(set(under.values)) == 2
    det = check_determinacy(spec({"x": "Real", "y": "Real"}, "x + y == 10", "x - y == 2", target="x"))
    assert det.status == "determined" and det.values == ["6"]


def test_converse_is_not_entailed():
    s = FormalSpec(sorts=["Obj"], vars={"x": "Obj"}, functions={"A": ["Obj", "Bool"], "B": ["Obj", "Bool"]},
                   assertions=[Assertion(id="p", expr="ForAll(x, Implies(A(x), B(x)))")],
                   entailment_goal="ForAll(x, Implies(B(x), A(x)))")
    entailed, counter = check_entailment(s)
    assert not entailed and counter


def test_diagonal_definition_is_unsat():
    s = FormalSpec(sorts=["Obj"], vars={"d": "Obj", "x": "Obj"}, functions={"R": ["Obj", "Obj", "Bool"]},
                   assertions=[Assertion(id="diag", expr="ForAll(x, R(d, x) == Not(R(x, x)))")])
    assert check_consistency(s).status == "unsat"


def test_kleene_liar_and_truth_teller():
    liar = kleene_fixed_points({"L": "Not(L)"})
    assert liar.is_paradox and liar.three_valued == [{"L": "U"}]
    truth_teller = kleene_fixed_points({"T": "T"})
    assert not truth_teller.is_paradox and len(truth_teller.classical) == 2
    cyclic = kleene_fixed_points({"A": "B", "B": "Not(A)"}, logic="lukasiewicz")
    assert cyclic.is_paradox and cyclic.three_valued == [{"A": "U", "B": "U"}]


@pytest.mark.parametrize("bad", ["__import__('os')", "x.__class__", "open('f')", "x ** y", "lambda: 1"])
def test_dsl_rejects_unsafe_syntax(bad):
    with pytest.raises(DSLError):
        compile_spec(spec({"x": "Int", "y": "Int"}, bad))


def test_logic_engine_priorities():
    engine = LogicEngine(MCAConfig())
    liar = RuleBasedFormalizer().formalize("Zdanie L brzmi: „Zdanie L jest fałszywe”. Czy L jest prawdziwe?")
    assert engine.check(liar).status is LogicStatus.PARADOX_SELF_REF
    mixed = [spec({"x": "Int"}, "x > 1"), spec({"x": "Int"}, "x > 1", "x < 0")]
    report = engine.check(mixed)
    assert report.status is LogicStatus.UNSAT and report.unsat_core
    assert engine.check([]).status is LogicStatus.NOT_APPLICABLE
    engine.close()


async def test_logic_engine_async():
    engine = LogicEngine(MCAConfig())
    specs = RuleBasedFormalizer().formalize("Liczby rzeczywiste x i y spełniają x + y = 10. Ile wynosi x?")
    report = await engine.acheck(specs)
    assert report.status is LogicStatus.UNDERDETERMINED
    engine.close()


def test_ill_formed_llm_formula_is_unknown_not_a_crash():
    # Kaggle v14 crashed on an LLM spec calling If() with two arguments.
    from mca.config import MCAConfig
    from mca.core.logic_engine import LogicEngine
    from mca.telemetry.schemas import LogicStatus
    from mca.utils.z3_helpers import FormalSpec

    engine = LogicEngine(MCAConfig(nli_backend="lexical", embedding_backend="hashing"))
    try:
        for expr in ("If(x > 0, x)", "ForAll(x, If(x > 0))", "Abs(x, x) == 1"):
            spec = FormalSpec.model_validate({"label": "llm.formal", "vars": {"x": "Real"},
                                              "assertions": [{"id": "a1", "expr": expr}], "target": "x"})
            report = engine.check([spec])
            assert report.status is LogicStatus.UNKNOWN, (expr, report)
    finally:
        engine.close()
