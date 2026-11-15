from __future__ import annotations

import pytest

from mca.core.oracles import NumberTheoryOracle, PythonSemanticsOracle, UnsafeCode, is_prime, run_snippet
from mca.telemetry.schemas import PremiseStatus


@pytest.mark.parametrize(("text", "status"), [
    ("Dlaczego all([]) w Pythonie zwraca False?", PremiseStatus.CONTRADICTED),
    ("all([]) w Pythonie zwraca True", PremiseStatus.SUPPORTED),
    ("Dlaczego przypisanie t[0] = 7 zmienia pierwszy element krotki t = (1, 2)?", PremiseStatus.CONTRADICTED),
    ("Przypisanie t[0] = 7 zmienia pierwszy element listy t = [1, 2]", PremiseStatus.SUPPORTED),
])
def test_python_oracle(text, status):
    verdict = PythonSemanticsOracle().evaluate(text)
    assert verdict is not None and verdict.status is status


@pytest.mark.parametrize("code", ["__import__('os')", "open('x')", "().__class__", "exec('1')", "2 ** 999999"])
def test_sandbox_blocks(code):
    with pytest.raises(UnsafeCode):
        run_snippet(code)


def test_number_theory():
    oracle = NumberTheoryOracle()
    assert oracle.evaluate("liczba pierwsza 1").status is PremiseStatus.CONTRADICTED
    assert oracle.evaluate("7 jest liczbą pierwszą").status is PremiseStatus.SUPPORTED
    assert oracle.evaluate("9 jest liczbą parzystą").status is PremiseStatus.CONTRADICTED
    assert [n for n in range(20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
