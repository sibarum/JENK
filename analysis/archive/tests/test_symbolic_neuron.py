"""Random arithmetic stress test for SymbolicNeuron.

Each step we draw a random problem  a op b  with

    a, b in [-10, 10]   and   op in {+, -, *, /, **}

evaluate it with sympy to get the expected value, feed ``a`` to the
neuron as input, and let the neuron's all-in-one step do feedforward +
additive update.  The full descent trajectory is written to

    tmp/diagnostics/symbolic_neuron_train.log

so the evolution of the neuron's expression can be inspected.

Note on the update rule
-----------------------
SymbolicNeuron uses  expr <- expr + lr * (actual - expected)  — the
*additive* sign convention James asked for.  Because ``actual`` and
``expected`` are both numeric here, every delta is a rational scalar:
the neuron can only ever learn the intercept of ``expr``, not the
slope.  The log makes that visible.
"""

from __future__ import annotations

import random
from pathlib import Path

import sympy as sp

from symbolic.neuron import SymbolicNeuron


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "tmp" / "diagnostics" / "symbolic_neuron_train.log"

OPS = ['+', '-', '*', '/', '**']


def _random_problem(rng: random.Random):
    """Sample (a, op, b, expected) or return None for non-finite results."""
    a = rng.randint(-10, 10)
    b = rng.randint(-10, 10)
    op = rng.choice(OPS)
    try:
        expected = sp.sympify(f"({a}) {op} ({b})")
    except (ZeroDivisionError, ValueError, TypeError):
        return None
    if expected in (sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    if expected.is_real is False:
        return None
    return a, op, b, expected


class TestRandomArithmeticTraining:

    def test_logs_descent_trajectory(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(20260502)
        x = sp.Symbol('x')
        neuron = SymbolicNeuron(expr=x, var=x)
        lr = sp.Rational(1, 100)
        n_steps = 50

        header = "step\ta\top\tb\texpected\tinput\tactual\tloss\tdelta\texpr_after"
        lines = [header]

        completed = 0
        attempts = 0
        while completed < n_steps:
            attempts += 1
            assert attempts < 10 * n_steps, "too many degenerate samples"
            problem = _random_problem(rng)
            if problem is None:
                continue
            a, op, b, expected = problem
            input_value = sp.Integer(a)

            expr_before = neuron.expr
            actual = neuron.step(input_value, expected, lr)
            loss = sp.sympify(actual) - sp.sympify(expected)
            delta = neuron.expr - expr_before

            lines.append(
                f"{completed}\t{a}\t{op}\t{b}\t{expected}\t"
                f"{input_value}\t{actual}\t{loss}\t{delta}\t{neuron.expr}"
            )
            completed += 1

        LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Sanity checks on what we logged.
        assert LOG_PATH.exists()
        assert len(lines) == n_steps + 1

        # Every delta should be a rational scalar (no x dependence) under
        # this additive update rule with numeric inputs.
        for line in lines[1:]:
            delta_str = line.split('\t')[8]
            delta_expr = sp.sympify(delta_str)
            assert x not in delta_expr.free_symbols, (
                f"unexpected x-dependence in delta: {delta_str}"
            )

        # Final expression should still be of the form  x + c  for some
        # rational c, since only the intercept can move.
        final = sp.expand(neuron.expr - x)
        assert x not in final.free_symbols, (
            f"final expr {neuron.expr} drifted off the affine x + c family"
        )
