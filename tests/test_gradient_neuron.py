"""Proper gradient descent on random arithmetic.

Companion to ``test_symbolic_neuron.py``.  That test exercised the
literal additive update James first asked for and showed the parameter
diverging.  This test wires up the same random-arithmetic flavour to
GradientNeuron, which uses

    L = (actual - expected) ** 2

with the standard  param <- param - lr * dL/dparam  update, computed via
sympy.diff.

Setup
-----
* One ``op`` and one ``b`` are sampled once per run, so the target
    f(a) = a op b
  is a real function of the single input ``a``.  Without that, every
  step would be a different problem and there would be nothing to
  converge to.
* Ops are restricted to ``{+, -, *, /}`` so the target is linear in ``a``
  and the neuron's affine form  w*x + bias  can fit it exactly.  ``**``
  is excluded because  a ** b  is not linear in ``a`` (would need a
  richer expression to fit).
* ``b`` is forbidden from being 0 when ``op == '/'``.
* Floats are used for parameter values: with Rationals, denominators
  grow by a factor of ~1/lr per step and become unworkable after a few
  hundred steps.  ``sp.diff`` is purely symbolic either way.

Output
------
Descent trajectory is logged to
    tmp/diagnostics/gradient_neuron_train.log
"""

from __future__ import annotations

import random
from pathlib import Path

import sympy as sp

from symbolic.neuron import GradientNeuron


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "tmp" / "diagnostics" / "gradient_neuron_train.log"

LINEAR_OPS = ['+', '-', '*', '/']


def _analytic_optimum(op: str, b):
    """Closed-form (w*, bias*) that makes  w*a + bias = a op b  exactly."""
    bs = sp.sympify(b)
    if op == '+':
        return sp.Integer(1), bs
    if op == '-':
        return sp.Integer(1), -bs
    if op == '*':
        return bs, sp.Integer(0)
    if op == '/':
        return sp.Rational(1, b), sp.Integer(0)
    raise ValueError(f"unsupported op: {op}")


class TestGradientDescentOnArithmetic:

    def test_converges_to_analytic_optimum(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20260502)

        # Fix one (op, b) per run so the target is a real learnable function.
        op = rng.choice(LINEAR_OPS)
        b_choices = [n for n in range(-10, 11) if n != 0] if op == '/' \
            else list(range(-10, 11))
        b = rng.choice(b_choices)

        x, w, bias = sp.symbols('x w bias')
        neuron = GradientNeuron(
            expr=w * x + bias,
            var=x,
            params={w: 0.0, bias: 0.0},
        )
        lr = 0.003
        n_steps = 2000

        header = ("step\ta\tb\top\texpected\tactual\tloss\t"
                  "w\tbias\tdL_dw\tdL_dbias")
        lines = [header]

        for step in range(n_steps):
            a = rng.randint(-10, 10)
            expected = sp.sympify(f"({a}) {op} ({b})")
            actual = neuron.feedforward(a)
            grads = neuron.gradients(a, expected)
            neuron.update(grads, lr)
            loss = float((sp.sympify(actual) - sp.sympify(expected)) ** 2)
            lines.append(
                f"{step}\t{a}\t{b}\t{op}\t{expected}\t{actual}\t{loss}\t"
                f"{neuron.params[w]}\t{neuron.params[bias]}\t"
                f"{grads[w]}\t{grads[bias]}"
            )

        LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Assert convergence to the analytic optimum.
        w_star, bias_star = _analytic_optimum(op, b)
        w_final = float(neuron.params[w])
        bias_final = float(neuron.params[bias])
        w_err = abs(w_final - float(w_star))
        bias_err = abs(bias_final - float(bias_star))

        assert w_err < 0.05, (
            f"op={op!r} b={b}: w converged to {w_final}, "
            f"expected {w_star} (err {w_err})"
        )
        assert bias_err < 0.5, (
            f"op={op!r} b={b}: bias converged to {bias_final}, "
            f"expected {bias_star} (err {bias_err})"
        )

    def test_converges_for_each_linear_op(self):
        """Sweep all four linear ops with a fixed b to make sure each fits."""
        results = []
        for op in LINEAR_OPS:
            b = 4 if op != '/' else 5  # arbitrary nonzero choices
            x, w, bias = sp.symbols('x w bias')
            neuron = GradientNeuron(
                expr=w * x + bias,
                var=x,
                params={w: 0.0, bias: 0.0},
            )
            rng = random.Random(0)
            for _ in range(2000):
                a = rng.randint(-10, 10)
                expected = sp.sympify(f"({a}) {op} ({b})")
                neuron.step(a, expected, 0.003)

            w_star, bias_star = _analytic_optimum(op, b)
            w_err = abs(float(neuron.params[w]) - float(w_star))
            bias_err = abs(float(neuron.params[bias]) - float(bias_star))
            results.append((op, b, w_err, bias_err))
            assert w_err < 0.05, f"op={op!r}: w err {w_err}"
            assert bias_err < 0.5, f"op={op!r}: bias err {bias_err}"
