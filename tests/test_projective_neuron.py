"""v0 of the projective-encoded neuron — addition only, single layer.

Three test classes:

* ``TestAlgebra`` — squaring rule, wraparound, non-commutativity.
* ``TestAnalyticGradients`` — analytic gradients agree with finite differences.
* ``TestGradientDescentOnAddition`` — random ``(a, +, b)`` problems,
  convergence to the analytical optimum (h=1, g=0); descent log written to
  ``tmp/diagnostics/projective_neuron_train.log``.

Inputs to the neuron are raw float arrays (Projective coefficients) — sympy
does not cross the neuron boundary.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from symbolic.projective import (
    IN_0, OP_0, IN_1, OUT_0,
    Projective, ProjectiveNeuron,
    basis_product_index, encode_problem,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "tmp" / "diagnostics" / "projective_neuron_train.log"

# v0 only trains on +; uses the planned-future op-enum value (1=−, 2=+, 3=*, 4=/).
OP_PLUS = 2


class TestAlgebra:

    def test_squaring_rule_each_index(self):
        for i in range(4):
            v = np.zeros(4); v[i] = 1.0
            squared = (Projective(v) * Projective(v)).coeffs
            expected = np.zeros(4); expected[(i + 1) % 4] = 1.0
            np.testing.assert_array_equal(squared, expected)

    def test_wraparound_top_squares_to_bottom(self):
        k3 = Projective(np.array([0, 0, 0, 1.0]))
        np.testing.assert_array_equal((k3 * k3).coeffs, [1, 0, 0, 0])

    def test_mixed_product_is_non_commutative(self):
        k0 = Projective(np.array([1.0, 0, 0, 0]))
        k1 = Projective(np.array([0, 1.0, 0, 0]))
        # k_0 · k_1: (2*0 + 1) mod 4 = 1
        np.testing.assert_array_equal((k0 * k1).coeffs, [0, 1, 0, 0])
        # k_1 · k_0: (2*1 + 0) mod 4 = 2
        np.testing.assert_array_equal((k1 * k0).coeffs, [0, 0, 1, 0])

    def test_basis_product_table_for_n4(self):
        # Spot-check the full 4x4 multiplication table.
        expected = {
            (0, 0): 1, (0, 1): 1, (0, 2): 2, (0, 3): 3,
            (1, 0): 2, (1, 1): 2, (1, 2): 0, (1, 3): 1,
            (2, 0): 0, (2, 1): 1, (2, 2): 3, (2, 3): 3,
            (3, 0): 2, (3, 1): 3, (3, 2): 0, (3, 3): 0,
        }
        for (i, j), k in expected.items():
            assert basis_product_index(i, j, 4) == k, f"({i},{j}) → {k}"


class TestAnalyticGradients:

    def test_match_finite_differences(self):
        rng = random.Random(0)
        eps = 1e-6
        for _ in range(8):
            a = rng.uniform(-5, 5)
            b = rng.uniform(-5, 5)
            target = a + b
            input_ = encode_problem(a, OP_PLUS, b)
            bias = np.array([rng.uniform(-1, 1) for _ in range(4)])

            neuron = ProjectiveNeuron(bias=Projective(bias.copy()))
            grad_analytic = neuron.gradients(input_, target)

            grad_numeric = np.zeros(4)
            for j in range(4):
                bp = bias.copy(); bp[j] += eps
                bm = bias.copy(); bm[j] -= eps
                pred_p = ProjectiveNeuron(bias=Projective(bp)).predict(input_)
                pred_m = ProjectiveNeuron(bias=Projective(bm)).predict(input_)
                loss_p = (pred_p - target) ** 2
                loss_m = (pred_m - target) ** 2
                grad_numeric[j] = (loss_p - loss_m) / (2 * eps)

            np.testing.assert_allclose(grad_analytic, grad_numeric, atol=1e-4)


class TestGradientDescentOnAddition:

    def test_converges_to_analytic_optimum(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20260502)

        neuron = ProjectiveNeuron.zeros(n=4)
        lr = 0.001
        n_steps = 5000

        header = "step\ta\tb\ttarget\tprediction\tloss\te\tf\tg\th"
        lines = [header]

        for step in range(n_steps):
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            target = a + b
            input_ = encode_problem(a, OP_PLUS, b)

            pred = neuron.step(input_, float(target), lr)
            loss = (pred - target) ** 2
            e, f, g, h = neuron.bias.coeffs
            lines.append(
                f"{step}\t{a}\t{b}\t{target}\t{pred}\t{loss}\t{e}\t{f}\t{g}\t{h}"
            )

        LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

        e, f, g, h = neuron.bias.coeffs

        # Analytical optimum for a + b: out_0 = h*(a+b) + g*b ⇒ h=1, g=0.
        # bias_0 (e) and bias_1 (f) never receive gradient — they index slots
        # that don't contribute to OUT_0 (and input[OUT_0]=0 zeroes the f path).
        assert abs(h - 1.0) < 0.05, f"h={h} (expected ≈ 1.0)"
        assert abs(g - 0.0) < 0.10, f"g={g} (expected ≈ 0.0)"
        assert e == 0.0, f"e={e} (expected exactly 0, no gradient path)"
        assert f == 0.0, f"f={f} (expected exactly 0, no gradient path)"
