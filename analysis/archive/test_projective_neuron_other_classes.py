"""Archived (whiteroom 2026-05-03): the 7 test classes from the old
test_projective_neuron.py that are NOT the one whiteroom test
(TestMultiplicationConvergenceTopology). Preserved verbatim.

These are *not* part of the active test path. To re-run them, you'd need
to restore the helpers they reference (notably ProjectiveNeuron, removed
from symbolic/projective.py at the same revert), and add them back to
tests/.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# Many of these reference ProjectiveNeuron, which was removed from
# symbolic/projective.py during whiteroom. The imports here will fail
# unless that class is restored.
from symbolic.projective import (
    IN_0, OP_0, IN_1, OUT_0,
    Projective, ProjectiveNeuron, SquaredInputProjectiveNeuron,
    basis_product_index, encode_problem,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = REPO_ROOT / "tmp" / "diagnostics" / "projective_neuron_train.log"
LOG_PATH_V1 = REPO_ROOT / "tmp" / "diagnostics" / "projective_neuron_v1_train.log"

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
        np.testing.assert_array_equal((k0 * k1).coeffs, [0, 1, 0, 0])
        np.testing.assert_array_equal((k1 * k0).coeffs, [0, 0, 1, 0])

    def test_basis_product_table_for_n4(self):
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
        assert abs(h - 1.0) < 0.05, f"h={h} (expected ≈ 1.0)"
        assert abs(g - 0.0) < 0.10, f"g={g} (expected ≈ 0.0)"
        assert e == 0.0, f"e={e} (expected exactly 0, no gradient path)"
        assert f == 0.0, f"f={f} (expected exactly 0, no gradient path)"


class TestGradientDescentOnMultiplication:
    """v1: (input · input) · bias on target a·b, with op slot = 0."""

    def test_converges_with_op_zero(self):
        LOG_PATH_V1.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20260502)

        neuron = SquaredInputProjectiveNeuron.zeros(n=4)
        lr = 1e-5
        n_steps = 20000

        header = "step\ta\tb\ttarget\tprediction\tloss\te\tf\tg\th"
        lines = [header]

        for step in range(n_steps):
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            target = a * b
            input_ = encode_problem(a, 0, b)

            pred = neuron.step(input_, float(target), lr)
            loss = (pred - target) ** 2
            e, f, g, h = neuron.bias.coeffs
            lines.append(
                f"{step}\t{a}\t{b}\t{target}\t{pred}\t{loss}\t{e}\t{f}\t{g}\t{h}"
            )

        LOG_PATH_V1.write_text("\n".join(lines) + "\n", encoding="utf-8")

        e, f, g, h = neuron.bias.coeffs
        assert e == 0.0, f"e={e} (expected exactly 0, no gradient path)"
        assert abs(f) < 0.01, f"f={f} (expected ≈ 0)"
        assert abs(g + 2 * h - 1.0) < 0.01, f"g+2h={g + 2*h} (expected ≈ 1.0)"
        assert abs(g - 0.2) < 0.05, f"g={g} (expected ≈ 0.2 from (1,2)-trajectory)"
        assert abs(h - 0.4) < 0.05, f"h={h} (expected ≈ 0.4 from (1,2)-trajectory)"

    def test_predicts_held_out_problems_exactly(self):
        rng = random.Random(20260502)
        neuron = SquaredInputProjectiveNeuron.zeros(n=4)

        for _ in range(20000):
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            input_ = encode_problem(a, 0, b)
            neuron.step(input_, float(a * b), 1e-5)

        held_out = [(3, 4), (-7, 9), (10, -10), (0, 5), (15, -3), (100, 100)]
        for a, b in held_out:
            input_ = encode_problem(a, 0, b)
            pred = neuron.predict(input_)
            target = a * b
            rel_err = abs(pred - target) / max(abs(target), 1.0)
            assert rel_err < 0.02, (
                f"({a}, {b}): predicted {pred}, target {target}, rel err {rel_err}"
            )


class TestHessianInGHPlane:

    def test_orthogonal_direction_has_zero_curvature(self):
        a, b = 3, 4
        target = a * b
        bias_at_optimum = np.array([0.0, 0.0, 0.2, 0.4])
        eps = 1e-4

        def loss(bias_arr):
            neuron = SquaredInputProjectiveNeuron(bias=Projective(bias_arr.copy()))
            pred = neuron.predict(encode_problem(a, 0, b))
            return (pred - target) ** 2

        H = np.zeros((2, 2))
        for ii, i in enumerate([2, 3]):
            for jj, j in enumerate([2, 3]):
                bpp = bias_at_optimum.copy(); bpp[i] += eps; bpp[j] += eps
                bpm = bias_at_optimum.copy(); bpm[i] += eps; bpm[j] -= eps
                bmp = bias_at_optimum.copy(); bmp[i] -= eps; bmp[j] += eps
                bmm = bias_at_optimum.copy(); bmm[i] -= eps; bmm[j] -= eps
                H[ii, jj] = (loss(bpp) - loss(bpm) - loss(bmp) + loss(bmm)) / (4 * eps**2)

        eigvals, eigvecs = np.linalg.eigh(H)

        assert abs(eigvals[0]) < 1e-3, f"smallest eigenvalue = {eigvals[0]:.6f}"
        expected_max = 10 * (a * b) ** 2
        assert abs(eigvals[1] - expected_max) / expected_max < 0.01

        z = eigvecs[:, 0]
        z_expected = np.array([2.0, -1.0]) / np.sqrt(5)
        assert abs(abs(float(np.dot(z, z_expected))) - 1.0) < 0.01

        nz = eigvecs[:, 1]
        nz_expected = np.array([1.0, 2.0]) / np.sqrt(5)
        assert abs(abs(float(np.dot(nz, nz_expected))) - 1.0) < 0.01


class TestSingleReadoutAtOutput0:

    def test_converges_init_independently_to_corner_optimum(self):
        inits = [
            ("zero",          np.zeros(4)),
            ("e_negative",    np.array([-1.0, 0.0, 0.0, 0.0])),
            ("all_active",    np.array([0.5, 0.5, 0.5, 0.5])),
            ("f_nonzero",     np.array([0.0, 0.7, 0.0, 0.0])),
            ("g_h_active",    np.array([0.0, 0.0, 0.5, 0.5])),
        ]
        results = []
        for label, init in inits:
            rng = random.Random(20260502)
            neuron = SquaredInputProjectiveNeuron(
                bias=Projective(init.copy()),
                readout_slot=0,
            )
            for _ in range(30000):
                a = rng.randint(-10, 10)
                b = rng.randint(-10, 10)
                neuron.step(encode_problem(a, 0, b), float(a * b), 5e-6)
            results.append((label, init, neuron.bias.coeffs.copy()))

        for label, init, final in results:
            assert abs(final[0] - 1.0) < 0.05
            assert abs(final[1] - init[1]) < 1e-6
            assert abs(final[2]) < 0.05
            assert abs(final[3]) < 0.05

        for label, init, final in results:
            for label2, init2, final2 in results:
                for idx, name in [(0, 'e'), (2, 'g'), (3, 'h')]:
                    assert abs(final[idx] - final2[idx]) < 0.05

    def test_hessian_at_optimum_only_zero_eigenvalue_along_f(self):
        bias_at_opt = np.array([1.0, 0.0, 0.0, 0.0])
        rng = random.Random(0)
        eps = 1e-4
        n_samples = 400

        H_sum = np.zeros((4, 4))
        for _ in range(n_samples):
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            target = a * b

            def loss(bias_arr, _a=a, _b=b, _t=target):
                neuron = SquaredInputProjectiveNeuron(
                    bias=Projective(bias_arr.copy()),
                    readout_slot=0,
                )
                return (neuron.predict(encode_problem(_a, 0, _b)) - _t) ** 2

            for i in range(4):
                for j in range(4):
                    bpp = bias_at_opt.copy(); bpp[i] += eps; bpp[j] += eps
                    bpm = bias_at_opt.copy(); bpm[i] += eps; bpm[j] -= eps
                    bmp = bias_at_opt.copy(); bmp[i] -= eps; bmp[j] += eps
                    bmm = bias_at_opt.copy(); bmm[i] -= eps; bmm[j] -= eps
                    H_sum[i, j] += (
                        loss(bpp) - loss(bpm) - loss(bmp) + loss(bmm)
                    ) / (4 * eps ** 2)

        H_avg = H_sum / n_samples
        eigvals, eigvecs = np.linalg.eigh(H_avg)

        assert abs(eigvals[0]) < 5.0
        z = np.abs(eigvecs[:, 0])
        assert z[1] > 0.99 and z[0] < 0.05 and z[2] < 0.05 and z[3] < 0.05
        assert all(v > 100 for v in eigvals[1:])


class TestOutput1AndOutput2Predicted:

    def test_output_1_e_f_degeneracy_with_g_dead(self):
        inits = [
            ("zero",         np.zeros(4)),
            ("e_perturbed",  np.array([0.5,  0.0,  0.3,  0.0])),
            ("f_perturbed",  np.array([0.0,  0.4,  0.0,  0.0])),
            ("g_nonzero",    np.array([0.0,  0.0, -0.6,  0.0])),
            ("ef_active",    np.array([-0.3, 0.4,  0.5,  0.2])),
        ]
        for label, init in inits:
            rng = random.Random(20260502)
            neuron = SquaredInputProjectiveNeuron(
                bias=Projective(init.copy()),
                readout_slot=1,
            )
            for _ in range(30000):
                a = rng.randint(-10, 10)
                b = rng.randint(-10, 10)
                neuron.step(encode_problem(a, 0, b), float(a * b), 5e-6)

            e, f, g, h = neuron.bias.coeffs
            e0, f0, g0, h0 = init

            assert abs(g - g0) < 1e-6
            assert abs(h) < 0.05
            assert abs(e + 2 * f - 1.0) < 0.05
            de, df_ = e - e0, f - f0
            if abs(de) > 1e-3:
                assert abs(df_ / de - 2.0) < 0.1
            t_pred = (1 - e0 - 2 * f0) / 5
            assert abs(de - t_pred) < 0.05

    def test_output_2_well_conditioned_with_h_dead(self):
        inits = [
            ("zero",            np.zeros(4)),
            ("e_negative",      np.array([-0.5, 0.0,  0.0,  0.0])),
            ("all_active",      np.array([0.5,  0.5,  0.5,  0.5])),
            ("h_nonzero",       np.array([0.0,  0.0,  0.0,  0.7])),
            ("efg_perturbed",   np.array([0.3, -0.4, -0.2,  0.0])),
        ]
        finals = []
        for label, init in inits:
            rng = random.Random(20260502)
            neuron = SquaredInputProjectiveNeuron(
                bias=Projective(init.copy()),
                readout_slot=2,
            )
            for _ in range(30000):
                a = rng.randint(-10, 10)
                b = rng.randint(-10, 10)
                neuron.step(encode_problem(a, 0, b), float(a * b), 5e-6)

            e, f, g, h = neuron.bias.coeffs
            e0, f0, g0, h0 = init

            assert abs(h - h0) < 1e-6
            assert abs(e) < 0.05
            assert abs(f) < 0.05
            assert abs(g - 1.0) < 0.05
            finals.append((label, (e, f, g, h)))

        for l1, (e1, f1, g1, _) in finals:
            for l2, (e2, f2, g2, _) in finals:
                for name, v1, v2 in [('e', e1, e2), ('f', f1, f2), ('g', g1, g2)]:
                    assert abs(v1 - v2) < 0.05
