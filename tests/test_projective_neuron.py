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
    N_V2,
    Projective, ProjectiveNeuron, SquaredInputProjectiveNeuron,
    basis_product_index, encode_problem, encode_problem_v2,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
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
            input_ = encode_problem(a, 0, b)  # op=0 for v1 single-op multiplication

            pred = neuron.step(input_, float(target), lr)
            loss = (pred - target) ** 2
            e, f, g, h = neuron.bias.coeffs
            lines.append(
                f"{step}\t{a}\t{b}\t{target}\t{pred}\t{loss}\t{e}\t{f}\t{g}\t{h}"
            )

        LOG_PATH_V1.write_text("\n".join(lines) + "\n", encoding="utf-8")

        e, f, g, h = neuron.bias.coeffs

        # Analytical optimum for a*b with op=0:
        #   output_3 = (g + 2h)·a·b + f·b² ⇒ g + 2h = 1, f = 0.
        # From init (g=0, h=0), the gradient ratio dL/dh : dL/dg = 2 : 1, so
        # the trajectory is (g, h) = (t, 2t) and convergence lands at t = 0.2:
        # g ≈ 0.2, h ≈ 0.4.  e is structurally dead (no path to OUT_0 via j=0).
        assert e == 0.0, f"e={e} (expected exactly 0, no gradient path)"
        assert abs(f) < 0.01, f"f={f} (expected ≈ 0)"
        assert abs(g + 2 * h - 1.0) < 0.01, f"g+2h={g + 2*h} (expected ≈ 1.0)"
        assert abs(g - 0.2) < 0.05, f"g={g} (expected ≈ 0.2 from (1,2)-trajectory)"
        assert abs(h - 0.4) < 0.05, f"h={h} (expected ≈ 0.4 from (1,2)-trajectory)"

    def test_predicts_held_out_problems_exactly(self):
        """After training, prediction matches a*b on fresh problems."""
        rng = random.Random(20260502)
        neuron = SquaredInputProjectiveNeuron.zeros(n=4)

        for _ in range(20000):
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            input_ = encode_problem(a, 0, b)
            neuron.step(input_, float(a * b), 1e-5)

        # Fresh problems, including ones outside training range.
        held_out = [(3, 4), (-7, 9), (10, -10), (0, 5), (15, -3), (100, 100)]
        for a, b in held_out:
            input_ = encode_problem(a, 0, b)
            pred = neuron.predict(input_)
            target = a * b
            rel_err = abs(pred - target) / max(abs(target), 1.0)
            assert rel_err < 0.02, (
                f"({a}, {b}): predicted {pred}, target {target}, rel err {rel_err}"
            )


class TestMultiplicationConvergenceTopology:
    """Init-sweep: distinguish 'preferred operating point' vs 'degenerate minimum'.

    Per-sample gradient on (g, h) is (a·b, 2·a·b)·2·residual.  The direction
    in (g, h) space is therefore always exactly (1, 2), regardless of input
    or current params — a consequence of h having two (i,j) pairs routing to
    OUT_0 with hidden coeff a·b ((0,3) and (2,3)), and g having one ((2,2)).

    From any init (g0, h0), the trajectory in (g, h) space is along (1, 2);
    convergence on the manifold g+2h=1 happens at  t* = (1 − g0 − 2h0)/5.

    Predictions:
      * all inits → same manifold  g + 2h ≈ 1
      * different inits → different (g, h) points  (init-dependent landing)
      * displacement (Δg, Δh) ∝ (1, 2)  for all inits

    A "preferred operating point" hypothesis would predict the same (g, h)
    regardless of init — which this test should falsify.
    """

    def test_landing_is_init_dependent_on_a_shared_manifold(self):
        log_path = REPO_ROOT / "tmp" / "diagnostics" / "projective_neuron_v1_init_sweep.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        inits = [
            ('zero',          np.zeros(4)),
            ('balanced',      np.array([0.0, 0.0,  0.5,  0.5])),
            ('flipped',       np.array([0.0, 0.0, -0.3,  0.7])),
            ('h-negative',    np.array([0.0, 0.0,  0.4, -0.1])),
        ]
        results = []
        log_lines = [
            "label\tinit_g\tinit_h\tfinal_g\tfinal_h\t"
            "g_plus_2h\tdelta_g\tdelta_h\tratio_dh_dg\tt_predicted"
        ]

        for label, init in inits:
            rng = random.Random(20260502)
            neuron = SquaredInputProjectiveNeuron(bias=Projective(init.copy()))
            for _ in range(10000):
                a = rng.randint(-10, 10)
                b = rng.randint(-10, 10)
                neuron.step(encode_problem(a, 0, b), float(a * b), 1e-5)

            g_final = float(neuron.bias.coeffs[2])
            h_final = float(neuron.bias.coeffs[3])
            g_plus_2h = g_final + 2 * h_final
            dg = g_final - init[2]
            dh = h_final - init[3]
            ratio = dh / dg if abs(dg) > 1e-3 else None
            t_pred = (1.0 - init[2] - 2 * init[3]) / 5.0

            results.append({
                'label': label, 'init': (init[2], init[3]),
                'final': (g_final, h_final),
                'g_plus_2h': g_plus_2h, 'dg': dg, 'dh': dh,
                'ratio': ratio, 't_pred': t_pred,
            })
            log_lines.append(
                f"{label}\t{init[2]}\t{init[3]}\t{g_final}\t{h_final}\t"
                f"{g_plus_2h}\t{dg}\t{dh}\t{ratio}\t{t_pred}"
            )

        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        # 1. All inits land on the same manifold g + 2h = 1.
        for r in results:
            assert abs(r['g_plus_2h'] - 1.0) < 0.01, (
                f"{r['label']}: g+2h = {r['g_plus_2h']:.4f} (expected ≈ 1)"
            )

        # 2. Displacement is always along (1, 2): Δh / Δg ≈ 2.
        for r in results:
            if r['ratio'] is not None:
                assert abs(r['ratio'] - 2.0) < 0.05, (
                    f"{r['label']}: Δh/Δg = {r['ratio']:.4f} (expected ≈ 2)"
                )

        # 3. Different inits → distinct convergence points (degenerate min,
        #    not a preferred operating point).
        unique = {(round(r['final'][0], 2), round(r['final'][1], 2))
                  for r in results}
        assert len(unique) >= 3, (
            f"only {len(unique)} distinct (g, h) convergence points across "
            f"{len(results)} inits — would suggest preferred operating point"
        )

        # 4. Predicted-vs-actual displacement t* matches.
        for r in results:
            t_actual = r['dg']  # since trajectory is (t, 2t), dg = t
            assert abs(t_actual - r['t_pred']) < 0.01, (
                f"{r['label']}: t actual {t_actual:.4f} vs predicted {r['t_pred']:.4f}"
            )


class TestHessianInGHPlane:
    """Empirical Hessian eigendecomposition on (g, h) at the v1 optimum.

    Per-sample Hessian factors as  2·(a·b)² · (1, 2)(1, 2)ᵀ  — rank 1.
    Eigenvalues: ``10·(a·b)²`` along ``(1, 2)/√5``, and exactly ``0``
    along ``(2, −1)/√5``.  Holds on every sample (not just in
    expectation), so the orthogonal direction has *no curvature*.
    """

    def test_orthogonal_direction_has_zero_curvature(self):
        a, b = 3, 4  # arbitrary nonzero
        target = a * b
        bias_at_optimum = np.array([0.0, 0.0, 0.2, 0.4])
        eps = 1e-4

        def loss(bias_arr):
            neuron = SquaredInputProjectiveNeuron(bias=Projective(bias_arr.copy()))
            pred = neuron.predict(encode_problem(a, 0, b))
            return (pred - target) ** 2

        # Numerical Hessian on (bias[2], bias[3]) = (g, h).
        H = np.zeros((2, 2))
        for ii, i in enumerate([2, 3]):
            for jj, j in enumerate([2, 3]):
                bpp = bias_at_optimum.copy(); bpp[i] += eps; bpp[j] += eps
                bpm = bias_at_optimum.copy(); bpm[i] += eps; bpm[j] -= eps
                bmp = bias_at_optimum.copy(); bmp[i] -= eps; bmp[j] += eps
                bmm = bias_at_optimum.copy(); bmm[i] -= eps; bmm[j] -= eps
                H[ii, jj] = (loss(bpp) - loss(bpm) - loss(bmp) + loss(bmm)) / (4 * eps**2)

        eigvals, eigvecs = np.linalg.eigh(H)  # ascending order

        # Smallest eigenvalue ≈ 0 (zero curvature in orthogonal direction).
        assert abs(eigvals[0]) < 1e-3, (
            f"smallest eigenvalue = {eigvals[0]:.6f} (expected ≈ 0)"
        )
        # Largest eigenvalue ≈ 10·(a·b)² = 1440.
        expected_max = 10 * (a * b) ** 2
        assert abs(eigvals[1] - expected_max) / expected_max < 0.01, (
            f"largest eigenvalue = {eigvals[1]:.4f} (expected ≈ {expected_max})"
        )

        # Zero-eigenvector aligned with (2, -1)/√5.
        z = eigvecs[:, 0]
        z_expected = np.array([2.0, -1.0]) / np.sqrt(5)
        assert abs(abs(float(np.dot(z, z_expected))) - 1.0) < 0.01, (
            f"zero-eigenvector {z} not aligned with (2, -1)/√5"
        )
        # Nonzero-eigenvector aligned with (1, 2)/√5.
        nz = eigvecs[:, 1]
        nz_expected = np.array([1.0, 2.0]) / np.sqrt(5)
        assert abs(abs(float(np.dot(nz, nz_expected))) - 1.0) < 0.01, (
            f"nonzero-eigenvector {nz} not aligned with (1, 2)/√5"
        )


class TestSingleReadoutAtOutput0:
    """v1.5: same architecture, predict from output[0] (k_0) instead of k_3.

    Cleanest separation of "the architecture has a degeneracy" vs "this
    particular readout has a degeneracy".  output[3]'s path-count in
    (g, h) is rigidly (1, 2) per sample → 1-D minimum.  output[0]'s
    path-count in (e, g, h) is (a·b, a²+b², b²) per sample, which varies
    with the input — so the gradient spans 3D over samples and the
    init-frozen orthogonal axis from v1 should not survive the slot swap.

    output[0] = e·hidden[2] + g·(hidden[1]+hidden[3]) + h·hidden[3]
              = e·a·b + g·(a²+b²) + h·b²       (with op=0)

    Optimum for target a·b: (e=1, f=stuck-at-init, g=0, h=0).  ``f`` is
    structurally dead at output[0] — its path-count component is 0 for
    every sample regardless of input.  Watch for the corner-solution
    pathologies James flagged: sign-flip from a wrong-side init, slow
    convergence in the smaller-eigenvalue direction.
    """

    def test_converges_init_independently_to_corner_optimum(self):
        inits = [
            ("zero",          np.zeros(4)),
            ("e_negative",    np.array([-1.0, 0.0, 0.0, 0.0])),  # wrong side, must sign-flip
            ("all_active",    np.array([0.5, 0.5, 0.5, 0.5])),
            ("f_nonzero",     np.array([0.0, 0.7, 0.0, 0.0])),   # f should stay at 0.7
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
            assert abs(final[0] - 1.0) < 0.05, (
                f"{label}: e converged to {final[0]:.4f} (expected ≈ 1.0); "
                "init {init}"
            )
            assert abs(final[1] - init[1]) < 1e-6, (
                f"{label}: f drifted from init {init[1]} to {final[1]} "
                "(expected structurally frozen)"
            )
            assert abs(final[2]) < 0.05, (
                f"{label}: g converged to {final[2]:.4f} (expected ≈ 0)"
            )
            assert abs(final[3]) < 0.05, (
                f"{label}: h converged to {final[3]:.4f} (expected ≈ 0)"
            )

        # Init-independence: across the diverse inits, (e, g, h) all land at
        # the same point (1, 0, 0) — only f differs (each holds its init).
        for label, init, final in results:
            for label2, init2, final2 in results:
                for idx, name in [(0, 'e'), (2, 'g'), (3, 'h')]:
                    assert abs(final[idx] - final2[idx]) < 0.05, (
                        f"{name} differs between {label} ({final[idx]:.4f}) "
                        f"and {label2} ({final2[idx]:.4f}) — init dependence"
                    )

    def test_hessian_at_optimum_only_zero_eigenvalue_along_f(self):
        """Full 4D expected Hessian at (e=1, f=0, g=0, h=0).

        Per sample, the Hessian is rank 1 (outer product of the path-count
        vector with itself).  Averaged across samples with varying (a, b),
        rank grows to 3 — only the f direction stays zero, because the
        path-count's f component is identically 0 for every sample.
        """
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

        # Smallest eigenvalue ≈ 0 (with sample-noise tolerance).
        assert abs(eigvals[0]) < 5.0, (
            f"smallest eigenvalue = {eigvals[0]:.4f} (expected ≈ 0)"
        )
        # Aligned with f-axis.
        z = np.abs(eigvecs[:, 0])
        assert z[1] > 0.99 and z[0] < 0.05 and z[2] < 0.05 and z[3] < 0.05, (
            f"smallest-eigenvalue eigenvector {eigvecs[:, 0]} not aligned with f-axis"
        )
        # Other 3 eigenvalues genuinely nonzero (full rank in (e, g, h)).
        assert all(v > 100 for v in eigvals[1:]), (
            f"expected 3 nonzero eigenvalues, got {eigvals[1:]}"
        )


class TestOutput1AndOutput2Predicted:
    """Closure: predict, then confirm, the topology of the remaining readouts.

    From the path-count table (with op=0 hidden = (a·b, a², a·b, b²)):

      * output[1]:  path-count (a·b, 2·a·b, 0, a²)
        - (e, f) plane: (1, 2)-rigid per sample — *same* degeneracy shape
          as output[3]'s (g, h), but in a different coordinate plane.
        - g: structurally dead (path-count component 0 every sample).
        - h: independent direction, converges to 0.
        - Optimum: e + 2f = 1, h = 0, g = g₀.  Closed form
            t* = (1 − e₀ − 2·f₀)/5,    Δe = t*,  Δf = 2·t*.

      * output[2]:  path-count (a²+b², a², a·b, 0)
        - (e, f, g) directions vary independently with (a, b) — 3D
          well-conditioned (same shape as output[0]).
        - h: structurally dead (path-count component 0 every sample).
        - Optimum: (e, f, g) = (0, 0, 1) regardless of init; h = h₀.

    Each output kills a different bias coefficient cyclically:
      output[0]→f, output[1]→g, output[2]→h, output[3]→e.
    """

    def test_output_1_e_f_degeneracy_with_g_dead(self):
        inits = [
            ("zero",         np.zeros(4)),
            ("e_perturbed",  np.array([0.5,  0.0,  0.3,  0.0])),
            ("f_perturbed",  np.array([0.0,  0.4,  0.0,  0.0])),
            ("g_nonzero",    np.array([0.0,  0.0, -0.6,  0.0])),  # g should stay at -0.6
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

            # g structurally dead.
            assert abs(g - g0) < 1e-6, (
                f"{label}: g={g:.4f} drifted from init {g0} (expected dead)"
            )
            # h converges to 0 (independent live direction).
            assert abs(h) < 0.05, f"{label}: h={h:.4f} (expected ≈ 0)"
            # Manifold e + 2f = 1.
            assert abs(e + 2 * f - 1.0) < 0.05, (
                f"{label}: e+2f = {e + 2*f:.4f} (expected ≈ 1)"
            )
            # Trajectory along (1, 2): Δf / Δe = 2.
            de, df_ = e - e0, f - f0
            if abs(de) > 1e-3:
                assert abs(df_ / de - 2.0) < 0.1, (
                    f"{label}: Δf/Δe = {df_/de:.4f} (expected ≈ 2)"
                )
            # Closed-form  t* = (1 − e₀ − 2·f₀)/5.
            t_pred = (1 - e0 - 2 * f0) / 5
            assert abs(de - t_pred) < 0.05, (
                f"{label}: Δe = {de:.4f} vs predicted t* = {t_pred:.4f}"
            )

    def test_output_2_well_conditioned_with_h_dead(self):
        inits = [
            ("zero",            np.zeros(4)),
            ("e_negative",      np.array([-0.5, 0.0,  0.0,  0.0])),
            ("all_active",      np.array([0.5,  0.5,  0.5,  0.5])),
            ("h_nonzero",       np.array([0.0,  0.0,  0.0,  0.7])),  # h stays
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

            # h structurally dead.
            assert abs(h - h0) < 1e-6, (
                f"{label}: h={h:.4f} drifted from init {h0} (expected dead)"
            )
            # (e, f, g) → (0, 0, 1) regardless of init.
            assert abs(e) < 0.05, f"{label}: e={e:.4f} (expected ≈ 0)"
            assert abs(f) < 0.05, f"{label}: f={f:.4f} (expected ≈ 0)"
            assert abs(g - 1.0) < 0.05, f"{label}: g={g:.4f} (expected ≈ 1)"
            finals.append((label, (e, f, g, h)))

        # Init-independence in (e, f, g): all converge to same point.
        for l1, (e1, f1, g1, _) in finals:
            for l2, (e2, f2, g2, _) in finals:
                for name, v1, v2 in [('e', e1, e2), ('f', f1, f2), ('g', g1, g2)]:
                    assert abs(v1 - v2) < 0.05, (
                        f"{name}: {l1}={v1:.4f} vs {l2}={v2:.4f} (init-dependence)"
                    )
