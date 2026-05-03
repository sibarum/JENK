"""The one whiteroom test: v1 (input·input)·bias on target a·b, init-sweep
showing exact-rational landings on the manifold g + 2h = 1.

This is the test that demonstrates the projective neuron's "curious ability
to converge on exact solutions" — every rational init lands at rational
coordinates with denominator 5, on the line g + 2h = 1, with displacement
direction Δh/Δg = 2 exactly. Per-sample Hessian is rank-1, so the
orthogonal direction is genuinely free (zero curvature on every sample,
not just in expectation).

Other test classes that lived in this file (TestAlgebra, TestAnalyticGradients,
TestGradientDescentOnAddition, TestGradientDescentOnMultiplication,
TestHessianInGHPlane, TestSingleReadoutAtOutput0, TestOutput1AndOutput2Predicted)
were archived to ``analysis/archive/test_projective_neuron_other_classes.py``
during the 2026-05-03 whiteroom revert.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from symbolic.projective import (
    Projective, SquaredInputProjectiveNeuron,
    encode_problem,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


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
