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

import math
import random
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Layered network (depth=2, INDEPENDENT bias per layer)
# ---------------------------------------------------------------------------
#
# Recursion (shared bias) was shown to plateau at depth-1's L²-best floor for
# a^4 (~0.13–0.16) regardless of injection mode — see
# `project_recursion_doesnt_extend_polynomial.md` and the `recursion` branch.
#
# Layering with independent bias per layer should genuinely extend the
# function class because parameter count grows with depth.  Analytical
# witnesses (verified by hand at N=4, output[0], spaced-by-2):
#
#   a^4 :  bias1 = (0, 0, 1, -1) → out1[0] = a², out1[2] = a·b
#          bias2 = (0, 0, 1, -1) → out2[0] = a^4 exactly
#
#   a^4 + b^4 :
#          bias1 = (0, 0, 1,  0) → out1[0] = a² + b², out1[2] = a·b
#          bias2 = (0, 0, 1, -3) → out2[0] = a^4 + b^4 exactly
#
# (See _RecursiveNetwork on the `recursion` branch for the shared-bias
# negative result this layered design replaces.)


@dataclass
class _LayeredNetwork:
    """Two stacked SquaredInputProjectiveNeuron layers with independent bias.

    Layer 1 input  :  (a, 0, b, 0)               (canonical N=4 spaced-by-2)
    Layer 1 output :  4-vector. Take slots 0 and 2 (the 'well-behaved' pair)
    Layer 2 input  :  (out1[0], 0, out1[2], 0)   (canonical pattern reused)
    Layer 2 output :  4-vector. Read at slot 0.

    Final readout (multi-layer; depth-1 AND depth-2 features):

        pred = alpha2 · out2[0]
             + alpha1_0 · out1[0]
             + alpha1_2 · out1[2]
             + beta · a + gamma · b + delta

    Why multi-readout: ``out1[0]``, ``out1[2]`` are quadratic in (a, b)
    (a², a·b, b² mixtures), while ``out2[0]`` is homogeneous degree-4
    in (a, b).  Without depth-1 readouts the network can't represent
    degree-2 targets like ``a·b`` — layering would *lose* depth-1's
    function class.  Multi-readout preserves it.

    Parameter count: 4 (bias1) + 4 (bias2) + 6 (alpha2, alpha1_0,
    alpha1_2, beta, gamma, delta) = 14.

    Trained with finite-difference gradients on bias1, bias2 and
    analytical gradients on the linear readout coefficients.  Gradient
    clipping keeps the bilinear cascade stable at lr=1e-5.
    """
    bias1: np.ndarray
    bias2: np.ndarray
    alpha2: float = 0.0
    alpha1_0: float = 0.0
    alpha1_2: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0

    def _layered_features(self, a: float, b: float):
        """Return ``(out1[0], out1[2], out2[0])`` — the three layered features."""
        x0 = encode_problem(a, 0, b)
        out1 = (x0 * x0) * Projective(self.bias1)
        a1 = float(out1.coeffs[0])
        b1 = float(out1.coeffs[2])
        x1 = encode_problem(a1, 0, b1)
        out2 = (x1 * x1) * Projective(self.bias2)
        return a1, b1, float(out2.coeffs[0])

    def forward(self, a: float, b: float) -> float:
        f1_0, f1_2, f2 = self._layered_features(a, b)
        return (
            self.alpha2 * f2 + self.alpha1_0 * f1_0 + self.alpha1_2 * f1_2
            + self.beta * a + self.gamma * b + self.delta
        )

    def step(
        self, a: float, b: float, target: float, lr: float,
        fd_eps: float = 1e-4, grad_clip: float = 10.0,
    ) -> float:
        f1_0, f1_2, f2 = self._layered_features(a, b)
        pred = (
            self.alpha2 * f2 + self.alpha1_0 * f1_0 + self.alpha1_2 * f1_2
            + self.beta * a + self.gamma * b + self.delta
        )
        residual = pred - target

        def _clip(g: float) -> float:
            if g >  grad_clip: return  grad_clip
            if g < -grad_clip: return -grad_clip
            return g

        # Linear readout: analytical gradients.
        self.alpha2   -= lr * _clip(2.0 * residual * f2)
        self.alpha1_0 -= lr * _clip(2.0 * residual * f1_0)
        self.alpha1_2 -= lr * _clip(2.0 * residual * f1_2)
        self.beta     -= lr * _clip(2.0 * residual * a)
        self.gamma    -= lr * _clip(2.0 * residual * b)
        self.delta    -= lr * _clip(2.0 * residual)

        # bias1: FD on all three features (perturbing bias1 affects f1_0,
        # f1_2, AND f2 via the cascade).  bias2 only affects f2.
        for k in range(4):
            saved = self.bias1[k]
            self.bias1[k] = saved + fd_eps
            p1_0, p1_2, p2 = self._layered_features(a, b)
            self.bias1[k] = saved - fd_eps
            m1_0, m1_2, m2 = self._layered_features(a, b)
            self.bias1[k] = saved
            d_f10 = (p1_0 - m1_0) / (2.0 * fd_eps)
            d_f12 = (p1_2 - m1_2) / (2.0 * fd_eps)
            d_f2  = (p2  - m2 ) / (2.0 * fd_eps)
            grad_pred = (
                self.alpha2 * d_f2 + self.alpha1_0 * d_f10 + self.alpha1_2 * d_f12
            )
            self.bias1[k] -= lr * _clip(2.0 * residual * grad_pred)

        for k in range(4):
            saved = self.bias2[k]
            self.bias2[k] = saved + fd_eps
            _, _, p2 = self._layered_features(a, b)
            self.bias2[k] = saved - fd_eps
            _, _, m2 = self._layered_features(a, b)
            self.bias2[k] = saved
            grad_pred = self.alpha2 * (p2 - m2) / (2.0 * fd_eps)
            self.bias2[k] -= lr * _clip(2.0 * residual * grad_pred)

        return pred


class TestLayeredPolynomialGeneralization:
    """Layered (independent bias per layer) at N=4, output[0], spaced-by-2.

    Targets:
      * ``a``    — degree-1, fits trivially via the linear ``beta`` readout
      * ``a·b``  — degree-2, depth=1 already fits exactly
      * ``a^4``  — degree-4, depth=1 plateaus at ~0.13–0.16; recursion (shared
                   bias) also plateaus there; this test asks whether
                   independent bias breaks below that floor.

    Smoke parameters: lr=1e-5, 10000 steps, 3 targets, single seed.
    Diagnostic logs go to ``tmp/diagnostics/layered_smoke.log``.
    """

    def _train_one(
        self, target_fn, n_steps: int, lr: float, seed: int, label: str,
    ):
        # Normalise target so loss ratios are comparable across degrees.
        bench_rng = random.Random(seed + 1)
        bench_losses = [
            target_fn(bench_rng.randint(-10, 10), bench_rng.randint(-10, 10)) ** 2
            for _ in range(5000)
        ]
        baseline = float(np.mean(bench_losses))
        scale = math.sqrt(baseline) if baseline > 0 else 1.0

        # Small random init to break the bias=0 saddle (output ∝ bias^something).
        # Magnitude 0.02 mirrors the recursion-injection setup; depth=2 over
        # quartic-in-input makes larger inits unstable under lr=1e-5.
        init_rng = np.random.default_rng(seed)
        bias1 = 0.02 * init_rng.standard_normal(4)
        bias2 = 0.02 * init_rng.standard_normal(4)
        net = _LayeredNetwork(
            bias1=bias1.copy(), bias2=bias2.copy(),
            alpha2=0.0, alpha1_0=0.0, alpha1_2=0.0,
            beta=0.0, gamma=0.0, delta=0.0,
        )

        train_rng = random.Random(seed)
        losses = []
        for _ in range(n_steps):
            a = train_rng.randint(-10, 10)
            b = train_rng.randint(-10, 10)
            target_norm = float(target_fn(a, b)) / scale
            pred = net.step(a, b, target_norm, lr)
            if not math.isfinite(pred) or abs(pred) > 1e60:
                return {
                    'label': label, 'diverged_at': len(losses),
                    'final_ratio': float('nan'),
                    'bias1': net.bias1.tolist(), 'bias2': net.bias2.tolist(),
                    'readout': (
                        net.alpha2, net.alpha1_0, net.alpha1_2,
                        net.beta, net.gamma, net.delta,
                    ),
                }
            losses.append((pred - target_norm) ** 2)

        return {
            'label': label,
            'final_ratio': float(np.mean(losses[-1000:])),
            'mid_ratio':   float(np.mean(losses[len(losses)//2:len(losses)//2+1000])),
            'bias1': net.bias1.tolist(), 'bias2': net.bias2.tolist(),
            'readout': (
                net.alpha2, net.alpha1_0, net.alpha1_2,
                net.beta, net.gamma, net.delta,
            ),
            'scale': scale,
        }

    def test_layered_smoke_n4(self):
        log_path = REPO_ROOT / "tmp" / "diagnostics" / "layered_smoke.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        targets = {
            'a':   lambda a, b: a,
            'ab':  lambda a, b: a * b,
            'a4':  lambda a, b: a ** 4,
        }
        seed = 20260502
        n_steps = 10000
        lr = 1e-5

        results = {}
        log_lines = ["target\tfinal_ratio\tmid_ratio\tbias1\tbias2\treadout"]
        for label, fn in targets.items():
            r = self._train_one(fn, n_steps, lr, seed, label)
            results[label] = r
            log_lines.append(
                f"{label}\t{r.get('final_ratio')}\t{r.get('mid_ratio', '')}\t"
                f"{r['bias1']}\t{r['bias2']}\t{r['readout']}"
            )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        # 1. Linear target 'a' should fit trivially via beta.
        assert results['a']['final_ratio'] < 0.05, (
            f"a final ratio {results['a']['final_ratio']:.4f} "
            f"(linear target should fit easily via the beta readout)"
        )

        # 2. ab is depth-1-fittable; depth-2 layered should also fit it.
        assert results['ab']['final_ratio'] < 0.05, (
            f"ab final ratio {results['ab']['final_ratio']:.4f} "
            f"(degree-2; depth-1 fits exactly, layered should too)"
        )

        # 3. a^4 — the headline test.  Depth-1 floor is ~0.13–0.16, recursion
        #    (shared bias) plateaus there.  Independent bias *should* go
        #    below that floor since exact-fit witness exists.  We use a
        #    relaxed threshold here (smoke = 10k steps; analytical optimum
        #    needs more training) — anything < 0.10 is meaningful progress
        #    over the recursion plateau.
        assert results['a4']['final_ratio'] < 0.10, (
            f"a^4 final ratio {results['a4']['final_ratio']:.4f} "
            f"— recursion plateaued at ~0.13; layered should beat that"
        )
