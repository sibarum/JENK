"""v8: gradient-magnitude diagnostic for the v6/v7 architecture.

Question: when training the per-op-k neuron, what's the magnitude of
gradients on bias vs gradients on k? Per op, over time?

If the magnitudes are wildly mismatched, that explains why a global LR
fails (the larger gradient sets the safe LR, the smaller-gradient param
moves too slowly). Adaptive LR (v7) should normalize this — comparing
raw vs adaptive-effective updates shows whether RMSprop is doing its job.

Method:
  Run joint training with k_init=1, vanilla SGD lr_bias=1e-5, lr_k=1e-3
  (matching v6). Per step record op, |grad_bias| (L2), |grad_k| (abs).
  Aggregate by (op, training-window).

Output:
  Per op:
    raw |grad_bias| stats (mean, median, p25, p75, p99) per time window
    raw |grad_k| stats per time window
    ratio |grad_bias| / |grad_k|
  Effective updates:
    Same stats for `lr · grad / (sqrt(v) + eps)` from RMSprop.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from symbolic.projective import (
    N_V2,
    Projective,
    SquaredInputProjectiveNeuron,
    V2_IN_0,
    V2_IN_1,
    V2_OP_SLOT,
    V2_OUT_0,
)


TARGETS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}
OPS = list(TARGETS.keys())
OP_TO_IDX = {op: i for i, op in enumerate(OPS)}


def encode_with_learned_k(a, op, b, k_op):
    if b == 0:
        raise ValueError("b=0 disallowed")
    coeffs = np.zeros(N_V2)
    coeffs[V2_IN_0] = a
    coeffs[V2_OP_SLOT[op]] = 1.0
    coeffs[V2_IN_1] = math.copysign(abs(b) ** k_op, b)
    coeffs[V2_OUT_0] = 1.0
    return Projective(coeffs)


class DiagnosticNeuron:
    """v6/v7 hybrid: tracks BOTH raw gradients and RMSprop-normalized updates,
    but actually applies the v7 (adaptive) update so trajectory matches v7.
    """

    def __init__(self, k_init=1.0, k_clamp=(-1.0, 1.0), readout_slot=0,
                 beta=0.99, eps=1e-8):
        self.bias = Projective.zeros(N_V2)
        self.k = np.full(len(OPS), k_init, dtype=float)
        self.bias_v = np.zeros(N_V2)
        self.k_v = np.zeros(len(OPS))
        self.beta = beta
        self.eps = eps
        self.k_clamp = k_clamp
        self._neuron = SquaredInputProjectiveNeuron(
            bias=self.bias, readout_slot=readout_slot
        )

    def step(self, a, op, b, target, lr, eps_fd=1e-4, grad_clip=1e3):
        op_idx = OP_TO_IDX[op]
        k_active = float(self.k[op_idx])

        input_ = encode_with_learned_k(a, op, b, k_active)
        pred = self._neuron.predict(input_)
        if not math.isfinite(pred):
            return None

        bias_grad_raw = self._neuron.gradients(input_, target)
        bias_grad_l2_raw = float(np.linalg.norm(bias_grad_raw))

        bias_grad = bias_grad_raw.copy()
        if bias_grad_l2_raw > grad_clip:
            bias_grad = bias_grad * (grad_clip / bias_grad_l2_raw)
        bias_grad_l2_clipped = float(np.linalg.norm(bias_grad))

        loss = (pred - target) ** 2
        if not math.isfinite(loss):
            return None

        coeffs_plus = input_.coeffs.copy()
        coeffs_minus = input_.coeffs.copy()
        coeffs_plus[V2_IN_1] = math.copysign(abs(b) ** (k_active + eps_fd), b)
        coeffs_minus[V2_IN_1] = math.copysign(abs(b) ** (k_active - eps_fd), b)
        loss_plus = (self._neuron.predict(Projective(coeffs_plus)) - target) ** 2
        loss_minus = (self._neuron.predict(Projective(coeffs_minus)) - target) ** 2
        k_grad = (loss_plus - loss_minus) / (2 * eps_fd)
        k_grad_abs = float(abs(k_grad))

        # RMSprop accumulators.
        self.bias_v[:] = self.beta * self.bias_v + (1.0 - self.beta) * (bias_grad ** 2)
        self.k_v[op_idx] = (
            self.beta * self.k_v[op_idx] + (1.0 - self.beta) * (k_grad ** 2)
        )

        # Effective updates.
        bias_update = lr * bias_grad / (np.sqrt(self.bias_v) + self.eps)
        k_update = lr * k_grad / (math.sqrt(self.k_v[op_idx]) + self.eps)
        bias_update_l2 = float(np.linalg.norm(bias_update))
        k_update_abs = float(abs(k_update))

        # Apply.
        self.bias.coeffs -= bias_update
        self.k[op_idx] = float(np.clip(
            self.k[op_idx] - k_update, self.k_clamp[0], self.k_clamp[1]
        ))

        return {
            'op': op,
            'bias_grad_l2_raw': bias_grad_l2_raw,
            'bias_grad_l2_clipped': bias_grad_l2_clipped,
            'k_grad_abs': k_grad_abs,
            'bias_update_l2': bias_update_l2,
            'k_update_abs': k_update_abs,
            'k_active_after': float(self.k[op_idx]),
            'pred': float(pred),
            'loss': float(loss),
        }


def sample_ab(rng):
    while True:
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if b != 0:
            return a, b


def percentiles(values, ps=(50, 75, 90, 99)):
    if not values:
        return {f'p{p}': float('nan') for p in ps}
    arr = np.array(values)
    return {f'p{p}': float(np.percentile(arr, p)) for p in ps}


def run_diagnostic(seed: int = 0, n_steps: int = 50000, lr: float = 1e-3,
                   window: int = 5000):
    rng = random.Random(seed)
    neuron = DiagnosticNeuron(k_init=1.0, k_clamp=(-1.0, 1.0), readout_slot=0)

    # buckets: per (op, window_idx) -> list of step records.
    n_windows = (n_steps + window - 1) // window
    buckets = {op: [[] for _ in range(n_windows)] for op in OPS}

    for step in range(n_steps):
        op = rng.choice(OPS)
        a, b = sample_ab(rng)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr)
        if info is None:
            continue
        w_idx = step // window
        buckets[op][w_idx].append(info)

    # Aggregate.
    summary_per_op_window = {}
    for op in OPS:
        summary_per_op_window[op] = []
        for w_idx in range(n_windows):
            recs = buckets[op][w_idx]
            n = len(recs)
            if n == 0:
                summary_per_op_window[op].append({'window': w_idx, 'n': 0})
                continue
            bg = [r['bias_grad_l2_raw'] for r in recs]
            kg = [r['k_grad_abs'] for r in recs]
            bu = [r['bias_update_l2'] for r in recs]
            ku = [r['k_update_abs'] for r in recs]
            ratios = [r['bias_grad_l2_raw'] / max(r['k_grad_abs'], 1e-30) for r in recs]
            update_ratios = [r['bias_update_l2'] / max(r['k_update_abs'], 1e-30) for r in recs]
            summary_per_op_window[op].append({
                'window': w_idx,
                'step_range': [w_idx * window, min((w_idx + 1) * window, n_steps) - 1],
                'n': n,
                'bias_grad_mean': float(np.mean(bg)),
                'bias_grad_median': float(np.median(bg)),
                'bias_grad_max': float(np.max(bg)),
                'k_grad_mean': float(np.mean(kg)),
                'k_grad_median': float(np.median(kg)),
                'k_grad_max': float(np.max(kg)),
                'k_grad_zero_frac': float(np.mean(np.array(kg) < 1e-30)),
                'bias_update_mean': float(np.mean(bu)),
                'k_update_mean': float(np.mean(ku)),
                'raw_grad_ratio_median': float(np.median(ratios)),
                'effective_update_ratio_median': float(np.median(update_ratios)),
                'k_active_at_window_end': recs[-1]['k_active_after'],
            })

    return {
        'seed': seed, 'n_steps': n_steps, 'lr': lr, 'window': window,
        'summary': summary_per_op_window,
        'final_k': {op: float(neuron.k[OP_TO_IDX[op]]) for op in OPS},
    }


def get_commit_hash():
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def main():
    commit = get_commit_hash()
    started = time.strftime('%Y-%m-%d %H:%M:%S')

    print("Running diagnostic (seed=0, lr=1e-3, n_steps=50000, window=5000)...")
    result = run_diagnostic(seed=0, n_steps=50000, lr=1e-3, window=5000)

    out_lines = []
    out_lines.append("v8 gradient magnitude diagnostic")
    out_lines.append("=" * 100)
    out_lines.append(f"Provenance: commit {commit}, started {started}")
    out_lines.append(f"Single seed=0, lr=1e-3, n_steps=50000, window=5000")
    out_lines.append(f"Architecture: v7 (per-op k, RMSprop adaptive). Recording RAW gradients + EFFECTIVE updates.")
    out_lines.append("")

    for op in OPS:
        out_lines.append(f"=== Op: {op}  (final k = {result['final_k'][op]:.4f}) ===")
        out_lines.append(
            f"  {'window':<10} {'n':>5}  "
            f"{'|grad_bias| med':>15} {'|grad_bias| max':>15}  "
            f"{'|grad_k|   med':>15} {'|grad_k|   max':>15}  "
            f"{'k=0 frac':>10}  "
            f"{'raw ratio b/k':>15}  "
            f"{'eff |Δbias|':>15} {'eff |Δk|':>15} {'eff ratio':>10}  "
            f"{'k at end':>10}"
        )
        for w in result['summary'][op]:
            if w['n'] == 0:
                out_lines.append(f"  {w['window']:<10}  (empty)")
                continue
            sr = w['step_range']
            out_lines.append(
                f"  {sr[0]:>5}-{sr[1]:<4} {w['n']:>5}  "
                f"{w['bias_grad_median']:>15.4g} {w['bias_grad_max']:>15.4g}  "
                f"{w['k_grad_median']:>15.4g} {w['k_grad_max']:>15.4g}  "
                f"{w['k_grad_zero_frac']:>10.3f}  "
                f"{w['raw_grad_ratio_median']:>15.4g}  "
                f"{w['bias_update_mean']:>15.4g} {w['k_update_mean']:>15.4g} "
                f"{w['effective_update_ratio_median']:>10.4g}  "
                f"{w['k_active_at_window_end']:>10.4f}"
            )
        out_lines.append("")

    out_lines.append("=" * 100)
    out_lines.append("Read this:")
    out_lines.append("  - 'raw ratio b/k' = median |grad_bias| / |grad_k| in that window.")
    out_lines.append("    Big = bias gradient dominates, k can't keep up under global LR.")
    out_lines.append("  - 'eff ratio' = same but for the RMSprop-normalized updates.")
    out_lines.append("    Should be near 1 if RMSprop is doing its job.")
    out_lines.append("  - 'k=0 frac' = fraction of samples in window where k_grad numerically vanishes.")
    out_lines.append("    (|b|=1 samples + bias≈0 region produce zero k-grad.)")

    text = "\n".join(out_lines)
    Path(__file__).parent.joinpath("v8_gradient_diagnostic_results.txt").write_text(
        text, encoding="utf-8"
    )
    Path(__file__).parent.joinpath("v8_gradient_diagnostic_data.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nResults written.")


if __name__ == "__main__":
    main()
