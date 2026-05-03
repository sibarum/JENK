"""v9: same as v7 (per-op learnable k + RMSprop adaptive LR) but with
PER-OP LOSS NORMALIZATION.

Hypothesis (James):
  v6/v7's asymmetric drift (k_*, k_/ converge to right values, but k_+
  and k_- get dragged toward k_/'s direction) is because squared error
  weights each op by target² scale. * has targets up to 100, / has
  targets up to 10 — so * dominates the bias-gradient signal by ~100×
  even though the architecture should treat them symmetrically.

  Fix: train with  loss_op = (pred − target)² / baseline_op
  where baseline_op = E[target²] under predict-zero. This makes every
  op's loss roughly unit-scale, so gradient magnitudes from each op
  contribute comparably to the bias dynamics.

What this should do if the hypothesis is right:
  - |grad_k_*| at k=+1 and |grad_k_/| at k=-1 become comparable
  - bias dynamics no longer dominated by * (its loss is normalized down)
  - k_+ and k_- stop drifting to negative (no asymmetric pull)
  - all four ops converge cleanly to (k_+, k_-, k_*, k_/) = (+1, +1, +1, -1)

What this can't fix (if it doesn't help):
  - bias-k coupling itself (would need architectural change)
  - the sign(b)·|b|^k parameterization vs Möbius (would need different encoder)

Provenance recorded in output file.
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


def sample_ab(rng):
    while True:
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if b != 0:
            return a, b


def sample_ab_b_eq_1(rng):
    a = rng.randint(-10, 10)
    b = rng.choice([-1, 1])
    return a, b


def baseline_loss(op: str, n: int = 5000, seed: int = 999) -> float:
    rng = random.Random(seed)
    target_fn = TARGETS[op]
    losses = []
    for _ in range(n):
        a, b = sample_ab(rng)
        target = target_fn(a, b)
        losses.append(target ** 2)
    return float(np.mean(losses))


# Pre-compute baselines once at module load — they don't depend on training state.
BASELINES = {op: baseline_loss(op) for op in OPS}


class LearnedKNeuronNormalized:
    """Per-op k + RMSprop adaptive LR + per-op loss normalization (1/baseline_op).

    The forward pass is identical to v7. The only difference is that the
    loss (and therefore both bias_grad and k_grad) is divided by baseline_op
    before backprop.
    """

    def __init__(
        self, k_init=1.0, k_clamp=(-1.0, 1.0), readout_slot=0,
        beta=0.99, eps=1e-8,
    ):
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

    def predict(self, a, op, b):
        op_idx = OP_TO_IDX[op]
        input_ = encode_with_learned_k(a, op, b, float(self.k[op_idx]))
        return self._neuron.predict(input_)

    def loss_normalized(self, a, op, b, target):
        return (self.predict(a, op, b) - target) ** 2 / BASELINES[op]

    def loss_raw(self, a, op, b, target):
        return (self.predict(a, op, b) - target) ** 2

    def step(self, a, op, b, target, lr, eps_fd=1e-4, grad_clip=1e3):
        op_idx = OP_TO_IDX[op]
        k_active = float(self.k[op_idx])
        baseline = BASELINES[op]

        input_ = encode_with_learned_k(a, op, b, k_active)
        pred = self._neuron.predict(input_)
        if not math.isfinite(pred):
            return None

        # Analytical bias gradient w.r.t. raw loss, then scale by 1/baseline for normalized loss.
        bias_grad_raw = self._neuron.gradients(input_, target)
        bias_grad = bias_grad_raw / baseline
        bias_grad_l2 = float(np.linalg.norm(bias_grad))
        if bias_grad_l2 > grad_clip:
            bias_grad = bias_grad * (grad_clip / bias_grad_l2)
            bias_grad_l2 = grad_clip

        loss_normed = (pred - target) ** 2 / baseline
        if not math.isfinite(loss_normed):
            return None

        # Numerical k gradient on normalized loss.
        coeffs_plus = input_.coeffs.copy()
        coeffs_minus = input_.coeffs.copy()
        coeffs_plus[V2_IN_1] = math.copysign(abs(b) ** (k_active + eps_fd), b)
        coeffs_minus[V2_IN_1] = math.copysign(abs(b) ** (k_active - eps_fd), b)
        loss_plus = (
            (self._neuron.predict(Projective(coeffs_plus)) - target) ** 2 / baseline
        )
        loss_minus = (
            (self._neuron.predict(Projective(coeffs_minus)) - target) ** 2 / baseline
        )
        k_grad = (loss_plus - loss_minus) / (2 * eps_fd)

        # RMSprop accumulators (on the normalized gradients).
        self.bias_v[:] = self.beta * self.bias_v + (1.0 - self.beta) * (bias_grad ** 2)
        self.k_v[op_idx] = (
            self.beta * self.k_v[op_idx] + (1.0 - self.beta) * (k_grad ** 2)
        )

        bias_update = lr * bias_grad / (np.sqrt(self.bias_v) + self.eps)
        k_update = lr * k_grad / (math.sqrt(self.k_v[op_idx]) + self.eps)

        self.bias.coeffs -= bias_update
        self.k[op_idx] = float(np.clip(
            self.k[op_idx] - k_update, self.k_clamp[0], self.k_clamp[1]
        ))

        return {
            'pred': float(pred),
            'loss_normed': float(loss_normed),
            'loss_raw': float((pred - target) ** 2),
            'bias_grad_l2': bias_grad_l2,
            'k_grad_abs': float(abs(k_grad)),
            'bias_update_l2': float(np.linalg.norm(bias_update)),
            'k_update_abs': float(abs(k_update)),
            'k_active_after': float(self.k[op_idx]),
        }


def run_joint_training(seed=0, n_steps=50000, lr=1e-3,
                       k_init=1.0, k_clamp=(-1.0, 1.0), log_every=500,
                       record_grads=False):
    rng = random.Random(seed)
    neuron = LearnedKNeuronNormalized(k_init=k_init, k_clamp=k_clamp, readout_slot=0)
    log = {'step': [], 'k_per_op': [], 'loss_per_op_raw': []}
    rolling_raw = {op: [] for op in OPS}
    grad_records = {op: [] for op in OPS} if record_grads else None

    for step in range(n_steps):
        op = rng.choice(OPS)
        a, b = sample_ab(rng)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr)
        if info is None:
            continue
        rolling_raw[op].append(info['loss_raw'])
        if len(rolling_raw[op]) > 200:
            rolling_raw[op].pop(0)
        if record_grads:
            grad_records[op].append({
                'step': step,
                'bias_grad_l2': info['bias_grad_l2'],
                'k_grad_abs': info['k_grad_abs'],
                'bias_update_l2': info['bias_update_l2'],
                'k_update_abs': info['k_update_abs'],
            })
        if step % log_every == 0 or step == n_steps - 1:
            log['step'].append(step)
            log['k_per_op'].append({op: float(neuron.k[OP_TO_IDX[op]]) for op in OPS})
            log['loss_per_op_raw'].append({
                op: (float(np.mean(rolling_raw[op])) if rolling_raw[op] else float('nan'))
                for op in OPS
            })

    eval_rng = random.Random(seed + 100000)
    eval_losses_raw = {op: [] for op in OPS}
    for _ in range(500):
        for op in OPS:
            a, b = sample_ab(eval_rng)
            target = TARGETS[op](a, b)
            eval_losses_raw[op].append(neuron.loss_raw(a, op, float(b), float(target)))
    eval_mean_raw = {op: float(np.mean(eval_losses_raw[op])) for op in OPS}
    eval_ratio = {op: eval_mean_raw[op] / BASELINES[op] for op in OPS}

    result = {
        'seed': seed, 'n_steps': n_steps, 'lr': lr, 'k_init': k_init, 'k_clamp': k_clamp,
        'final_bias': neuron.bias.coeffs.tolist(),
        'final_k': {op: float(neuron.k[OP_TO_IDX[op]]) for op in OPS},
        'eval_loss_per_op_raw': eval_mean_raw,
        'eval_ratio_to_baseline': eval_ratio,
        'log': log,
    }
    if record_grads:
        result['grad_records'] = grad_records
    return result


def run_softlock_test(seed, n_steps, lr, k_init):
    rng = random.Random(seed)
    neuron = LearnedKNeuronNormalized(k_init=k_init, readout_slot=0)
    op = '/'
    initial_k = float(neuron.k[OP_TO_IDX[op]])
    k_grads = []
    losses = []
    for _ in range(n_steps):
        a, b = sample_ab_b_eq_1(rng)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr)
        if info is None:
            continue
        k_grads.append(info['k_grad_abs'])
        losses.append(info['loss_raw'])
    final_k = float(neuron.k[OP_TO_IDX[op]])
    return {
        'seed': seed, 'restriction': '|b|==1, op==/',
        'initial_k_/': initial_k, 'final_k_/': final_k,
        'k_drift': final_k - initial_k,
        'max_abs_k_grad': float(np.max(k_grads)) if k_grads else 0.0,
        'mean_abs_k_grad': float(np.mean(k_grads)) if k_grads else 0.0,
        'final_loss_raw_window': float(np.mean(losses[-200:])) if len(losses) >= 200 else float('nan'),
    }


def percentile(values, p):
    return float(np.percentile(values, p)) if values else float('nan')


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

    out_lines = []
    out_lines.append("v9 normalized-loss training — results")
    out_lines.append("=" * 78)
    out_lines.append(f"Provenance: commit {commit}, started {started}")
    out_lines.append(f"Architecture: v7 (per-op k + RMSprop) with PER-OP LOSS NORMALIZATION")
    out_lines.append(f"  loss = (pred - target)² / baseline_op")
    out_lines.append(f"Baselines: " + ", ".join(f"{op}: {BASELINES[op]:.2f}" for op in OPS))
    out_lines.append("")

    # Part 1: LR sweep
    out_lines.append("=" * 78)
    out_lines.append("Part 1: LR sweep, joint {+,-,*,/} training, k_init=1, clamp [-1,1], 5 seeds")
    out_lines.append("=" * 78)
    seeds = [0, 1, 2, 3, 4]
    lrs = [1e-4, 1e-3, 3e-3, 1e-2]
    all_part1 = {}
    for lr in lrs:
        out_lines.append(f"\n--- lr = {lr:.0e} ---")
        runs = []
        for seed in seeds:
            print(f"  joint training (normalized) lr={lr} seed={seed}...")
            r = run_joint_training(
                seed=seed, n_steps=50000, lr=lr,
                k_init=1.0, k_clamp=(-1.0, 1.0),
            )
            runs.append(r)
        all_part1[lr] = runs
        out_lines.append("  per-op aggregate (mean ± std across seeds):")
        for op in OPS:
            ks = [r['final_k'][op] for r in runs]
            ls = [r['eval_loss_per_op_raw'][op] for r in runs]
            ratios = [r['eval_ratio_to_baseline'][op] for r in runs]
            out_lines.append(
                f"    {op}: k = {np.mean(ks):>7.4f} ± {np.std(ks):.4f}   "
                f"loss = {np.mean(ls):>10.3f} ± {np.std(ls):.3f}   "
                f"ratio = {np.mean(ratios):.4f}× ± {np.std(ratios):.4f}"
            )

    # Best LR by total normalized eval loss (sum of ratios)
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Best base LR by SUM of per-op eval-ratios (normalized objective)")
    out_lines.append("=" * 78)
    lr_scores = {}
    for lr in lrs:
        runs = all_part1[lr]
        total_ratio = float(np.mean([
            sum(r['eval_ratio_to_baseline'].values()) for r in runs
        ]))
        lr_scores[lr] = total_ratio
        out_lines.append(f"  lr = {lr:.0e}:  mean total eval ratio = {total_ratio:.4f}")
    best_lr = min(lr_scores, key=lambda x: lr_scores[x])
    out_lines.append(f"\n  -> best lr = {best_lr:.0e}")

    # k trajectory at best LR
    log0 = all_part1[best_lr][0]['log']
    sample_indices = [0, len(log0['step']) // 10, len(log0['step']) // 4,
                      len(log0['step']) // 2, len(log0['step']) - 1]
    out_lines.append(f"\n  k(step) trajectory at lr={best_lr:.0e}, seed 0:")
    out_lines.append(f"    {'step':>8}  " + "  ".join(f"k_{op:<3}" for op in OPS))
    for idx in sample_indices:
        s = log0['step'][idx]
        ks = log0['k_per_op'][idx]
        out_lines.append(f"    {s:>8}  " + "  ".join(f"{ks[op]:>6.3f}" for op in OPS))

    # Part 2: gradient diagnostic at best LR
    out_lines.append("\n" + "=" * 78)
    out_lines.append(f"Part 2: gradient magnitude diagnostic at best lr={best_lr:.0e}, seed=0")
    out_lines.append("=" * 78)
    print(f"  gradient diagnostic at best lr={best_lr}...")
    diag = run_joint_training(
        seed=0, n_steps=50000, lr=best_lr,
        k_init=1.0, k_clamp=(-1.0, 1.0), record_grads=True,
    )

    out_lines.append("Per-op summary (last 10000 training steps):")
    out_lines.append(
        f"  {'op':<3} {'|grad_bias| med':>16} {'|grad_k| med':>14} "
        f"{'raw ratio b/k':>15} {'eff |Δbias|':>14} {'eff |Δk|':>14} {'eff ratio':>10}"
    )
    for op in OPS:
        recs = [r for r in diag['grad_records'][op] if r['step'] >= 40000]
        if not recs:
            out_lines.append(f"  {op}: (no records)")
            continue
        bg = [r['bias_grad_l2'] for r in recs]
        kg = [r['k_grad_abs'] for r in recs]
        bu = [r['bias_update_l2'] for r in recs]
        ku = [r['k_update_abs'] for r in recs]
        raw_ratios = [r['bias_grad_l2'] / max(r['k_grad_abs'], 1e-30) for r in recs]
        eff_ratios = [r['bias_update_l2'] / max(r['k_update_abs'], 1e-30) for r in recs]
        out_lines.append(
            f"  {op:<3} {percentile(bg, 50):>16.4g} {percentile(kg, 50):>14.4g} "
            f"{percentile(raw_ratios, 50):>15.4g} {percentile(bu, 50):>14.4g} "
            f"{percentile(ku, 50):>14.4g} {percentile(eff_ratios, 50):>10.4g}"
        )

    # Part 3: soft-lock confirmation
    out_lines.append("\n" + "=" * 78)
    out_lines.append(f"Part 3: soft-lock test at lr={best_lr:.0e}, op=/, |b|=1 only, k_init=1.0")
    out_lines.append("=" * 78)
    softlock_results = []
    for seed in [0, 1, 2]:
        sr = run_softlock_test(seed=seed, n_steps=5000, lr=best_lr, k_init=1.0)
        softlock_results.append(sr)
        out_lines.append(
            f"  seed {seed}: initial k_/ = {sr['initial_k_/']:.4f}, "
            f"final k_/ = {sr['final_k_/']:.6f}, drift = {sr['k_drift']:.6e}, "
            f"max|grad| = {sr['max_abs_k_grad']:.4e}"
        )
    softlock_max_drift = max(abs(sr['k_drift']) for sr in softlock_results)
    if softlock_max_drift < 1e-9:
        out_lines.append(f"  CONFIRMED frozen.")
    else:
        out_lines.append(f"  k_/ moved (drift {softlock_max_drift:.2e}).")

    # Comparison
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Comparison: v9 (normalized loss) vs v7 (raw loss)")
    out_lines.append("=" * 78)
    out_lines.append("v7 best (lr=1e-3) reference:")
    out_lines.append("  +:  k = -0.6700 ± 0.0355   loss = 43.58 (ratio 0.576×)")
    out_lines.append("  -:  k = -0.9986 ± 0.0016   loss = 47.27 (ratio 0.623×)")
    out_lines.append("  *:  k = +0.9996 ± 0.0006   loss = 39.39 (ratio 0.028×)")
    out_lines.append("  /:  k = -0.9521 ± 0.0213   loss =  2.43 (ratio 0.437×)")
    out_lines.append("\nv9 best aggregate:")
    runs = all_part1[best_lr]
    for op in OPS:
        ks = [r['final_k'][op] for r in runs]
        ls = [r['eval_loss_per_op_raw'][op] for r in runs]
        ratios = [r['eval_ratio_to_baseline'][op] for r in runs]
        out_lines.append(
            f"  {op}: k = {np.mean(ks):>7.4f} ± {np.std(ks):.4f}   "
            f"loss = {np.mean(ls):>10.3f}   "
            f"ratio = {np.mean(ratios):.4f}×"
        )

    out_lines.append("")
    out_lines.append("=" * 78)
    out_lines.append("Verdict reading guide")
    out_lines.append("=" * 78)
    out_lines.append("If hypothesis correct:")
    out_lines.append("  k_+ → +1, k_- → +1, k_* → +1, k_/ → -1  (clean per-op convergence)")
    out_lines.append("  raw |grad_k| roughly equal between * and /  (no scale asymmetry)")
    out_lines.append("If hypothesis wrong:")
    out_lines.append("  k_+, k_- still drift to negative  (issue is deeper than scale)")
    out_lines.append("  in which case test (2) Möbius parameterization or (3) algebra change")

    text = "\n".join(out_lines)
    Path(__file__).parent.joinpath("v9_normalized_loss_results.txt").write_text(
        text, encoding="utf-8"
    )

    # Strip grad records from saved JSON (huge); keep summaries.
    json_data = {
        'provenance': {'commit': commit, 'started': started},
        'baselines': BASELINES,
        'part1_lr_sweep': {
            f'{lr:.0e}': [
                {k: v for k, v in r.items() if k != 'grad_records'}
                for r in all_part1[lr]
            ]
            for lr in lrs
        },
        'part3_softlock': softlock_results,
        'best_lr': best_lr,
    }
    Path(__file__).parent.joinpath("v9_normalized_loss_data.json").write_text(
        json.dumps(json_data, indent=2), encoding="utf-8"
    )
    print("\nResults written.")


if __name__ == "__main__":
    main()
