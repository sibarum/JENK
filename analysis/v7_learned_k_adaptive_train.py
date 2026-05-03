"""v7: same as v6 (per-op learnable k) but with PER-PARAMETER ADAPTIVE LR.

Motivation: in v6 the bias gradients are large (~10²) while k gradients are
much smaller, so a single global LR forces the most-conservative regime to
set the pace for everything. Adaptive per-parameter LR (RMSprop-style)
divides each parameter's gradient by a running estimate of its own gradient
RMS, leaving a single base LR that means "step size in normalized units."

Update rule (per parameter p):
  v_p ← β · v_p + (1 − β) · grad_p²
  p   ← p − lr · grad_p / (sqrt(v_p) + ε)

with β = 0.99, ε = 1e-8. Clamping on k applied AFTER the update.

This experiment tests whether v6's "k_+ drifts to wrong attractor"
behavior was a function of the global-LR optimization choice rather than
a fundamental property of the architecture.

Configurations tested:
  Part 1: joint training, base lr ∈ {1e-4, 1e-3, 3e-3, 1e-2}, 5 seeds, k_init=1, clamp [-1,1]
  Part 2: best lr × widened clamp [-2, 2]
  Part 3: soft-lock test (same as v6 — should still freeze on |b|=1)

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


def encode_with_learned_k(a: float, op: str, b: float, k_op: float) -> Projective:
    if b == 0:
        raise ValueError("b=0 disallowed")
    coeffs = np.zeros(N_V2)
    coeffs[V2_IN_0] = a
    coeffs[V2_OP_SLOT[op]] = 1.0
    coeffs[V2_IN_1] = math.copysign(abs(b) ** k_op, b)
    coeffs[V2_OUT_0] = 1.0
    return Projective(coeffs)


class LearnedKNeuronAdaptive:
    """Per-op learnable encoding exponent + RMSprop per-parameter adaptive LR.

    State:
      bias:    7 floats
      k:       4 floats (per op)
      bias_v:  7 floats (running mean of squared bias gradients)
      k_v:     4 floats (running mean of squared k gradients)
    """

    def __init__(
        self,
        k_init: float = 1.0,
        k_clamp: tuple = (-1.0, 1.0),
        readout_slot: int = 0,
        beta: float = 0.99,
        eps: float = 1e-8,
    ):
        self.bias = Projective.zeros(N_V2)
        self.k = np.full(len(OPS), k_init, dtype=float)
        self.bias_v = np.zeros(N_V2)
        self.k_v = np.zeros(len(OPS))
        self.beta = beta
        self.eps = eps
        self.k_clamp = k_clamp
        self.readout_slot = readout_slot
        self._neuron = SquaredInputProjectiveNeuron(
            bias=self.bias, readout_slot=readout_slot
        )

    def predict(self, a: float, op: str, b: float) -> float:
        op_idx = OP_TO_IDX[op]
        input_ = encode_with_learned_k(a, op, b, float(self.k[op_idx]))
        return self._neuron.predict(input_)

    def loss(self, a: float, op: str, b: float, target: float) -> float:
        return (self.predict(a, op, b) - target) ** 2

    def step(
        self,
        a: float,
        op: str,
        b: float,
        target: float,
        lr: float,
        eps_fd: float = 1e-4,
        grad_clip: float = 1e3,
    ) -> dict:
        """Single SGD step with RMSprop per-param normalization."""
        op_idx = OP_TO_IDX[op]
        k_active = float(self.k[op_idx])

        input_ = encode_with_learned_k(a, op, b, k_active)
        pred = self._neuron.predict(input_)

        if not math.isfinite(pred):
            return {'pred': float('nan'), 'loss': float('nan'),
                    'bias_grad_norm': float('nan'), 'k_grad': float('nan'),
                    'k_active_after': float(self.k[op_idx]), 'skipped': True}

        bias_grad = self._neuron.gradients(input_, target)
        bias_grad_norm = float(np.linalg.norm(bias_grad))
        if bias_grad_norm > grad_clip:
            bias_grad = bias_grad * (grad_clip / bias_grad_norm)
            bias_grad_norm = grad_clip

        loss = (pred - target) ** 2
        if not math.isfinite(loss):
            return {'pred': float(pred), 'loss': float('nan'),
                    'bias_grad_norm': bias_grad_norm, 'k_grad': float('nan'),
                    'k_active_after': float(self.k[op_idx]), 'skipped': True}

        # Numerical k gradient (only for the active op).
        coeffs_plus = input_.coeffs.copy()
        coeffs_minus = input_.coeffs.copy()
        coeffs_plus[V2_IN_1] = math.copysign(abs(b) ** (k_active + eps_fd), b)
        coeffs_minus[V2_IN_1] = math.copysign(abs(b) ** (k_active - eps_fd), b)
        loss_plus = (self._neuron.predict(Projective(coeffs_plus)) - target) ** 2
        loss_minus = (self._neuron.predict(Projective(coeffs_minus)) - target) ** 2
        k_grad = (loss_plus - loss_minus) / (2 * eps_fd)

        # RMSprop accumulators.
        self.bias_v[:] = self.beta * self.bias_v + (1.0 - self.beta) * (bias_grad ** 2)
        self.k_v[op_idx] = (
            self.beta * self.k_v[op_idx] + (1.0 - self.beta) * (k_grad ** 2)
        )

        # Normalized updates.
        self.bias.coeffs -= lr * bias_grad / (np.sqrt(self.bias_v) + self.eps)
        self.k[op_idx] -= lr * k_grad / (math.sqrt(self.k_v[op_idx]) + self.eps)

        # Clamp k.
        self.k[op_idx] = float(np.clip(
            self.k[op_idx], self.k_clamp[0], self.k_clamp[1]
        ))

        return {
            'pred': float(pred),
            'loss': float(loss),
            'bias_grad_norm': bias_grad_norm,
            'k_grad': float(k_grad),
            'k_active_after': float(self.k[op_idx]),
        }


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


def sample_problem(rng, op_set):
    op = rng.choice(op_set)
    a, b = sample_ab(rng)
    return op, a, b


def run_joint_training(
    seed: int = 0,
    n_steps: int = 50000,
    lr: float = 1e-3,
    k_init: float = 1.0,
    k_clamp: tuple = (-1.0, 1.0),
    log_every: int = 500,
):
    rng = random.Random(seed)
    neuron = LearnedKNeuronAdaptive(
        k_init=k_init, k_clamp=k_clamp, readout_slot=0
    )
    log = {'step': [], 'k_per_op': [], 'loss_per_op': []}
    rolling = {op: [] for op in OPS}

    for step in range(n_steps):
        op, a, b = sample_problem(rng, OPS)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr)
        rolling[op].append(info['loss'])
        if len(rolling[op]) > 200:
            rolling[op].pop(0)
        if step % log_every == 0 or step == n_steps - 1:
            log['step'].append(step)
            log['k_per_op'].append({op: float(neuron.k[OP_TO_IDX[op]]) for op in OPS})
            log['loss_per_op'].append({
                op: (float(np.mean(rolling[op])) if rolling[op] else float('nan'))
                for op in OPS
            })

    eval_rng = random.Random(seed + 100000)
    eval_losses = {op: [] for op in OPS}
    for _ in range(500):
        for op in OPS:
            a, b = sample_ab(eval_rng)
            target = TARGETS[op](a, b)
            eval_losses[op].append(neuron.loss(a, op, float(b), float(target)))
    eval_mean = {op: float(np.mean(eval_losses[op])) for op in OPS}

    return {
        'seed': seed,
        'n_steps': n_steps,
        'lr': lr,
        'k_init': k_init,
        'k_clamp': k_clamp,
        'final_bias': neuron.bias.coeffs.tolist(),
        'final_k': {op: float(neuron.k[OP_TO_IDX[op]]) for op in OPS},
        'eval_loss_per_op': eval_mean,
        'log': log,
    }


def run_softlock_test(seed: int, n_steps: int, lr: float, k_init: float):
    rng = random.Random(seed)
    neuron = LearnedKNeuronAdaptive(k_init=k_init, readout_slot=0)
    op = '/'
    initial_k = float(neuron.k[OP_TO_IDX[op]])
    k_grads = []
    losses = []
    for _ in range(n_steps):
        a, b = sample_ab_b_eq_1(rng)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr)
        k_grads.append(info['k_grad'])
        losses.append(info['loss'])
    final_k = float(neuron.k[OP_TO_IDX[op]])
    return {
        'seed': seed, 'n_steps': n_steps, 'restriction': '|b|==1, op==/',
        'initial_k_/': initial_k, 'final_k_/': final_k,
        'k_drift': final_k - initial_k,
        'max_abs_k_grad': float(np.max(np.abs(k_grads))),
        'mean_abs_k_grad': float(np.mean(np.abs(k_grads))),
        'final_loss_window': float(np.mean(losses[-200:])),
    }


def get_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def get_git_dirty() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or '(clean)'
    except Exception:
        return 'unknown'


def main() -> None:
    commit = get_commit_hash()
    dirty = get_git_dirty()
    started = time.strftime('%Y-%m-%d %H:%M:%S')

    out_lines = []
    out_lines.append("v7 learnable k + RMSprop adaptive LR — results")
    out_lines.append("=" * 78)
    out_lines.append(f"Provenance:")
    out_lines.append(f"  commit: {commit}")
    out_lines.append(f"  git status at run: {dirty[:200]}")
    out_lines.append(f"  started: {started}")
    out_lines.append(f"  optimizer: RMSprop (β=0.99, ε=1e-8) per parameter")
    out_lines.append("")

    # Part 1: LR sweep
    out_lines.append("=" * 78)
    out_lines.append("Part 1: LR sweep, joint {+,-,*,/} training, k_init=1, clamp [-1,1]")
    out_lines.append("=" * 78)
    seeds = [0, 1, 2, 3, 4]
    lrs = [1e-4, 1e-3, 3e-3, 1e-2]
    all_part1 = {}
    for lr in lrs:
        out_lines.append(f"\n--- lr = {lr:.0e} ---")
        runs = []
        for seed in seeds:
            print(f"  joint training lr={lr} seed={seed}...")
            r = run_joint_training(
                seed=seed, n_steps=50000, lr=lr, k_init=1.0, k_clamp=(-1.0, 1.0),
            )
            runs.append(r)
        all_part1[lr] = runs

        out_lines.append("  per-op aggregate (mean ± std across seeds):")
        for op in OPS:
            ks = [r['final_k'][op] for r in runs]
            ls = [r['eval_loss_per_op'][op] for r in runs]
            out_lines.append(
                f"    {op}: k = {np.mean(ks):>7.4f} ± {np.std(ks):.4f}   "
                f"loss = {np.mean(ls):>10.3f} ± {np.std(ls):.3f}"
            )

    # Best LR by total loss
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Best base LR by total mean eval loss (sum over ops, mean over seeds)")
    out_lines.append("=" * 78)
    lr_scores = {}
    for lr in lrs:
        runs = all_part1[lr]
        total = float(np.mean([sum(r['eval_loss_per_op'].values()) for r in runs]))
        lr_scores[lr] = total
        out_lines.append(f"  lr = {lr:.0e}:  total mean eval loss = {total:.3f}")
    best_lr = min(lr_scores, key=lambda x: lr_scores[x])
    out_lines.append(f"\n  -> best lr = {best_lr:.0e}")

    # k trajectory at best LR, seed 0
    log0 = all_part1[best_lr][0]['log']
    sample_indices = [0, len(log0['step']) // 10, len(log0['step']) // 4,
                      len(log0['step']) // 2, len(log0['step']) - 1]
    out_lines.append(f"\n  k(step) trajectory at lr={best_lr:.0e}, seed 0:")
    out_lines.append(f"    {'step':>8}  " + "  ".join(f"k_{op:<3}" for op in OPS))
    for idx in sample_indices:
        s = log0['step'][idx]
        ks = log0['k_per_op'][idx]
        out_lines.append(f"    {s:>8}  " + "  ".join(f"{ks[op]:>6.3f}" for op in OPS))

    # Part 2: best LR + widened clamp
    out_lines.append("\n" + "=" * 78)
    out_lines.append(f"Part 2: best lr={best_lr:.0e} with clamp widened to [-2, 2], single seed")
    out_lines.append("=" * 78)
    print(f"  widened-clamp run lr={best_lr}...")
    r_wide = run_joint_training(
        seed=0, n_steps=50000, lr=best_lr, k_init=1.0, k_clamp=(-2.0, 2.0),
    )
    out_lines.append(f"  final k: {r_wide['final_k']}")
    out_lines.append(f"  eval loss per op: {r_wide['eval_loss_per_op']}")

    # Part 3: soft-lock test (should freeze regardless of optimizer)
    out_lines.append("\n" + "=" * 78)
    out_lines.append(f"Part 3: soft-lock test at lr={best_lr:.0e}, op=/, |b|=1 only, k_init=1.0")
    out_lines.append("=" * 78)
    out_lines.append("Expectation: k_/ frozen — gradient is identically zero on |b|=1 regardless of optimizer.")
    print(f"  soft-lock test...")
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
    softlock_max_kgrad = max(sr['max_abs_k_grad'] for sr in softlock_results)
    if softlock_max_drift < 1e-9 and softlock_max_kgrad < 1e-9:
        out_lines.append(f"  CONFIRMED frozen.")
    else:
        out_lines.append(f"  k_/ moved (drift {softlock_max_drift:.2e}) — gradient was not exactly zero.")

    # Comparison block
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Comparison: v7 (RMSprop adaptive) vs v6 (vanilla SGD, lr_bias=1e-5)")
    out_lines.append("=" * 78)
    out_lines.append("v6 reference (5-seed aggregate):")
    out_lines.append("  +:  k = -0.85 ± 0.10   loss = 46.0")
    out_lines.append("  -:  k = -0.97 ± 0.03   loss = 49.1")
    out_lines.append("  *:  k = -0.25 ± 0.76   loss = 963")
    out_lines.append("  /:  k = -0.98 ± 0.02   loss = 2.56")
    out_lines.append("\nv7 best-lr aggregate:")
    runs = all_part1[best_lr]
    for op in OPS:
        ks = [r['final_k'][op] for r in runs]
        ls = [r['eval_loss_per_op'][op] for r in runs]
        out_lines.append(
            f"  {op}: k = {np.mean(ks):>7.4f} ± {np.std(ks):.4f}   "
            f"loss = {np.mean(ls):>10.3f} ± {np.std(ls):.3f}"
        )

    out_lines.append("")
    out_lines.append("=" * 78)
    out_lines.append("Configuration")
    out_lines.append("=" * 78)
    out_lines.append(f"  optimizer: RMSprop (β=0.99, ε=1e-8) per parameter")
    out_lines.append(f"  lrs swept: {lrs}")
    out_lines.append(f"  seeds: {seeds}")
    out_lines.append(f"  n_steps: 50000")
    out_lines.append(f"  bias gradient L2 clipped at 1e3")
    out_lines.append(f"  sample range: a, b ∈ {{-10..10}}, b != 0")

    text = "\n".join(out_lines)
    Path(__file__).parent.joinpath("v7_learned_k_adaptive_results.txt").write_text(
        text, encoding="utf-8"
    )

    Path(__file__).parent.joinpath("v7_learned_k_adaptive_data.json").write_text(
        json.dumps({
            'provenance': {'commit': commit, 'started': started, 'dirty': dirty},
            'part1_lr_sweep': {f'{lr:.0e}': all_part1[lr] for lr in lrs},
            'part2_widened_clamp': r_wide,
            'part3_softlock': softlock_results,
            'best_lr': best_lr,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nResults written.")


if __name__ == "__main__":
    main()
