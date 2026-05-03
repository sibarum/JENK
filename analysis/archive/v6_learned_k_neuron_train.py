"""v6: single neuron with PER-OP LEARNABLE EXPONENT k_op on the in_1 input.

Architecture:
  N = 7 (v2 layout, one-hot op, out_0 = 1)
  in_1 = sign(b) · |b|^k_active   where k_active = k[op_idx]
  hidden = input · input (algebra-squared)
  output = hidden · bias
  prediction = output[readout_slot]

Per-op k is the learnable encoding choice:
  k_/ should converge to -1 (reciprocal — fits a/b exactly)
  k_+, k_-, k_* should converge to +1 (identity — fits +,-,* exactly)

Initialization: k = 1 for all ops (identity start; SGD must actively pull k_/ down).
Clamp: k ∈ [-1, 1] (configurable; can be widened to [-2, 2] for diagnostics).

Gradient strategy:
  - bias grad: analytical (SquaredInputProjectiveNeuron's existing path).
  - k grad:    numerical finite-difference via two extra forward passes per step
               (cheap — only 1 active k per sample, 2 perturbations).

Includes a SOFT-LOCK TEST:
  Train division-only with samples restricted to |b| = 1. The gradient through
  k is |b|^k · ln(|b|) = 1 · 0 = 0 for every sample, so k_/ should never move
  from its init regardless of step count. This is the zero-gradient family
  James flagged — this test forces it to confirm the failure mode is real.

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
    """Build the v2-layout input with in_1 = sign(b)·|b|^k_op."""
    if b == 0:
        raise ValueError("b=0 disallowed")
    coeffs = np.zeros(N_V2)
    coeffs[V2_IN_0] = a
    coeffs[V2_OP_SLOT[op]] = 1.0
    coeffs[V2_IN_1] = math.copysign(abs(b) ** k_op, b)
    coeffs[V2_OUT_0] = 1.0
    return Projective(coeffs)


class LearnedKNeuron:
    """SquaredInputProjective neuron + per-op learnable encoding exponent.

    Holds bias (7 floats) and k (4 floats, one per op in OPS order).
    """

    def __init__(self, k_init: float = 1.0, k_clamp: tuple = (-1.0, 1.0), readout_slot: int = 0):
        self.bias = Projective.zeros(N_V2)
        self.k = np.full(len(OPS), k_init, dtype=float)
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

    def step(self, a: float, op: str, b: float, target: float, lr_bias: float, lr_k: float, eps: float = 1e-4, grad_clip: float = 1e3) -> dict:
        """Single SGD step: analytical bias grad, numerical k grad on the active op only.

        Skips the update if pred or loss is non-finite (prevents NaN propagation).
        Clips bias gradient L2 norm to grad_clip.
        """
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

        # Numerical gradient on k_active (other k's get zero this step).
        # Only k_active matters: input[5] depends on k_active, the rest of the input is independent.
        k_plus = k_active + eps
        k_minus = k_active - eps
        in1_plus = math.copysign(abs(b) ** k_plus, b)
        in1_minus = math.copysign(abs(b) ** k_minus, b)
        # Build perturbed inputs (only in_1 changes).
        coeffs_plus = input_.coeffs.copy()
        coeffs_minus = input_.coeffs.copy()
        coeffs_plus[V2_IN_1] = in1_plus
        coeffs_minus[V2_IN_1] = in1_minus
        loss_plus = (self._neuron.predict(Projective(coeffs_plus)) - target) ** 2
        loss_minus = (self._neuron.predict(Projective(coeffs_minus)) - target) ** 2
        k_grad = (loss_plus - loss_minus) / (2 * eps)

        # Update.
        self.bias.coeffs -= lr_bias * bias_grad
        self.k[op_idx] = float(np.clip(
            self.k[op_idx] - lr_k * k_grad, self.k_clamp[0], self.k_clamp[1]
        ))

        return {
            'pred': float(pred),
            'loss': float(loss),
            'bias_grad_norm': float(np.linalg.norm(bias_grad)),
            'k_grad': float(k_grad),
            'k_active_after': float(self.k[op_idx]),
        }


def sample_ab(rng):
    """(a, b) integers in [-10, 10] with b != 0."""
    while True:
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if b != 0:
            return a, b


def sample_ab_b_eq_1(rng):
    """(a, b) with |b| = 1 — used to force the zero-gradient soft-lock."""
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
    lr_bias: float = 1e-4,
    lr_k: float = 1e-3,
    k_init: float = 1.0,
    k_clamp: tuple = (-1.0, 1.0),
    log_every: int = 500,
):
    """Train on joint {+,-,*,/} task; record per-op loss and per-op k(t) trajectory."""
    rng = random.Random(seed)
    neuron = LearnedKNeuron(k_init=k_init, k_clamp=k_clamp, readout_slot=0)

    log = {
        'step': [],
        'k_per_op': [],   # list of dicts {op: k_value}
        'loss_per_op': [],  # rolling-window losses per op
    }
    rolling = {op: [] for op in OPS}

    for step in range(n_steps):
        op, a, b = sample_problem(rng, OPS)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr_bias, lr_k)
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

    # Held-out eval per op.
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
        'lr_bias': lr_bias,
        'lr_k': lr_k,
        'k_init': k_init,
        'k_clamp': k_clamp,
        'final_bias': neuron.bias.coeffs.tolist(),
        'final_k': {op: float(neuron.k[OP_TO_IDX[op]]) for op in OPS},
        'eval_loss_per_op': eval_mean,
        'log': log,
    }


def run_softlock_test(
    seed: int = 0,
    n_steps: int = 5000,
    lr_bias: float = 1e-4,
    lr_k: float = 1e-3,
    k_init: float = 1.0,
):
    """Force the |b|=1 zero-gradient soft-lock: train division on |b|=1 samples only.

    Expectation: k_/ stays exactly at k_init forever (0 gradient on every sample).
    Bias may still adjust (it has its own gradient path, independent of k).
    """
    rng = random.Random(seed)
    neuron = LearnedKNeuron(k_init=k_init, readout_slot=0)
    op = '/'

    initial_k = float(neuron.k[OP_TO_IDX[op]])
    k_grads_seen = []
    losses = []

    for step in range(n_steps):
        a, b = sample_ab_b_eq_1(rng)
        target = TARGETS[op](a, b)
        info = neuron.step(a, op, float(b), float(target), lr_bias, lr_k)
        k_grads_seen.append(info['k_grad'])
        losses.append(info['loss'])

    final_k = float(neuron.k[OP_TO_IDX[op]])
    return {
        'seed': seed,
        'n_steps': n_steps,
        'restriction': '|b| == 1 only, op == /',
        'initial_k_/': initial_k,
        'final_k_/': final_k,
        'k_drift': final_k - initial_k,
        'max_abs_k_grad': float(np.max(np.abs(k_grads_seen))),
        'mean_abs_k_grad': float(np.mean(np.abs(k_grads_seen))),
        'final_loss_window': float(np.mean(losses[-200:])),
    }


def get_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def get_git_dirty() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or '(clean)'
    except Exception:
        return 'unknown'


def main() -> None:
    commit = get_commit_hash()
    dirty = get_git_dirty()
    started = time.strftime('%Y-%m-%d %H:%M:%S')

    out_lines = []
    out_lines.append("v6 learnable per-op encoding exponent — results")
    out_lines.append("=" * 78)
    out_lines.append(f"Provenance:")
    out_lines.append(f"  commit: {commit}")
    out_lines.append(f"  git status at run: {dirty}")
    out_lines.append(f"  started: {started}")
    out_lines.append("")

    # === Part 1: joint training, k clamped to [-1, 1] ===
    out_lines.append("=" * 78)
    out_lines.append("Part 1: joint {+,-,*,/} training, k_init=1.0, clamp=[-1,1]")
    out_lines.append("=" * 78)

    seeds = [0, 1, 2, 3, 4]
    results_part1 = []
    for seed in seeds:
        print(f"  joint training seed={seed}...")
        r = run_joint_training(
            seed=seed,
            n_steps=50000,
            lr_bias=1e-5,   # lower than v5's 1e-4 because identity (k_init=1) diverges at 1e-4
            lr_k=1e-3,
            k_init=1.0,
            k_clamp=(-1.0, 1.0),
        )
        results_part1.append(r)
        out_lines.append(f"\n  seed {seed}:")
        out_lines.append(f"    final k: {r['final_k']}")
        out_lines.append(f"    eval loss per op: {r['eval_loss_per_op']}")

    # Mean across seeds
    out_lines.append("\n  Aggregate across seeds:")
    for op in OPS:
        ks = [r['final_k'][op] for r in results_part1]
        ls = [r['eval_loss_per_op'][op] for r in results_part1]
        out_lines.append(
            f"    {op}: k = {np.mean(ks):.4f} ± {np.std(ks):.4f}  "
            f"loss = {np.mean(ls):.4f} ± {np.std(ls):.4f}"
        )

    # k trajectory (just seed 0 for the main spine)
    out_lines.append("\n  k(step) trajectory, seed 0:")
    log0 = results_part1[0]['log']
    sample_indices = [0, len(log0['step']) // 10, len(log0['step']) // 4,
                      len(log0['step']) // 2, len(log0['step']) - 1]
    out_lines.append(f"    {'step':>8}  " + "  ".join(f"k_{op:<3}" for op in OPS))
    for idx in sample_indices:
        s = log0['step'][idx]
        ks = log0['k_per_op'][idx]
        out_lines.append(f"    {s:>8}  " + "  ".join(f"{ks[op]:>6.3f}" for op in OPS))

    # === Part 2: widened clamp [-2, 2] (curiosity) ===
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Part 2: same training but clamp widened to [-2, 2] (k_init=1.0)")
    out_lines.append("=" * 78)
    print(f"  widened-clamp training, single seed=0...")
    r_wide = run_joint_training(
        seed=0, n_steps=50000, lr_bias=1e-5, lr_k=1e-3,
        k_init=1.0, k_clamp=(-2.0, 2.0),
    )
    out_lines.append(f"\n  final k: {r_wide['final_k']}")
    out_lines.append(f"  eval loss per op: {r_wide['eval_loss_per_op']}")

    # === Part 3: soft-lock force test ===
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Part 3: soft-lock force test — division-only, |b|=1 only, k_init=1.0")
    out_lines.append("=" * 78)
    out_lines.append("Expectation: k_/ does NOT move (gradient is identically zero on |b|=1).")
    out_lines.append("")
    print(f"  soft-lock test, multiple seeds...")
    softlock_results = []
    for seed in [0, 1, 2]:
        sr = run_softlock_test(seed=seed, n_steps=5000, lr_bias=1e-5, lr_k=1e-3, k_init=1.0)
        softlock_results.append(sr)
        out_lines.append(
            f"  seed {seed}: initial k_/ = {sr['initial_k_/']:.4f}, "
            f"final k_/ = {sr['final_k_/']:.6f}, drift = {sr['k_drift']:.6e}, "
            f"max|k_grad| = {sr['max_abs_k_grad']:.4e}, "
            f"mean|k_grad| = {sr['mean_abs_k_grad']:.4e}"
        )

    softlock_max_drift = max(abs(sr['k_drift']) for sr in softlock_results)
    softlock_max_kgrad = max(sr['max_abs_k_grad'] for sr in softlock_results)
    out_lines.append("")
    if softlock_max_drift < 1e-9 and softlock_max_kgrad < 1e-9:
        out_lines.append(
            f"  CONFIRMED: k_/ frozen (max drift {softlock_max_drift:.2e}, "
            f"max |grad| {softlock_max_kgrad:.2e}). Zero-gradient family is real."
        )
    else:
        out_lines.append(
            f"  k_/ moved (max drift {softlock_max_drift:.2e}). "
            f"Gradient was not exactly zero — investigate."
        )

    # === Save ===
    out_lines.append("")
    out_lines.append("=" * 78)
    out_lines.append("Configuration summary")
    out_lines.append("=" * 78)
    out_lines.append(f"  ops: {OPS}")
    out_lines.append(f"  encoding: in_1 = sign(b) · |b|^k_op (per-op exponent)")
    out_lines.append(f"  bias gradient: analytical (SquaredInputProjectiveNeuron)")
    out_lines.append(f"  k gradient: numerical finite-difference, eps=1e-4")
    out_lines.append(f"  joint training: 50000 steps, lr_bias=1e-5, lr_k=1e-3, 5 seeds")
    out_lines.append(f"  bias gradient L2 clipped at 1e3; non-finite pred/loss skips update")
    out_lines.append(f"  sample range: a, b ∈ {{-10..10}}, b != 0")

    text = "\n".join(out_lines)
    Path(__file__).parent.joinpath("v6_learned_k_results.txt").write_text(text, encoding="utf-8")

    # Save per-seed JSON for follow-up analysis (k trajectories etc.)
    json_data = {
        'provenance': {'commit': commit, 'started': started, 'dirty': dirty},
        'part1_joint': results_part1,
        'part2_widened_clamp': r_wide,
        'part3_softlock': softlock_results,
    }
    Path(__file__).parent.joinpath("v6_learned_k_data.json").write_text(
        json.dumps(json_data, indent=2), encoding="utf-8"
    )

    print(f"\nResults written to:")
    print(f"  analysis/v6_learned_k_results.txt")
    print(f"  analysis/v6_learned_k_data.json")


if __name__ == "__main__":
    main()
