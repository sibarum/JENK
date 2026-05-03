"""v10: collapse {+, -, *, /} into two architectural ops, with deterministic
input filters for - and / on the second operand.

Insight (from James):
  - and / are not separate operations from + and *. They are + and *
  composed with the canonical inverses on the second operand:
      a − b == a + (−b)
      a / b == a · (1/b)
  Negation and reciprocal are *constitutive* of subtraction and division by
  definition; using them is not "cheating," it's the actual algebra.

Architecture (N = 5):
  k_0 = in_0 (a)
  k_1 = op_+         one-hot for {+, −}
  k_2 = op_*         one-hot for {*, /}
  k_3 = in_1 (b')    where  b' = b   (for +, *)
                            b' = −b  (for −)
                            b' = 1/b (for /)
  k_4 = out_0 (= 1)

The neuron has a single shared bias of length 5. There is NO learned k,
NO per-op anything in the model itself. The network learns one bias to
fit two architectural ops; user-facing ops are dispatched to those two
through a deterministic input transform.

Training: joint sampling over all four user-facing ops, equal weight.
Evaluation reported per logical op (the user-facing one).

Sweeps: lr × readout_slot × seed.
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
    Projective,
    SquaredInputProjectiveNeuron,
)


N = 5

# Slot layout
SLOT_IN_0 = 0
SLOT_OP_ADD = 1     # active for {+, −}
SLOT_OP_MUL = 2     # active for {*, /}
SLOT_IN_1 = 3
SLOT_OUT_0 = 4


TARGETS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}
LOGICAL_OPS = ['+', '-', '*', '/']


def encode_two_op(a: float, op: str, b: float) -> Projective:
    """Route user-facing op to architectural op with input transform on b."""
    if b == 0:
        raise ValueError("b=0 disallowed (avoids both /-by-zero and trivial sample)")
    coeffs = np.zeros(N)
    coeffs[SLOT_IN_0] = a
    coeffs[SLOT_OUT_0] = 1.0

    if op == '+':
        coeffs[SLOT_OP_ADD] = 1.0
        coeffs[SLOT_IN_1] = b
    elif op == '-':
        coeffs[SLOT_OP_ADD] = 1.0
        coeffs[SLOT_IN_1] = -b
    elif op == '*':
        coeffs[SLOT_OP_MUL] = 1.0
        coeffs[SLOT_IN_1] = b
    elif op == '/':
        coeffs[SLOT_OP_MUL] = 1.0
        coeffs[SLOT_IN_1] = 1.0 / b
    else:
        raise ValueError(f"unknown op: {op}")

    return Projective(coeffs)


def sample_ab(rng):
    while True:
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if b != 0:
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


BASELINES = {op: baseline_loss(op) for op in LOGICAL_OPS}


def train_run(
    seed: int = 0,
    n_steps: int = 30000,
    lr: float = 1e-5,
    readout_slot: int = 0,
    log_every: int = 500,
):
    rng = random.Random(seed)
    neuron = SquaredInputProjectiveNeuron(
        bias=Projective.zeros(N),
        readout_slot=readout_slot,
    )

    rolling = {op: [] for op in LOGICAL_OPS}
    log = {'step': [], 'loss_per_op': []}

    skipped = 0
    diverged = False
    for step in range(n_steps):
        op = rng.choice(LOGICAL_OPS)
        a, b = sample_ab(rng)
        target = TARGETS[op](a, b)
        input_ = encode_two_op(a, op, float(b))

        try:
            pred = neuron.step(input_, float(target), lr)
        except (OverflowError, FloatingPointError):
            diverged = True
            break

        if not math.isfinite(pred) or abs(pred) > 1e100:
            diverged = True
            break

        try:
            loss = (pred - target) ** 2
        except OverflowError:
            diverged = True
            break

        if not math.isfinite(loss):
            skipped += 1
            continue

        rolling[op].append(loss)
        if len(rolling[op]) > 200:
            rolling[op].pop(0)

        if step % log_every == 0 or step == n_steps - 1:
            log['step'].append(step)
            log['loss_per_op'].append({
                op: (float(np.mean(rolling[op])) if rolling[op] else float('nan'))
                for op in LOGICAL_OPS
            })

    if diverged:
        return {
            'seed': seed, 'n_steps': step, 'lr': lr, 'readout_slot': readout_slot,
            'final_bias': neuron.bias.coeffs.tolist(),
            'eval_loss_per_op': {op: float('inf') for op in LOGICAL_OPS},
            'eval_ratio_to_baseline': {op: float('inf') for op in LOGICAL_OPS},
            'log': log,
            'skipped_steps': skipped,
            'diverged': True,
            'diverged_at_step': step,
        }

    # Held-out eval per op.
    eval_rng = random.Random(seed + 100000)
    eval_losses = {op: [] for op in LOGICAL_OPS}
    n_eval = 500
    for _ in range(n_eval):
        for op in LOGICAL_OPS:
            a, b = sample_ab(eval_rng)
            target = TARGETS[op](a, b)
            input_ = encode_two_op(a, op, float(b))
            pred = neuron.predict(input_)
            eval_losses[op].append((pred - target) ** 2)
    eval_mean = {op: float(np.mean(eval_losses[op])) for op in LOGICAL_OPS}
    eval_ratio = {op: eval_mean[op] / BASELINES[op] for op in LOGICAL_OPS}

    return {
        'seed': seed, 'n_steps': n_steps, 'lr': lr, 'readout_slot': readout_slot,
        'final_bias': neuron.bias.coeffs.tolist(),
        'eval_loss_per_op': eval_mean,
        'eval_ratio_to_baseline': eval_ratio,
        'log': log,
        'skipped_steps': skipped,
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

    out_lines = []
    out_lines.append("v10 two-op-architecture training — results")
    out_lines.append("=" * 78)
    out_lines.append(f"Provenance: commit {commit}, started {started}")
    out_lines.append("Architecture: N=5; 2 op slots (op_+, op_*); shared bias; NO learned k.")
    out_lines.append("Slots: k_0=in_0(a) k_1=op_+ k_2=op_* k_3=in_1(b') k_4=out_0(=1)")
    out_lines.append("Input transforms:")
    out_lines.append("  +: (a, 1, 0, b,    1)   -- op_+ active, b'=b")
    out_lines.append("  -: (a, 1, 0, -b,   1)   -- op_+ active, b'=-b")
    out_lines.append("  *: (a, 0, 1, b,    1)   -- op_* active, b'=b")
    out_lines.append("  /: (a, 0, 1, 1/b,  1)   -- op_* active, b'=1/b")
    out_lines.append("")
    out_lines.append(f"Baselines (predict 0): " + ", ".join(f"{op}: {BASELINES[op]:.2f}" for op in LOGICAL_OPS))
    out_lines.append("")

    # Sweep over readout × lr × seed
    out_lines.append("=" * 78)
    out_lines.append("Sweep: readout_slot × lr × seed (5 seeds)")
    out_lines.append("=" * 78)

    seeds = [0, 1, 2, 3, 4]
    lrs = [1e-6, 1e-5, 1e-4, 1e-3]
    readouts = list(range(N))
    n_steps = 30000

    all_results = {}  # (readout, lr) -> list of seed runs
    for readout in readouts:
        out_lines.append(f"\n--- readout_slot = {readout} ---")
        for lr in lrs:
            runs = []
            for seed in seeds:
                print(f"  readout={readout}, lr={lr:.0e}, seed={seed}...")
                r = train_run(seed=seed, n_steps=n_steps, lr=lr, readout_slot=readout)
                runs.append(r)
            all_results[(readout, lr)] = runs
            line = f"  lr={lr:.0e}"
            for op in LOGICAL_OPS:
                ratios = [r['eval_ratio_to_baseline'][op] for r in runs]
                if any(not math.isfinite(x) for x in ratios):
                    line += f"  {op}: NaN/inf"
                else:
                    line += f"  {op}: {np.mean(ratios):>6.4f}±{np.std(ratios):.4f}"
            out_lines.append(line)

    # Best (readout, lr) by sum of ratios
    out_lines.append("\n" + "=" * 78)
    out_lines.append("Best (readout, lr) by mean SUM of per-op eval-ratios across seeds")
    out_lines.append("=" * 78)
    scored = []
    for key, runs in all_results.items():
        ratios = [sum(r['eval_ratio_to_baseline'].values()) for r in runs]
        if any(not math.isfinite(x) for x in ratios):
            continue
        scored.append((key, float(np.mean(ratios))))
    scored.sort(key=lambda x: x[1])
    out_lines.append("")
    for key, score in scored[:10]:
        readout, lr = key
        out_lines.append(f"  readout={readout}, lr={lr:.0e}: total ratio = {score:.4f}")

    if scored:
        best_key = scored[0][0]
        best_runs = all_results[best_key]
        out_lines.append("")
        out_lines.append(f"=== Best: readout={best_key[0]}, lr={best_key[1]:.0e} ===")
        for op in LOGICAL_OPS:
            losses = [r['eval_loss_per_op'][op] for r in best_runs]
            ratios = [r['eval_ratio_to_baseline'][op] for r in best_runs]
            out_lines.append(
                f"  {op}: loss = {np.mean(losses):>10.4f} ± {np.std(losses):.4f}   "
                f"ratio = {np.mean(ratios):.4f}× ± {np.std(ratios):.4f}"
            )
        # Trajectory of best
        log0 = best_runs[0]['log']
        sample_indices = [0, len(log0['step']) // 10, len(log0['step']) // 4,
                          len(log0['step']) // 2, len(log0['step']) - 1]
        out_lines.append("")
        out_lines.append("  Loss trajectory (seed 0, every fifth-of-run):")
        out_lines.append(f"    {'step':>8}  " + "  ".join(f"loss_{op:<2s}".rjust(11) for op in LOGICAL_OPS))
        for idx in sample_indices:
            s = log0['step'][idx]
            ls = log0['loss_per_op'][idx]
            out_lines.append(f"    {s:>8}  " + "  ".join(f"{ls[op]:>11.4f}" for op in LOGICAL_OPS))

        out_lines.append("")
        out_lines.append("  Final bias (seed 0):")
        out_lines.append(f"    {best_runs[0]['final_bias']}")

    text = "\n".join(out_lines)
    Path(__file__).parent.joinpath("v10_two_op_results.txt").write_text(text, encoding="utf-8")

    json_data = {
        'provenance': {'commit': commit, 'started': started},
        'baselines': BASELINES,
        'all_results': {
            f"readout={k[0]}_lr={k[1]:.0e}": v for k, v in all_results.items()
        },
        'best': {
            'readout': scored[0][0][0] if scored else None,
            'lr': scored[0][0][1] if scored else None,
            'total_ratio': scored[0][1] if scored else None,
        },
    }
    Path(__file__).parent.joinpath("v10_two_op_data.json").write_text(
        json.dumps(json_data, indent=2), encoding="utf-8"
    )
    print("\nResults written.")


if __name__ == "__main__":
    main()
