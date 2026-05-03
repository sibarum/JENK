"""v11: same insight as v10 (collapse {+,-,*,/} to two architectural ops via
input filters) but at N=7 (v2 layout) so the bias has 7 dims instead of 5.

The v10 N=5 result fell short: even at the best readout, + and − ended up
at ratio > 1 (worse than predict-zero), suggesting the 5-dim bias couldn't
satisfy both + and * constraints simultaneously at any single readout.

Slot layout (v2-compatible):
  k_0 = in_0 (a)
  k_1 = (zero — was op_- slot in v2; unused here)
  k_2 = op_+         one-hot for {+, −}
  k_3 = op_*         one-hot for {*, /}
  k_4 = (zero — was op_/ slot in v2; unused here)
  k_5 = in_1 (b' = b for {+, *};  -b for {-};  1/b for {/})
  k_6 = out_0 (= 1)

Same shared-bias, no learned k. b is preprocessed deterministically based on
which user-facing op is requested.
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
    V2_OP_ADD,    # slot 2
    V2_OP_MUL,    # slot 3
    V2_OUT_0,
)


TARGETS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}
LOGICAL_OPS = ['+', '-', '*', '/']


def encode_two_op_n7(a: float, op: str, b: float) -> Projective:
    """v2 slot layout, N=7, but only op_+ and op_* are activated.
    Subtraction and division are routed via input transform on b.
    """
    if b == 0:
        raise ValueError("b=0 disallowed")
    coeffs = np.zeros(N_V2)
    coeffs[V2_IN_0] = a
    coeffs[V2_OUT_0] = 1.0

    if op == '+':
        coeffs[V2_OP_ADD] = 1.0
        coeffs[V2_IN_1] = b
    elif op == '-':
        coeffs[V2_OP_ADD] = 1.0
        coeffs[V2_IN_1] = -b
    elif op == '*':
        coeffs[V2_OP_MUL] = 1.0
        coeffs[V2_IN_1] = b
    elif op == '/':
        coeffs[V2_OP_MUL] = 1.0
        coeffs[V2_IN_1] = 1.0 / b
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


def train_run(seed=0, n_steps=30000, lr=1e-5, readout_slot=0, log_every=500):
    rng = random.Random(seed)
    neuron = SquaredInputProjectiveNeuron(
        bias=Projective.zeros(N_V2), readout_slot=readout_slot,
    )
    rolling = {op: [] for op in LOGICAL_OPS}
    log = {'step': [], 'loss_per_op': []}
    skipped = 0
    diverged = False
    for step in range(n_steps):
        op = rng.choice(LOGICAL_OPS)
        a, b = sample_ab(rng)
        target = TARGETS[op](a, b)
        input_ = encode_two_op_n7(a, op, float(b))
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
            'seed': seed, 'lr': lr, 'readout_slot': readout_slot,
            'final_bias': neuron.bias.coeffs.tolist(),
            'eval_loss_per_op': {op: float('inf') for op in LOGICAL_OPS},
            'eval_ratio_to_baseline': {op: float('inf') for op in LOGICAL_OPS},
            'log': log, 'skipped_steps': skipped, 'diverged': True,
            'diverged_at_step': step,
        }

    eval_rng = random.Random(seed + 100000)
    eval_losses = {op: [] for op in LOGICAL_OPS}
    for _ in range(500):
        for op in LOGICAL_OPS:
            a, b = sample_ab(eval_rng)
            target = TARGETS[op](a, b)
            pred = neuron.predict(encode_two_op_n7(a, op, float(b)))
            eval_losses[op].append((pred - target) ** 2)
    eval_mean = {op: float(np.mean(eval_losses[op])) for op in LOGICAL_OPS}
    eval_ratio = {op: eval_mean[op] / BASELINES[op] for op in LOGICAL_OPS}
    return {
        'seed': seed, 'lr': lr, 'readout_slot': readout_slot,
        'final_bias': neuron.bias.coeffs.tolist(),
        'eval_loss_per_op': eval_mean,
        'eval_ratio_to_baseline': eval_ratio,
        'log': log, 'skipped_steps': skipped, 'diverged': False,
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

    out = []
    out.append("v11 two-op architecture at N=7 — results")
    out.append("=" * 78)
    out.append(f"Provenance: commit {commit}, started {started}")
    out.append("Architecture: N=7 v2 layout; op_+ at slot 2, op_* at slot 3.")
    out.append("Slots 1 (was op_-) and 4 (was op_/) are zero — - and / use input transforms.")
    out.append("Slots: k_0=in_0 k_1=0 k_2=op_+ k_3=op_* k_4=0 k_5=in_1(b') k_6=out_0(=1)")
    out.append("")
    out.append(f"Baselines: " + ", ".join(f"{op}: {BASELINES[op]:.2f}" for op in LOGICAL_OPS))
    out.append("")

    seeds = [0, 1, 2, 3, 4]
    lrs = [1e-7, 1e-6, 1e-5, 1e-4]
    readouts = list(range(N_V2))
    n_steps = 30000

    all_results = {}
    for readout in readouts:
        out.append(f"\n--- readout_slot = {readout} ---")
        for lr in lrs:
            runs = []
            for seed in seeds:
                print(f"  readout={readout}, lr={lr:.0e}, seed={seed}...")
                runs.append(train_run(seed=seed, n_steps=n_steps, lr=lr, readout_slot=readout))
            all_results[(readout, lr)] = runs
            line = f"  lr={lr:.0e}"
            for op in LOGICAL_OPS:
                ratios = [r['eval_ratio_to_baseline'][op] for r in runs]
                if any(not math.isfinite(x) for x in ratios):
                    line += f"  {op}: NaN/inf"
                else:
                    line += f"  {op}: {np.mean(ratios):>6.4f}±{np.std(ratios):.4f}"
            out.append(line)

    # Best by sum of ratios
    out.append("\n" + "=" * 78)
    out.append("Best (readout, lr) by SUM of per-op eval-ratios across seeds")
    out.append("=" * 78)
    scored = []
    for key, runs in all_results.items():
        ratios = [sum(r['eval_ratio_to_baseline'].values()) for r in runs]
        if any(not math.isfinite(x) for x in ratios):
            continue
        scored.append((key, float(np.mean(ratios))))
    scored.sort(key=lambda x: x[1])
    out.append("")
    for key, score in scored[:10]:
        readout, lr = key
        out.append(f"  readout={readout}, lr={lr:.0e}: total ratio = {score:.4f}")

    if scored:
        best_key = scored[0][0]
        best_runs = all_results[best_key]
        out.append("")
        out.append(f"=== Best: readout={best_key[0]}, lr={best_key[1]:.0e} ===")
        for op in LOGICAL_OPS:
            losses = [r['eval_loss_per_op'][op] for r in best_runs]
            ratios = [r['eval_ratio_to_baseline'][op] for r in best_runs]
            out.append(
                f"  {op}: loss = {np.mean(losses):>11.4f} ± {np.std(losses):.4f}   "
                f"ratio = {np.mean(ratios):.4f}× ± {np.std(ratios):.4f}"
            )
        log0 = best_runs[0]['log']
        if log0['step']:
            sample_indices = [0, len(log0['step']) // 10, len(log0['step']) // 4,
                              len(log0['step']) // 2, len(log0['step']) - 1]
            out.append("")
            out.append("  Loss trajectory (seed 0):")
            out.append(f"    {'step':>8}  " + "  ".join(f"loss_{op:<2s}".rjust(11) for op in LOGICAL_OPS))
            for idx in sample_indices:
                s = log0['step'][idx]
                ls = log0['loss_per_op'][idx]
                out.append(f"    {s:>8}  " + "  ".join(f"{ls[op]:>11.4f}" for op in LOGICAL_OPS))
        out.append("")
        out.append(f"  Final bias (seed 0): {best_runs[0]['final_bias']}")

    text = "\n".join(out)
    Path(__file__).parent.joinpath("v11_two_op_n7_results.txt").write_text(text, encoding="utf-8")
    Path(__file__).parent.joinpath("v11_two_op_n7_data.json").write_text(
        json.dumps({
            'provenance': {'commit': commit, 'started': started},
            'baselines': BASELINES,
            'all_results': {f"readout={k[0]}_lr={k[1]:.0e}": v for k, v in all_results.items()},
            'best': {'key': scored[0][0] if scored else None, 'score': scored[0][1] if scored else None},
        }, indent=2),
        encoding="utf-8",
    )
    print("\nResults written.")


if __name__ == "__main__":
    main()
