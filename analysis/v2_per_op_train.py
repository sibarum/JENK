"""v2 single-op vs multi-op comparison — refactored, all four ops included.

Trains output[0] readout on each op individually, then on growing subsets,
including division. Reports per-op losses against a predict-zero baseline
so each cell is interpretable.

Single source of truth: TARGETS at module level. The original exclusion
of division was a baked assumption in 6 places; we now include it and let
the loss curve speak.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from symbolic.projective import (
    Projective,
    SquaredInputProjectiveNeuron,
    N_V2,
    encode_problem_v2,
)


# Single source of truth — change here, change everywhere.
TARGETS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}
OPS = list(TARGETS.keys())


def sample_problem(rng, op_set):
    """Pick a random op from op_set and operands; retry div on b=0."""
    while True:
        op = rng.choice(op_set)
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if op == '/' and b == 0:
            continue
        return op, a, b


def train(op_set, n_steps=30000, lr=1e-5, seed=20260502):
    rng = random.Random(seed)
    neuron = SquaredInputProjectiveNeuron(
        bias=Projective.zeros(N_V2),
        readout_slot=0,
    )
    last_losses = {op: [] for op in op_set}
    for step in range(n_steps):
        op, a, b = sample_problem(rng, op_set)
        target = TARGETS[op](a, b)
        input_ = encode_problem_v2(a, op, b)
        pred = neuron.step(input_, float(target), lr)
        if step >= n_steps - 5000:
            last_losses[op].append((pred - target) ** 2)

    bias = neuron.bias.coeffs.copy()
    avg_losses = {
        op: (sum(ls) / len(ls)) if ls else float('nan')
        for op, ls in last_losses.items()
    }
    return bias, avg_losses


def evaluate_on_all_ops(bias, n_eval=1000, seed=42):
    """For a given bias, compute avg per-op loss across n_eval random problems."""
    neuron = SquaredInputProjectiveNeuron(
        bias=Projective(bias.copy()), readout_slot=0
    )
    rng = random.Random(seed)
    losses = {op: [] for op in OPS}
    for _ in range(n_eval):
        for op in OPS:
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            if op == '/' and b == 0:
                continue
            target = TARGETS[op](a, b)
            pred = neuron.predict(encode_problem_v2(a, op, b))
            losses[op].append((pred - target) ** 2)
    return {op: sum(ls) / len(ls) for op, ls in losses.items()}


def baseline_per_op(n=10000, seed=42):
    """Baseline: predict 0 always, compute avg loss per op."""
    rng = random.Random(seed)
    losses = {op: [] for op in OPS}
    for _ in range(n):
        for op in OPS:
            a = rng.randint(-10, 10)
            b = rng.randint(-10, 10)
            if op == '/' and b == 0:
                continue
            target = TARGETS[op](a, b)
            losses[op].append(target ** 2)
    return {op: sum(ls) / len(ls) for op, ls in losses.items()}


def main() -> None:
    # Scenarios: each op alone + growing combinations.
    scenarios = [(op,) for op in OPS]
    scenarios.append(('+', '-'))
    scenarios.append(('+', '-', '*'))
    scenarios.append(tuple(OPS))

    print("Baseline (predict 0 always):")
    baseline = baseline_per_op()
    for op in OPS:
        print(f"  {op}: {baseline[op]:.4f}")
    print()

    print("Single-op and multi-op training scenarios at output[0]:\n")
    for ops_subset in scenarios:
        bias, train_losses = train(list(ops_subset))
        eval_losses = evaluate_on_all_ops(bias)
        ops_str = ','.join(ops_subset)
        print(f"--- Trained on {{{ops_str}}} ---")
        print(f"  bias: {np.array2string(bias, precision=4, suppress_small=True)}")
        print(f"  Final training loss (last 5K steps):")
        for op in ops_subset:
            print(f"    {op}: {train_losses[op]:.4f}")
        print(f"  Eval loss on all ops with this bias (vs baseline):")
        for op in OPS:
            ratio = eval_losses[op] / baseline[op] if baseline[op] > 0 else float('nan')
            print(f"    {op}: {eval_losses[op]:.4f}  (baseline {baseline[op]:.2f}, ratio {ratio:.2f}x)")
        print()


if __name__ == "__main__":
    main()
