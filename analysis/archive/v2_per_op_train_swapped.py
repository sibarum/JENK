"""v2 per-op comparison with the SWAPPED encoding (div ↔ in_0).

Slot layout: k_0=op_/  k_1=op_-  k_2=op_+  k_3=op_*  k_4=in_0(a)  k_5=in_1(b)  k_6=out_0(=1)

Cardinal-alignment hypothesis: putting division at slot 0 (the cardinal/pit
position) changes training dynamics even though the analytical cross-op
dead-slot pattern is invariant under the swap (we showed this earlier).
This script tests the empirical effect.

lr = 1e-5, same as v2_per_op_train.py — direct comparison.
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
)


# Swapped slot layout: ops in 0..3 (cardinal band), inputs in 4..6.
SLOT_OP_DIV = 0
SLOT_OP_SUB = 1
SLOT_OP_ADD = 2
SLOT_OP_MUL = 3
SLOT_IN_0 = 4
SLOT_IN_1 = 5
SLOT_OUT_0 = 6

OP_SLOT = {'-': SLOT_OP_SUB, '+': SLOT_OP_ADD, '*': SLOT_OP_MUL, '/': SLOT_OP_DIV}

TARGETS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}
OPS = list(TARGETS.keys())


def encode_problem_swapped(a: float, op: str, b: float) -> Projective:
    coeffs = np.zeros(N_V2)
    coeffs[SLOT_IN_0] = a
    coeffs[OP_SLOT[op]] = 1.0
    coeffs[SLOT_IN_1] = b
    coeffs[SLOT_OUT_0] = 1.0
    return Projective(coeffs)


def sample_problem(rng, op_set):
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
        readout_slot=0,  # same as before — read at cardinal pit position
    )
    last_losses = {op: [] for op in op_set}
    for step in range(n_steps):
        op, a, b = sample_problem(rng, op_set)
        target = TARGETS[op](a, b)
        input_ = encode_problem_swapped(a, op, b)
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
            pred = neuron.predict(encode_problem_swapped(a, op, b))
            losses[op].append((pred - target) ** 2)
    return {op: sum(ls) / len(ls) for op, ls in losses.items()}


def baseline_per_op(n=10000, seed=42):
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
    scenarios = [(op,) for op in OPS]
    scenarios.append(('+', '-'))
    scenarios.append(('+', '-', '*'))
    scenarios.append(tuple(OPS))

    print("=== SWAPPED encoding (div at slot 0) ===")
    print(f"Layout: k_0=op_/  k_1=op_-  k_2=op_+  k_3=op_*  "
          f"k_4=in_0  k_5=in_1  k_6=out_0\n")
    print("Baseline (predict 0 always):")
    baseline = baseline_per_op()
    for op in OPS:
        print(f"  {op}: {baseline[op]:.4f}")
    print()

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
