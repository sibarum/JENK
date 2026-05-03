"""v2 multi-op training run — all four ops including division.

Refactored from earlier version that excluded division based on an
analytical assumption that may be wrong (the polynomial-floor argument
depends on the mixed-product rule being faithful to the framework, which
we now suspect it isn't). Single source of truth for the op set; division
gets a chance to fit and we let the loss curve speak.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from symbolic.projective import (
    Projective,
    SquaredInputProjectiveNeuron,
    N_V2,
    encode_problem_v2,
)


LOG_PATH = REPO_ROOT / "tmp" / "diagnostics" / "v2_train.log"


# Single source of truth — change here, change everywhere.
TARGETS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}
OPS = list(TARGETS.keys())


def sample_problem(rng):
    """Pick a random op and operands. For division, retry until b != 0."""
    while True:
        op = rng.choice(OPS)
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if op == '/' and b == 0:
            continue
        return op, a, b


def main() -> None:
    rng = random.Random(20260502)
    neuron = SquaredInputProjectiveNeuron(
        bias=Projective.zeros(N_V2),
        readout_slot=0,
    )
    lr = 1e-5
    n_steps = 50_000

    header = "step\top\ta\tb\ttarget\tprediction\tloss\t" + "\t".join(
        f"e{i}" for i in range(N_V2)
    )
    log = [header]

    for step in range(n_steps):
        op, a, b = sample_problem(rng)
        target = TARGETS[op](a, b)
        input_ = encode_problem_v2(a, op, b)
        pred = neuron.step(input_, float(target), lr)
        loss = (pred - target) ** 2
        biases = "\t".join(f"{c}" for c in neuron.bias.coeffs)
        log.append(f"{step}\t{op}\t{a}\t{b}\t{target}\t{pred}\t{loss}\t{biases}")

    LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")

    # Per-op loss summary over the last 5000 steps.
    last_n = 5000
    per_op = {op: [] for op in OPS}
    for line in log[-last_n:]:
        parts = line.split('\t')
        per_op[parts[1]].append(float(parts[6]))

    print(f"Final bias: {neuron.bias.coeffs}")
    print(f"\nPer-op average loss over last {last_n} steps:")
    for op in OPS:
        if per_op[op]:
            avg = sum(per_op[op]) / len(per_op[op])
            print(f"  {op}: avg loss = {avg:.6f}  (n={len(per_op[op])})")
        else:
            print(f"  {op}: (no samples in last {last_n} steps)")


if __name__ == "__main__":
    main()
