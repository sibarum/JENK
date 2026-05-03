"""v5 training benchmark — does reciprocal-input encoding actually train division?

Tests whether SGD finds the analytically-feasible bias for division (verified
in v5_reciprocal_input_feasibility.py) when trained with reciprocal-input
encoding. Compares against identity-input encoding across all four ops.

Configurations:
  encoding: 'identity' (in_1 = b), 'reciprocal' (in_1 = 1/b)
  op:       +, -, *, /
  lr:       1e-7, 1e-6, 1e-5, 1e-4, 1e-3
  seed:     0, 1, 2, 3, 4
  n_steps:  30000
  readout:  0
  range:    a, b ∈ {-10..10}, integer, b ≠ 0 always (matches reciprocal encoder)

Records per run:
  final eval loss (held-out 500 samples), final bias coefficients,
  windowed training loss (last 1000 steps).

Aggregates per (encoding, op, lr):
  mean ± std eval loss across seeds, ratio to predict-0 baseline, SNR.

Reports best LR per (encoding, op).
Sanity check: best reciprocal-/ bias close to analytical output[0] solution
  (free, 1/2, 1/2, -1/2, 1/2, 0, -1/2)?

Predicted shape (from v5 analytical):
  identity   on {+, -, *}: low loss (degree-2 polynomial fits exactly)
  identity   on /:         hits baseline (polynomial can't represent a/b)
  reciprocal on /:         low loss (analytical exact solution exists)
  reciprocal on {+, -, *}: hits baseline or worse (encoding can't fit those)

Provenance recorded inside output file (commit, command, time).
"""

from __future__ import annotations

import json
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


def encode_identity(a: float, op: str, b: float) -> Projective:
    coeffs = np.zeros(N_V2)
    coeffs[V2_IN_0] = a
    coeffs[V2_OP_SLOT[op]] = 1.0
    coeffs[V2_IN_1] = b
    coeffs[V2_OUT_0] = 1.0
    return Projective(coeffs)


def encode_reciprocal(a: float, op: str, b: float) -> Projective:
    if b == 0:
        raise ValueError("b=0 disallowed in reciprocal encoding")
    coeffs = np.zeros(N_V2)
    coeffs[V2_IN_0] = a
    coeffs[V2_OP_SLOT[op]] = 1.0
    coeffs[V2_IN_1] = 1.0 / b
    coeffs[V2_OUT_0] = 1.0
    return Projective(coeffs)


ENCODERS = {
    'identity': encode_identity,
    'reciprocal': encode_reciprocal,
}


def sample_ab(rng):
    """Pick (a, b) with b != 0; identical for both encodings."""
    while True:
        a = rng.randint(-10, 10)
        b = rng.randint(-10, 10)
        if b != 0:
            return a, b


def train_run(
    encoding: str, op: str, lr: float, seed: int, n_steps: int, readout: int = 0
):
    rng = random.Random(seed)
    encoder = ENCODERS[encoding]
    target_fn = TARGETS[op]

    neuron = SquaredInputProjectiveNeuron(
        bias=Projective.zeros(N_V2),
        readout_slot=readout,
    )

    tail_n = 1000
    train_losses_tail = []
    for step in range(n_steps):
        a, b = sample_ab(rng)
        target = target_fn(a, b)
        input_ = encoder(a, op, float(b))
        pred = neuron.step(input_, float(target), lr)
        if step >= n_steps - tail_n:
            train_losses_tail.append((pred - target) ** 2)

    bias = neuron.bias.coeffs.copy()
    train_loss = float(np.mean(train_losses_tail)) if train_losses_tail else float('nan')

    # Held-out eval — fresh seed, no overlap with training stream.
    eval_rng = random.Random(seed + 100000)
    eval_losses = []
    n_eval = 500
    for _ in range(n_eval):
        a, b = sample_ab(eval_rng)
        target = target_fn(a, b)
        input_ = encoder(a, op, float(b))
        pred = neuron.predict(input_)
        eval_losses.append((pred - target) ** 2)
    eval_loss = float(np.mean(eval_losses))

    return bias, train_loss, eval_loss


def baseline_loss(op: str, n: int = 5000, seed: int = 999) -> float:
    rng = random.Random(seed)
    target_fn = TARGETS[op]
    losses = []
    for _ in range(n):
        a, b = sample_ab(rng)
        target = target_fn(a, b)
        losses.append(target ** 2)  # predict 0
    return float(np.mean(losses))


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
    encodings = ['identity', 'reciprocal']
    lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    seeds = list(range(5))
    n_steps = 30000

    commit = get_commit_hash()
    dirty = get_git_dirty()
    started = time.strftime('%Y-%m-%d %H:%M:%S')

    total_runs = len(encodings) * len(OPS) * len(lrs) * len(seeds)
    print(f"v5 per-op training benchmark")
    print(f"  commit: {commit}")
    print(f"  git status: {dirty[:80]}")
    print(f"  started: {started}")
    print(
        f"  config: {len(encodings)} encodings × {len(OPS)} ops × "
        f"{len(lrs)} LRs × {len(seeds)} seeds × {n_steps} steps "
        f"= {total_runs} runs"
    )

    baselines = {op: baseline_loss(op) for op in OPS}
    print(f"  baselines (predict 0): {baselines}")
    print()

    records = []
    run_idx = 0
    for encoding in encodings:
        for op in OPS:
            for lr in lrs:
                eval_losses = []
                train_losses = []
                final_biases = []
                for seed in seeds:
                    bias, tl, el = train_run(encoding, op, lr, seed, n_steps)
                    eval_losses.append(el)
                    train_losses.append(tl)
                    final_biases.append(bias)
                    run_idx += 1

                mean_loss = float(np.mean(eval_losses))
                std_loss = float(np.std(eval_losses))
                baseline = baselines[op]
                signal = baseline - mean_loss
                snr = signal / std_loss if std_loss > 1e-12 else float('inf')
                ratio = mean_loss / baseline

                records.append({
                    'encoding': encoding,
                    'op': op,
                    'lr': lr,
                    'mean_eval_loss': mean_loss,
                    'std_eval_loss': std_loss,
                    'baseline': baseline,
                    'signal': signal,
                    'snr': snr,
                    'ratio_to_baseline': ratio,
                    'eval_losses': eval_losses,
                    'train_losses': train_losses,
                    'final_biases': [b.tolist() for b in final_biases],
                })

                print(
                    f"  [{run_idx:3d}/{total_runs}] {encoding:10s} {op}  lr={lr:.0e}  "
                    f"loss={mean_loss:>10.3f}  ±{std_loss:<10.3f}  "
                    f"ratio={ratio:>8.3f}x  SNR={snr:>8.2f}"
                )

    elapsed = time.time() - time.mktime(time.strptime(started, '%Y-%m-%d %H:%M:%S'))
    print(f"\ndone in {elapsed:.0f}s")

    out_lines = []
    out_lines.append("v5 per-op training benchmark — results")
    out_lines.append("=" * 78)
    out_lines.append(f"Provenance:")
    out_lines.append(f"  commit: {commit}")
    out_lines.append(f"  git status at run: {dirty}")
    out_lines.append(f"  started: {started}")
    out_lines.append(f"  encodings: {encodings}")
    out_lines.append(f"  ops: {OPS}")
    out_lines.append(f"  lrs: {lrs}")
    out_lines.append(f"  seeds: {seeds}")
    out_lines.append(f"  n_steps: {n_steps}")
    out_lines.append(f"  readout_slot: 0")
    out_lines.append(f"  sample range: a, b in {{-10..10}}, b != 0 always")
    out_lines.append(f"  eval: 500 held-out samples, seed = train_seed + 100000")
    out_lines.append(f"  baselines (predict 0): {baselines}")
    out_lines.append("")

    for encoding in encodings:
        out_lines.append(f"=== Encoding: {encoding} ===")
        out_lines.append(
            f"  {'op':<3} {'lr':<10} {'mean':>10} {'±std':>10} "
            f"{'baseline':>10} {'ratio':>10} {'SNR':>10}"
        )
        for op in OPS:
            for lr in lrs:
                rec = next(
                    r for r in records
                    if r['encoding'] == encoding and r['op'] == op and r['lr'] == lr
                )
                out_lines.append(
                    f"  {op:<3} {lr:<10.0e} {rec['mean_eval_loss']:>10.3f} "
                    f"{rec['std_eval_loss']:>10.3f} {rec['baseline']:>10.1f} "
                    f"{rec['ratio_to_baseline']:>9.3f}x {rec['snr']:>10.2f}"
                )
            out_lines.append("")

    out_lines.append("=" * 78)
    out_lines.append("Best LR per (encoding, op) — minimum mean eval loss")
    out_lines.append("=" * 78)
    out_lines.append("")
    out_lines.append(
        f"  {'encoding':<10} {'op':<3} {'best_lr':<10} "
        f"{'mean':>10} {'±std':>10} {'ratio':>10} {'SNR':>10}"
    )
    for encoding in encodings:
        for op in OPS:
            recs = [
                r for r in records if r['encoding'] == encoding and r['op'] == op
            ]
            best = min(recs, key=lambda r: r['mean_eval_loss'])
            out_lines.append(
                f"  {encoding:<10} {best['op']:<3} {best['lr']:<10.0e} "
                f"{best['mean_eval_loss']:>10.3f} {best['std_eval_loss']:>10.3f} "
                f"{best['ratio_to_baseline']:>9.3f}x {best['snr']:>10.2f}"
            )

    out_lines.append("")
    out_lines.append("=" * 78)
    out_lines.append("Sanity check vs analytical v5 solution (reciprocal /, output[0])")
    out_lines.append("=" * 78)
    out_lines.append("")
    out_lines.append("Analytical: bias = (free, 1/2, 1/2, -1/2, 1/2, 0, -1/2)")
    out_lines.append("  → e1=0.5, e2=0.5, e3=-0.5, e4=0.5, e5=0, e6=-0.5; e0 free")
    out_lines.append("")

    rec = next(r for r in records if r['encoding'] == 'reciprocal' and r['op'] == '/')
    best_recip = min(
        [r for r in records if r['encoding'] == 'reciprocal' and r['op'] == '/'],
        key=lambda r: r['mean_eval_loss'],
    )
    target_bias = np.array([np.nan, 0.5, 0.5, -0.5, 0.5, 0.0, -0.5])
    out_lines.append(f"Trained at best lr = {best_recip['lr']:.0e}, n_steps = {n_steps}:")
    for seed_idx, bias in enumerate(best_recip['final_biases']):
        bias_arr = np.array(bias)
        # Compare nontrivial slots only (skip e0 which is free).
        diff = bias_arr[1:] - target_bias[1:]
        l2 = float(np.linalg.norm(diff))
        out_lines.append(
            f"  seed {seeds[seed_idx]}: bias = {bias_arr.tolist()}, "
            f"L2 distance from analytical (skip e0) = {l2:.4f}"
        )

    out_lines.append("")
    out_lines.append("=" * 78)
    out_lines.append("Predicted vs observed shape")
    out_lines.append("=" * 78)
    out_lines.append("")
    out_lines.append("v5 analytical predictions:")
    out_lines.append("  identity   on {+,-,*}:  exactly fittable -> low loss expected")
    out_lines.append("  identity   on /:        unfittable (poly-vs-rational) -> ratio ≈ 1.0× baseline")
    out_lines.append("  reciprocal on /:        exact solution exists -> low loss expected")
    out_lines.append("  reciprocal on {+,-,*}:  unfittable -> ratio ≥ 1.0× baseline")

    text = "\n".join(out_lines)
    out_path = Path(__file__).parent.joinpath("v5_per_op_train_results.txt")
    out_path.write_text(text, encoding="utf-8")

    json_path = Path(__file__).parent.joinpath("v5_per_op_train_data.json")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"\nResults written to:")
    print(f"  {out_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
