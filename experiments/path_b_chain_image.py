"""Stage-5c: empirical chain-image test.

The Stage-3 finding established that M_proj = [[0, 0], [0, 1]] is
*outside* the image of any 2-layer chain in the framework — the chain
composes determinant-preserving maps up to scaling and lands in
SL(2)-flavored territory, while M_proj has det=0 (rank 1). The
constraint d1·d2·(s2-s1)=0 is necessary but not sufficient for
reachability. So a 2-layer student trained against M_proj-generated
data should NOT reach zero loss — it should bottom out at a
measurable floor.

This experiment puts that prediction on a measurement scale:

  Part A: 2-layer student vs in-image teacher (s_target=0.7 in both
          layers). Should converge to ~zero loss.
  Part B: 2-layer student vs M_proj target. Should hit a non-zero floor.
  Part C: depth sweep (N=1, 2, 3, 4) against M_proj. Does the floor
          drop as depth grows? At what N (if any) does it break?

A drops to ~0 confirms the training infrastructure works on chains.
B hitting a non-zero floor confirms the theoretical prediction
empirically. C maps the chain image's growth with depth.

Run:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. conda run -n traction \
        python experiments/path_b_chain_image.py
"""

from __future__ import annotations

import sys
import numpy as np

from jenk.chain_learnable import ChainLearnable
from jenk.learnable_neuron import (
    LearnableNeuron,
    LearnableParameter,
    SymbolicOptimizer,
    squared_error_loss,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

def in_image_teacher_pairs(s_value: float, n: int = 5):
    """Generate (input, output) pairs from a two-layer in-image teacher.

    Both teacher layers at s=s_value, c=0, d=1. The composition is
    guaranteed to live in the 2-layer chain image.
    """
    teacher1 = LearnableNeuron(s=s_value, c=0, d=1)
    teacher2 = LearnableNeuron(s=s_value, c=0, d=1)
    teacher = ChainLearnable([teacher1, teacher2])

    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(n):
        x_a = float(rng.uniform(-1, 1))
        x_b = float(rng.uniform(-1, 1))
        out_a, out_b = teacher.forward(x_a, x_b)
        pairs.append(((x_a, x_b), (float(out_a), float(out_b))))
    return pairs


def m_proj_pairs(n: int = 5):
    """Pairs (input, M_proj·input) where M_proj = [[0, 0], [0, 1]].

    For input (x_a, x_b), M_proj·input = (0, x_b).
    """
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(n):
        x_a = float(rng.uniform(-1, 1))
        x_b = float(rng.uniform(-1, 1))
        pairs.append(((x_a, x_b), (0.0, x_b)))
    return pairs


# -----------------------------------------------------------------------------
# Student building
# -----------------------------------------------------------------------------

def build_student(depth: int, *, init_seed: int = 0):
    """Build a depth-N student chain with all 3*N parameters learnable.

    Each layer has learnable s, c, d (no biases for simplicity).
    Initialization: s near 0, c near 1, d near 1 with a small random offset
    per layer so we don't start at a saturating point of the loss landscape.
    """
    rng = np.random.default_rng(init_seed)
    layers = []
    params = []
    for i in range(depth):
        s = LearnableParameter.named(f's{i}', float(rng.uniform(-0.5, 0.5)))
        c = LearnableParameter.named(f'c{i}', float(rng.uniform(0.5, 1.5)))
        d = LearnableParameter.named(f'd{i}', float(rng.uniform(0.5, 1.5)))
        layers.append(LearnableNeuron(s=s, c=c, d=d))
        params.extend([s, c, d])
    return ChainLearnable(layers), params


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_chain(chain: ChainLearnable, pairs, *,
                lr: float = 0.05, n_steps: int = 600, log_every: int | None = None):
    """Train the chain on pairs. Returns final loss + loss curve."""
    loss_expr = squared_error_loss(chain, pairs)
    opt = SymbolicOptimizer(chain, learning_rate=lr)

    losses = []
    for step in range(n_steps):
        L = opt.loss_value(loss_expr)
        losses.append(L)
        if log_every and step % log_every == 0:
            print(f'    step {step:>4}: loss = {L:.6e}')
        opt.step(loss_expr)

    final = opt.loss_value(loss_expr)
    return final, losses


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def part_a_in_image(n_pairs: int = 5):
    print('=' * 78)
    print('Part A: 2-layer student vs in-image 2-layer teacher (s=0.7 both layers)')
    print('  Theory predicts: reachable. Loss should converge to ~0.')
    print('=' * 78)

    pairs = in_image_teacher_pairs(s_value=0.7, n=n_pairs)
    student, _ = build_student(depth=2, init_seed=1)
    print(f'  Training {len(pairs)} pairs, depth 2, lr 0.05, 600 steps...')
    final, losses = train_chain(student, pairs, lr=0.05, n_steps=600, log_every=100)

    print(f'  Final loss:        {final:.6e}')
    print(f'  Per-pair loss:     {final / n_pairs:.6e}')
    print(f'  Verdict:           {"CONVERGED" if final < 1e-4 else "did NOT converge"}')
    print()
    return final, losses


def part_b_m_proj(n_pairs: int = 5):
    print('=' * 78)
    print('Part B: 2-layer student vs M_proj = [[0, 0], [0, 1]]')
    print('  Theory predicts: outside chain image. Loss should NOT reach 0.')
    print('=' * 78)

    pairs = m_proj_pairs(n=n_pairs)
    student, _ = build_student(depth=2, init_seed=1)
    print(f'  Training {len(pairs)} pairs, depth 2, lr 0.05, 600 steps...')
    final, losses = train_chain(student, pairs, lr=0.05, n_steps=600, log_every=100)

    print(f'  Final loss:        {final:.6e}')
    print(f'  Per-pair loss:     {final / n_pairs:.6e}')
    print(f'  Verdict:           {"FLOORED (consistent with theory)" if final > 1e-4 else "REACHED ZERO (contradicts theory)"}')
    print()
    return final, losses


def part_c_depth_sweep(depths: list[int], n_pairs: int = 5):
    print('=' * 78)
    print('Part C: depth sweep against M_proj — does the floor break with more layers?')
    print('=' * 78)

    pairs = m_proj_pairs(n=n_pairs)
    results = []
    for depth in depths:
        student, _ = build_student(depth=depth, init_seed=1)
        # lr scales down with depth — gradients grow non-linearly with chain length,
        # so a constant lr that works at depth 2 overflows at depth 3+.
        depth_lr = 0.05 if depth <= 2 else 0.02 / max(1, depth - 2)
        final, _ = train_chain(student, pairs, lr=depth_lr, n_steps=600, log_every=None)
        results.append((depth, final))
        print(f'  depth {depth} (lr={depth_lr}): final loss = {final:.6e} ({final / n_pairs:.6e} per pair)')

    print()
    print('  Reading the sweep:')
    print('    - depth 1: provably can\'t reach M_proj (single det-preserving layer)')
    print('    - depth 2: provably outside chain image per Stage-3 finding')
    print('    - depth 3+: empirical, would tell us if/when M_proj enters reach')
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print('=' * 78)
    print('Stage-5c: empirical chain-image test')
    print('=' * 78)
    print()
    print('Theoretical claim being tested empirically:')
    print('  "M_proj = [[0, 0], [0, 1]] is outside the image of any 2-layer chain")')
    print()

    a_final, _ = part_a_in_image()
    b_final, _ = part_b_m_proj()
    sweep = part_c_depth_sweep(depths=[1, 2, 3, 4])

    print('=' * 78)
    print('Summary')
    print('=' * 78)
    print(f'  Part A (in-image, depth 2):        loss = {a_final:.4e}')
    print(f'  Part B (M_proj, depth 2):          loss = {b_final:.4e}')
    print()
    print('  Depth sweep against M_proj:')
    for depth, loss in sweep:
        print(f'    depth {depth}: loss = {loss:.6e}')
    print()

    if a_final < 1e-4 and b_final > 1e-4:
        print('  HEADLINE: theory confirmed.')
        print('  In-image teacher reached, M_proj floored — exactly as predicted.')
    else:
        print('  HEADLINE: result is off the predicted pattern; needs investigation.')
        print(f'    in-image converged: {a_final < 1e-4}')
        print(f'    M_proj floored:     {b_final > 1e-4}')


if __name__ == '__main__':
    main()
