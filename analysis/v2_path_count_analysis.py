"""v2 path-count derivation: one-hot op encoding, out_0 = 1, N = 7.

Slot layout (per James's planned ordering 1=−, 2=+, 3=*, 4=/):
  k_0 = in_0   (a)
  k_1 = op_-
  k_2 = op_+
  k_3 = op_*
  k_4 = op_/
  k_5 = in_1   (b)
  k_6 = out_0  (= 1 always)

Algebra (N=7):
  k_i · k_i = k_{(i+1) mod 7}
  k_i · k_j = k_{(2i+j) mod 7}   for i ≠ j

For each op, derives  hidden = input · input  symbolically in (a, b), then
the path-count tensor  ∂(output[k])/∂(bias[j])  per output slot.  Reports
which bias slots are live per op, dead per op, and dead-across-all-ops; and
whether each readout's per-sample path-count vector spans its live bias
subspace as (a, b) vary (i.e. is well-conditioned vs degenerate per op).
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp


N = 7
OUT_0_SLOT = 6  # which slot holds out_0 = 1

a, b = sp.symbols('a b')

OP_NAME = {1: 'sub (−)', 2: 'add (+)', 3: 'mul (*)', 4: 'div (/)'}
OP_TARGET = {1: a - b, 2: a + b, 3: a * b, 4: a / b}


def basis_product_index(i: int, j: int) -> int:
    if i == j:
        return (i + 1) % N
    return (2 * i + j) % N


def make_input(active_op_slot: int) -> list:
    """Input with one-hot active op, out_0 = 1, in_0 = a, in_1 = b."""
    inp = [sp.Integer(0)] * N
    inp[0] = a
    inp[active_op_slot] = sp.Integer(1)
    inp[5] = b
    inp[OUT_0_SLOT] = sp.Integer(1)
    return inp


def square(vec: list) -> list:
    """vec · vec under the algebra."""
    hidden = [sp.Integer(0)] * N
    for i in range(N):
        for j in range(N):
            hidden[basis_product_index(i, j)] += vec[i] * vec[j]
    return [sp.expand(h) for h in hidden]


def path_count(hidden: list, k: int) -> list:
    """Coefficient of bias[j] in output[k] = sum_i hidden[i]·bias[j]·[f(i,j)=k]."""
    pc = [sp.Integer(0)] * N
    for j in range(N):
        for i in range(N):
            if basis_product_index(i, j) == k:
                pc[j] += hidden[i]
    return [sp.expand(c) for c in pc]


def is_zero(expr) -> bool:
    return sp.simplify(expr) == 0


def live_subspace_dim_via_polynomial_independence(pc_vector) -> int:
    """How many of these symbolic entries are linearly independent as polys in (a, b)?

    Builds the matrix of polynomial coefficients (one row per nonzero PC entry,
    one column per (a, b) monomial) and computes its rank.  This is the
    dimension of the subspace spanned by the per-sample path-count vectors
    (restricted to the live bias slots) as (a, b) vary.
    """
    nonzero_idx = [j for j, c in enumerate(pc_vector) if not is_zero(c)]
    if not nonzero_idx:
        return 0
    monomials = set()
    for j in nonzero_idx:
        poly = sp.Poly(pc_vector[j], a, b)
        for monom in poly.monoms():
            monomials.add(monom)
    monoms_sorted = sorted(monomials)
    # Row per nonzero PC entry; column per monomial; entry = coefficient.
    M = sp.Matrix(len(nonzero_idx), len(monoms_sorted), lambda r, c: 0)
    for r, j in enumerate(nonzero_idx):
        poly = sp.Poly(pc_vector[j], a, b)
        for monom, coef in poly.terms():
            col = monoms_sorted.index(monom)
            M[r, col] = coef
    return M.rank()


def main() -> None:
    out = []
    out.append("v2 path-count tables — one-hot op, out_0 = 1, N = 7")
    out.append("=" * 72)
    out.append(f"Slot layout: k_0=in_0(a) k_1=op_- k_2=op_+ k_3=op_* k_4=op_/ k_5=in_1(b) k_6=out_0(=1)")
    out.append(f"Targets: 1=a−b, 2=a+b, 3=a·b, 4=a/b")
    out.append("")

    all_results = {}

    for op_slot in [1, 2, 3, 4]:
        inp = make_input(op_slot)
        hidden = square(inp)
        pcs = [path_count(hidden, k) for k in range(N)]
        all_results[op_slot] = {'hidden': hidden, 'pcs': pcs}

        out.append(f"--- Op slot k_{op_slot} active ({OP_NAME[op_slot]}); target = {OP_TARGET[op_slot]} ---")
        out.append("hidden = input · input:")
        for k, h in enumerate(hidden):
            out.append(f"  hidden[{k}] = {h}")
        out.append("")

    out.append("=" * 72)
    out.append("Per-output-slot path-count summary")
    out.append("=" * 72)

    for k in range(N):
        out.append(f"\noutput[{k}]:")
        for op_slot in [1, 2, 3, 4]:
            pc = all_results[op_slot]['pcs'][k]
            live = [j for j, c in enumerate(pc) if not is_zero(c)]
            dead = [j for j in range(N) if j not in live]
            dim = live_subspace_dim_via_polynomial_independence(pc)
            out.append(f"  {OP_NAME[op_slot]:8s} live={live} dead={dead} dim_spanned_over_(a,b)={dim}")
            for j in live:
                out.append(f"           bias[{j}]: {pc[j]}")

    out.append("\n" + "=" * 72)
    out.append("Cross-op structural deadness (bias slot dead under EVERY op)")
    out.append("=" * 72)
    for k in range(N):
        dead_everywhere = set(range(N))
        for op_slot in [1, 2, 3, 4]:
            pc = all_results[op_slot]['pcs'][k]
            live_in_op = {j for j, c in enumerate(pc) if not is_zero(c)}
            dead_everywhere -= live_in_op
        out.append(f"  output[{k}]: dead-in-all-ops bias slots = {sorted(dead_everywhere)}")

    out.append("\n" + "=" * 72)
    out.append("Per-output-slot summary: degeneracy verdict per op")
    out.append("=" * 72)
    out.append("(well-cond = dim_spanned == |live|; degenerate = dim_spanned < |live|)")
    for k in range(N):
        out.append(f"\noutput[{k}]:")
        for op_slot in [1, 2, 3, 4]:
            pc = all_results[op_slot]['pcs'][k]
            live = [j for j, c in enumerate(pc) if not is_zero(c)]
            dim = live_subspace_dim_via_polynomial_independence(pc)
            verdict = "WELL-COND" if dim == len(live) else f"DEGENERATE ({dim}/{len(live)})"
            out.append(f"  {OP_NAME[op_slot]:8s}: |live|={len(live)} dim={dim} → {verdict}")

    text = "\n".join(out)
    print(text)
    Path(__file__).parent.joinpath("v2_path_count_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
