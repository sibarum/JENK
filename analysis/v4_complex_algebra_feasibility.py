"""v4: complex-algebra feasibility — definitive analytical check.

Algebra (algebra-side equivariance, option A):
  k_i · k_i = k_{(i+1) mod N}              (squared, stays real)
  k_i · k_j = i · k_{(2i+j) mod N}         (off-diagonal, factor of i)

Encoding: v2 layout (N = 7, one-hot op + out_0 = 1)
  k_0=in_0(a), k_1=op_-, k_2=op_+, k_3=op_*, k_4=op_/, k_5=in_1(b), k_6=out_0(=1)

Bias: complex per slot, bias_j = Re_j + i·Im_j.  14 real parameters total.

For each candidate single-readout slot k, the prediction is Re(output[k]).
Per-op constraint:
  - polynomial-target ops (+, −, *):  Re(output[k]) = target_p(a, b)
  - division:                          b · Re(output[k]) = a   (polynomial form)

For each readout slot and op subset, builds the linear system in the 14
real bias parameters from polynomial coefficient matching, then checks
joint feasibility via rank consistency.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp


N = 7

a, b = sp.symbols('a b', real=True)
Re_b = sp.symbols(f'Re0:{N}', real=True)
Im_b = sp.symbols(f'Im0:{N}', real=True)
bias_c = [Re_b[j] + sp.I * Im_b[j] for j in range(N)]
all_params = list(Re_b) + list(Im_b)

OP_INFO = {
    'sub': (1, a - b),
    'add': (2, a + b),
    'mul': (3, a * b),
    'div': (4, a / b),
}


def basis_product(i: int, j: int):
    """Returns (k, factor) where k_i · k_j = factor · k_k."""
    if i == j:
        return ((i + 1) % N, sp.Integer(1))
    return ((2 * i + j) % N, sp.I)


def make_input(active_op_slot: int) -> list:
    inp = [sp.Integer(0)] * N
    inp[0] = a
    inp[active_op_slot] = sp.Integer(1)
    inp[5] = b
    inp[6] = sp.Integer(1)
    return inp


def square_complex(vec: list) -> list:
    hidden = [sp.Integer(0)] * N
    for i in range(N):
        for j in range(N):
            k, factor = basis_product(i, j)
            hidden[k] += factor * vec[i] * vec[j]
    return [sp.expand(h) for h in hidden]


def output_complex(hidden: list, k: int) -> sp.Expr:
    out = sp.Integer(0)
    for i in range(N):
        for j in range(N):
            k_target, factor = basis_product(i, j)
            if k_target == k:
                out += hidden[i] * bias_c[j] * factor
    return sp.expand(out)


def constraints_for_op(op_name: str, k: int) -> list:
    active_slot, target = OP_INFO[op_name]
    inp = make_input(active_slot)
    hidden = square_complex(inp)
    out = output_complex(hidden, k)
    re_out = sp.expand(sp.re(out))
    if op_name == 'div':
        diff = sp.expand(b * re_out - a)
    else:
        diff = sp.expand(re_out - target)
    if diff == 0:
        return []
    poly = sp.Poly(diff, a, b)
    return [sp.expand(coef) for _monom, coef in poly.terms()]


def check_feasibility(k: int, ops_subset: tuple):
    cons = []
    for op_name in ops_subset:
        cons.extend(constraints_for_op(op_name, k))
    if not cons:
        return True, 0, 0, 'no constraints (trivially feasible)'

    M = sp.Matrix([[sp.diff(c, p) for p in all_params] for c in cons])
    consts = sp.Matrix([
        -c.subs({p: 0 for p in all_params}) for c in cons
    ])
    rank_M = M.rank()
    rank_aug = M.row_join(consts).rank()
    feasible = (rank_M == rank_aug)
    nontrivial_dim = (2 * N) - rank_M  # dim of solution space if feasible
    note = f"solution-space dim = {nontrivial_dim}" if feasible else "inconsistent"
    return feasible, rank_M, len(cons), note


def main():
    out_lines = ["v4 complex-algebra feasibility analysis"]
    out_lines.append("=" * 72)
    out_lines.append(
        "Algebra: k_i·k_i = k_{(i+1) mod 7}, "
        "k_i·k_j = i·k_{(2i+j) mod 7} for i≠j"
    )
    out_lines.append("Bias: complex per slot, 14 real parameters")
    out_lines.append(
        "Constraint per op: Re(output[k]) = target_p   "
        "(div: b·Re(output[k]) = a)"
    )
    out_lines.append("")

    subsets = [
        ('add',),
        ('sub',),
        ('mul',),
        ('div',),
        ('add', 'sub'),
        ('add', 'sub', 'mul'),
        ('add', 'sub', 'mul', 'div'),
    ]

    for k in range(N):
        out_lines.append(f"--- Readout k_{k} ---")
        for subset in subsets:
            feasible, rank, n_cons, note = check_feasibility(k, subset)
            label = '{' + ','.join(subset) + '}'
            verdict = 'FEASIBLE' if feasible else 'infeasible'
            out_lines.append(
                f"  {label:<26} {verdict:<11} "
                f"(rank {rank}/{2*N}, {n_cons} constraints) — {note}"
            )
        out_lines.append("")

    text = "\n".join(out_lines)
    print(text)
    Path(__file__).parent.joinpath("v4_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
