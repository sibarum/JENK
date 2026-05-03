"""v5: reciprocal-input single-neuron feasibility — does in_1 = 1/b unlock division?

Architecture:
  N = 7 (v2 layout, one-hot op, out_0 = 1)
  Slot layout:  k_0 = in_0(a)
                k_1 = op_-
                k_2 = op_+
                k_3 = op_*
                k_4 = op_/
                k_5 = in_1
                k_6 = out_0 (= 1)

Difference from v2: in_1 holds 1/b instead of b.  Hidden = input · input
then contains rational expressions in (a, b): a/b at slot 3, 1/b² at slot 6,
etc.  Output is a polynomial-in-bias whose coefficients are rational
in (a, b).

For each (op, readout_slot) pair, this script:
  1. Builds the symbolic input with active op slot one-hot at 1, in_1 = 1/b.
  2. Computes hidden = input · input.
  3. Computes output[k] symbolically (linear in bias).
  4. Forms the constraint  output[k] - target == 0  as a rational expression
     in (a, b) with bias as parameters; clears the denominator; takes the
     numerator as a polynomial-in-(a, b)-with-bias-coefficients.
  5. Each monomial coefficient = one linear constraint on the bias vector.
  6. Solves for bias and reports feasibility + non-triviality.

Hypothesis: division becomes feasible at some readout slot under reciprocal
input encoding because the architecture's hidden layer now natively contains
a/b — degree-2 in (a, 1/b) is a rational function in (a, b).

Companion to v2's identity-input neuron: pair = (identity, reciprocal),
together covering {+, −, *} via the identity neuron and {/} via the
reciprocal neuron.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp


N = 7

a, b = sp.symbols('a b')
biases = sp.symbols(f'e0:{N}')

SLOT_IN_0 = 0
SLOT_OP_SUB = 1
SLOT_OP_ADD = 2
SLOT_OP_MUL = 3
SLOT_OP_DIV = 4
SLOT_IN_1 = 5
SLOT_OUT_0 = 6

OP_INFO = {
    '-': (SLOT_OP_SUB, a - b),
    '+': (SLOT_OP_ADD, a + b),
    '*': (SLOT_OP_MUL, a * b),
    '/': (SLOT_OP_DIV, a / b),
}


def basis_product_index(i: int, j: int) -> int:
    if i == j:
        return (i + 1) % N
    return (2 * i + j) % N


def make_input_reciprocal(active_op_slot: int) -> list:
    """Input vector with in_1 = 1/b (the reciprocal-encoding twist)."""
    inp = [sp.Integer(0)] * N
    inp[SLOT_IN_0] = a
    inp[active_op_slot] = sp.Integer(1)
    inp[SLOT_IN_1] = sp.Integer(1) / b
    inp[SLOT_OUT_0] = sp.Integer(1)
    return inp


def square(vec: list) -> list:
    hidden = [sp.Integer(0)] * N
    for i in range(N):
        for j in range(N):
            k = basis_product_index(i, j)
            hidden[k] += vec[i] * vec[j]
    return [sp.together(h) for h in hidden]


def output_expr(hidden: list, k: int) -> sp.Expr:
    expr = sp.Integer(0)
    for j in range(N):
        for i in range(N):
            if basis_product_index(i, j) == k:
                expr += hidden[i] * biases[j]
    return sp.together(expr)


def feasibility(op_name: str, k: int):
    """Check whether a bias vector exists with output[k] == target.

    Returns dict with: feasible, has_nontrivial, solution, residual, n_constraints, rank.
    """
    active_slot, target = OP_INFO[op_name]
    inp = make_input_reciprocal(active_slot)
    hidden = square(inp)
    out = output_expr(hidden, k)

    diff = sp.together(out - target)
    num, _den = sp.fraction(diff)
    num = sp.expand(num)

    if num == 0:
        return {
            'feasible': True,
            'has_nontrivial': False,
            'solution': dict(zip(biases, [sp.Integer(0)] * N)),
            'residual': sp.Integer(0),
            'n_constraints': 0,
            'rank': 0,
            'note': 'identically zero before any constraint',
        }

    poly = sp.Poly(num, a, b)
    constraints = [sp.expand(coef) for _monom, coef in poly.terms()]

    M = sp.Matrix([[sp.diff(c, e) for e in biases] for c in constraints])
    consts = sp.Matrix([-c.subs({e: 0 for e in biases}) for c in constraints])
    aug = M.row_join(consts)
    rank_M = M.rank()
    rank_aug = aug.rank()
    feasible = (rank_M == rank_aug)

    if not feasible:
        return {
            'feasible': False,
            'has_nontrivial': False,
            'solution': None,
            'residual': None,
            'n_constraints': len(constraints),
            'rank': rank_M,
            'note': f'inconsistent (rank {rank_M} of M vs {rank_aug} of [M|c])',
        }

    sol_set = sp.linsolve(constraints, biases)
    if not sol_set:
        return {
            'feasible': True,
            'has_nontrivial': False,
            'solution': None,
            'residual': None,
            'n_constraints': len(constraints),
            'rank': rank_M,
            'note': 'feasible but linsolve returned empty',
        }

    sol_tuple = list(sol_set)[0]
    solution = dict(zip(biases, sol_tuple))
    has_nontrivial = any(v != 0 for v in solution.values())

    out_sub = sp.simplify(out.subs(solution) - target)

    return {
        'feasible': True,
        'has_nontrivial': has_nontrivial,
        'solution': solution,
        'residual': out_sub,
        'n_constraints': len(constraints),
        'rank': rank_M,
        'note': 'ok',
    }


def main() -> None:
    out_lines = []
    out_lines.append("v5 reciprocal-input feasibility — single neuron, in_1 = 1/b")
    out_lines.append("=" * 72)
    out_lines.append(
        "Architecture: N=7, v2 slot layout, but in_1 holds 1/b (not b)."
    )
    out_lines.append(
        "Slots: k_0=in_0(a) k_1=op_- k_2=op_+ k_3=op_* k_4=op_/ "
        "k_5=in_1(1/b) k_6=out_0(=1)"
    )
    out_lines.append("")
    out_lines.append("--- Hidden = input · input, per op ---")
    for op_name in ['+', '-', '*', '/']:
        active_slot, _target = OP_INFO[op_name]
        inp = make_input_reciprocal(active_slot)
        hidden = square(inp)
        out_lines.append(f"\n{op_name} (op slot k_{active_slot} active):")
        for k, h in enumerate(hidden):
            out_lines.append(f"  hidden[{k}] = {h}")

    out_lines.append("\n" + "=" * 72)
    out_lines.append("Feasibility per (op, readout_slot) pair")
    out_lines.append("=" * 72)

    summary_table = []  # rows for the final summary
    for op_name in ['+', '-', '*', '/']:
        out_lines.append(f"\n--- Target op: {op_name},  target = {OP_INFO[op_name][1]} ---")
        for k in range(N):
            res = feasibility(op_name, k)
            if not res['feasible']:
                out_lines.append(
                    f"  output[{k}]: INFEASIBLE  ({res['n_constraints']} constraints, "
                    f"rank {res['rank']}) — {res['note']}"
                )
                summary_table.append((op_name, k, 'infeasible'))
            elif not res['has_nontrivial']:
                out_lines.append(
                    f"  output[{k}]: feasible but TRIVIAL only (bias=0)"
                )
                summary_table.append((op_name, k, 'trivial-only'))
            else:
                out_lines.append(
                    f"  output[{k}]: NON-TRIVIAL FEASIBLE  "
                    f"({res['n_constraints']} constraints, rank {res['rank']})"
                )
                out_lines.append(f"    bias: {res['solution']}")
                out_lines.append(f"    residual after substitution: {res['residual']}")
                summary_table.append((op_name, k, 'non-trivial'))

    out_lines.append("\n" + "=" * 72)
    out_lines.append("Summary table: which (op, readout_slot) pairs admit a non-trivial bias?")
    out_lines.append("=" * 72)
    out_lines.append("")
    header = "          " + "  ".join(f"k_{k}" for k in range(N))
    out_lines.append(header)
    for op_name in ['+', '-', '*', '/']:
        row_cells = []
        for k in range(N):
            verdict = next(v for o, kk, v in summary_table if o == op_name and kk == k)
            cell = {'non-trivial': ' ✓ ', 'trivial-only': ' . ', 'infeasible': ' x '}[verdict]
            row_cells.append(cell)
        out_lines.append(f"  {op_name}     " + "  ".join(row_cells))
    out_lines.append("")
    out_lines.append("Legend:  ✓ = non-trivial bias exists (target is fittable exactly)")
    out_lines.append("         . = only trivial bias=0 satisfies the constraints")
    out_lines.append("         x = infeasible (no bias makes it hold)")

    text = "\n".join(out_lines)
    print(text)
    Path(__file__).parent.joinpath("v5_reciprocal_input_results.txt").write_text(
        text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
