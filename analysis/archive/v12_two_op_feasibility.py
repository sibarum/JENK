"""v12: analytical feasibility check for the v10/v11 architecture.

Question: at the v2 N=7 layout with op_+ at slot 2 and op_* at slot 3
(slots 1 and 4 zero, in_1 at slot 5, out_0=1 at slot 6), does there exist
a single bias and a single readout slot k such that:

  output[k] under op_+ encoding ≡ a + b'   (linear in (a, b'))
  output[k] under op_* encoding ≡ a · b'   (bilinear in (a, b'))

simultaneously, as polynomial identities?

If feasible at some k: v11's empirical failure was an optimization issue.
If infeasible at every k: the architecture is structurally limited for
the joint task at a single readout. Next step would be multi-readout
(different k per op).

Also checks the multi-readout variant: separate (readout_+, readout_*)
pair, with the same bias serving both. This is the cheapest "channel
separation at network level" that doesn't require per-op bias.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import sympy as sp


N = 7
a, bp = sp.symbols("a bp")  # bp = b'  (the second-input slot value, after preprocessing)
biases = sp.symbols(f"e0:{N}")

# Slot indices
SLOT_IN_0 = 0
SLOT_OP_ADD = 2
SLOT_OP_MUL = 3
SLOT_IN_1 = 5
SLOT_OUT_0 = 6


def basis_product_index(i: int, j: int) -> int:
    if i == j:
        return (i + 1) % N
    return (2 * i + j) % N


def make_input(active_op_slot: int) -> list:
    inp = [sp.Integer(0)] * N
    inp[SLOT_IN_0] = a
    inp[active_op_slot] = sp.Integer(1)
    inp[SLOT_IN_1] = bp
    inp[SLOT_OUT_0] = sp.Integer(1)
    return inp


def square(vec: list) -> list:
    hidden = [sp.Integer(0)] * N
    for i in range(N):
        for j in range(N):
            k = basis_product_index(i, j)
            hidden[k] += vec[i] * vec[j]
    return [sp.expand(h) for h in hidden]


def output_expr(hidden: list, k: int) -> sp.Expr:
    expr = sp.Integer(0)
    for j in range(N):
        for i in range(N):
            if basis_product_index(i, j) == k:
                expr += hidden[i] * biases[j]
    return sp.expand(expr)


def constraints_from_identity(expr: sp.Expr) -> list:
    """Given expr that should vanish identically in (a, bp), extract one
    linear-in-bias constraint per (a, bp) monomial coefficient."""
    if expr == 0:
        return []
    poly = sp.Poly(expr, a, bp)
    return [sp.expand(coef) for _monom, coef in poly.terms()]


def check_pair(k_plus: int, k_mul: int) -> dict:
    """Joint feasibility: ∃ bias such that output[k_plus]_op_+ = a+bp
    AND output[k_mul]_op_* = a·bp."""
    inp_plus = make_input(SLOT_OP_ADD)
    inp_mul = make_input(SLOT_OP_MUL)
    hidden_plus = square(inp_plus)
    hidden_mul = square(inp_mul)
    out_plus = output_expr(hidden_plus, k_plus)
    out_mul = output_expr(hidden_mul, k_mul)

    target_plus = a + bp
    target_mul = a * bp

    diff_plus = sp.expand(out_plus - target_plus)
    diff_mul = sp.expand(out_mul - target_mul)

    constraints = constraints_from_identity(diff_plus) + constraints_from_identity(diff_mul)
    if not constraints:
        return {'feasible': True, 'has_nontrivial': True, 'rank': 0,
                'n_constraints': 0, 'solution': dict(zip(biases, [sp.Integer(0)] * N)),
                'note': 'identically zero before any constraint'}

    M = sp.Matrix([[sp.diff(c, e) for e in biases] for c in constraints])
    consts = sp.Matrix([-c.subs({e: 0 for e in biases}) for c in constraints])
    rank_M = M.rank()
    rank_aug = M.row_join(consts).rank()
    feasible = (rank_M == rank_aug)

    if not feasible:
        return {'feasible': False, 'has_nontrivial': False, 'rank': rank_M,
                'n_constraints': len(constraints),
                'note': f'inconsistent (rank M={rank_M} vs [M|c]={rank_aug})'}

    sol_set = sp.linsolve(constraints, biases)
    if not sol_set:
        return {'feasible': True, 'has_nontrivial': False, 'rank': rank_M,
                'n_constraints': len(constraints), 'note': 'feasible but linsolve empty'}

    sol_tuple = list(sol_set)[0]
    solution = dict(zip(biases, sol_tuple))
    has_nontrivial = any(v != 0 for v in solution.values())

    # Verify
    res_plus = sp.simplify(out_plus.subs(solution) - target_plus)
    res_mul = sp.simplify(out_mul.subs(solution) - target_mul)

    return {
        'feasible': True, 'has_nontrivial': has_nontrivial,
        'rank': rank_M, 'n_constraints': len(constraints),
        'solution': solution,
        'residual_plus': res_plus, 'residual_mul': res_mul,
        'note': 'ok',
    }


def main():
    out_lines = []
    out_lines.append("v12 two-op feasibility analysis")
    out_lines.append("=" * 78)
    out_lines.append("Question: at v2 N=7 layout with op_+ at slot 2, op_* at slot 3,")
    out_lines.append("does there exist a (bias, k_plus, k_mul) such that:")
    out_lines.append("  output[k_plus] under op_+ encoding ≡ a + b'")
    out_lines.append("  output[k_mul] under op_* encoding ≡ a · b'")
    out_lines.append("simultaneously?")
    out_lines.append("")
    out_lines.append("Slots: k_0=in_0(a) k_1=0 k_2=op_+ k_3=op_* k_4=0 k_5=in_1(b') k_6=out_0(=1)")
    out_lines.append("")

    # Hidden vectors per op
    out_lines.append("=== Hidden = input · input, per active op ===")
    for label, slot in [("op_+", SLOT_OP_ADD), ("op_*", SLOT_OP_MUL)]:
        hidden = square(make_input(slot))
        out_lines.append(f"\n{label} (slot {slot} active):")
        for k, h in enumerate(hidden):
            out_lines.append(f"  hidden[{k}] = {h}")

    out_lines.append("\n" + "=" * 78)
    out_lines.append("Part 1: SAME readout for both ops (the v10/v11 setup)")
    out_lines.append("=" * 78)
    same_readout_results = []
    for k in range(N):
        res = check_pair(k, k)
        if not res['feasible']:
            verdict = 'INFEASIBLE'
            tag = res['note']
        elif not res['has_nontrivial']:
            verdict = 'feasible (TRIVIAL only)'
            tag = ''
        else:
            verdict = 'NON-TRIVIAL FEASIBLE'
            tag = f"residuals: +={res['residual_plus']}, *={res['residual_mul']}"
        out_lines.append(
            f"  readout=k_{k}: {verdict}  (rank {res.get('rank', 'NA')}, "
            f"{res.get('n_constraints', 'NA')} constraints) {('-- ' + tag) if tag else ''}"
        )
        same_readout_results.append((k, res))

    same_feasible = [k for k, r in same_readout_results if r['feasible'] and r['has_nontrivial']]
    out_lines.append("")
    if same_feasible:
        out_lines.append(f"  Single-readout joint feasibility: works at slots {same_feasible}")
    else:
        out_lines.append("  Single-readout joint feasibility: NONE work.")
        out_lines.append("  -> v10/v11 empirical failure is structural, not optimization.")

    out_lines.append("\n" + "=" * 78)
    out_lines.append("Part 2: DIFFERENT readouts per op (multi-readout with shared bias)")
    out_lines.append("=" * 78)
    multi_results = []
    for k_plus, k_mul in product(range(N), range(N)):
        if k_plus == k_mul:
            continue
        res = check_pair(k_plus, k_mul)
        multi_results.append(((k_plus, k_mul), res))

    feasible_pairs = [(p, r) for p, r in multi_results if r['feasible'] and r['has_nontrivial']]
    out_lines.append(f"\nTotal pairs checked: {len(multi_results)}")
    out_lines.append(f"Pairs with non-trivial joint solution: {len(feasible_pairs)}")
    if feasible_pairs:
        out_lines.append("\nFirst few feasible (readout_+, readout_*) pairs:")
        for (k_plus, k_mul), r in feasible_pairs[:8]:
            out_lines.append(
                f"  (k_+={k_plus}, k_*={k_mul}): rank {r['rank']}, "
                f"residuals + = {r['residual_plus']}, * = {r['residual_mul']}"
            )
            out_lines.append(f"    bias = {r['solution']}")
    else:
        out_lines.append("\nNO multi-readout pair admits a joint solution.")
        out_lines.append("-> Two op categories are not jointly feasible with shared bias")
        out_lines.append("   under this routing rule. Next step: per-category bias (= 2 neurons).")

    text = "\n".join(out_lines)
    Path(__file__).parent.joinpath("v12_two_op_feasibility_results.txt").write_text(
        text, encoding="utf-8"
    )
    print("Results written.")


if __name__ == "__main__":
    main()
