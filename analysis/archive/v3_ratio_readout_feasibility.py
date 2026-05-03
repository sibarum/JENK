"""v3 ratio-readout feasibility — definitive analytical check.

Architecture under analysis:
  N = 5 slots, layout
    k_0 = in_0 (a)
    k_1 = op_num
    k_2 = op_den
    k_3 = in_1 (b)
    k_4 = out_0 = 1

  Op encoding (num, den):
    +  ↔ (1, 2)
    −  ↔ (−1, 2)
    *  ↔ (2, 1)
    /  ↔ (−2, 1)

  Algebra (same as v1/v2):
    k_i · k_i = k_{(i+1) mod N}
    k_i · k_j = k_{(2i+j) mod N},  i ≠ j

  Forward: hidden = input · input;  output = hidden · bias.

  Two readouts (k_N, k_D); prediction = output[k_N] / output[k_D].
  Convergence-loss-free target identity:
    output[k_N] = target_op · output[k_D],
    or (for division)  b·output[k_N] = a·output[k_D].

For every candidate (k_N, k_D) pair, this script:
  1. Computes hidden symbolically for each op.
  2. Forms the polynomial identity per op.
  3. Extracts coefficients of all monomials → linear system on the
     5-element bias vector.
  4. Solves the joint system across all 4 ops and reports whether a
     single bias makes every op's identity hold *exactly*.

Definitively answers: which (k_N, k_D) ratio readouts (if any) make the
v3 architecture jointly feasible across +, −, *, /.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import sympy as sp


N = 5
OUT_0_INIT = sp.Integer(1)

a, b = sp.symbols('a b')
biases = sp.symbols(f'e0:{N}')   # e0, e1, e2, e3, e4

OP_TARGETS = {
    '+': (sp.Integer(1),  sp.Integer(2),  a + b),
    '-': (sp.Integer(-1), sp.Integer(2),  a - b),
    '*': (sp.Integer(2),  sp.Integer(1),  a * b),
    '/': (sp.Integer(-2), sp.Integer(1),  a / b),
}


def basis_product_index(i: int, j: int) -> int:
    if i == j:
        return (i + 1) % N
    return (2 * i + j) % N


def make_input(op_num, op_den) -> list:
    """Layout: [in_0, op_num, op_den, in_1, out_0]."""
    return [a, op_num, op_den, b, OUT_0_INIT]


def square(vec: list) -> list:
    hidden = [sp.Integer(0)] * N
    for i in range(N):
        for j in range(N):
            hidden[basis_product_index(i, j)] += vec[i] * vec[j]
    return [sp.expand(h) for h in hidden]


def output_poly(hidden: list, k: int) -> sp.Expr:
    """output[k] as a polynomial in (a, b) with bias-coefficients."""
    expr = sp.Integer(0)
    for j in range(N):
        for i in range(N):
            if basis_product_index(i, j) == k:
                expr += hidden[i] * biases[j]
    return sp.expand(expr)


def precompute_outputs():
    """outputs[op][k] = output_poly for slot k under op (in (a, b, biases))."""
    table = {}
    for op_name, (op_n, op_d, _target) in OP_TARGETS.items():
        inp = make_input(op_n, op_d)
        hidden = square(inp)
        table[op_name] = [output_poly(hidden, k) for k in range(N)]
    return table


def constraint_polynomial(op_name: str, outputs_for_op: list, k_N: int, k_D: int) -> sp.Expr:
    """The polynomial that must be identically zero for the op to be exactly fit.

    Polynomial-target ops:  output[k_N] − target · output[k_D] ≡ 0
    Division:               b·output[k_N] − a·output[k_D]    ≡ 0
    """
    _, _, target = OP_TARGETS[op_name]
    if op_name == '/':
        constraint = b * outputs_for_op[k_N] - a * outputs_for_op[k_D]
    else:
        constraint = outputs_for_op[k_N] - target * outputs_for_op[k_D]
    return sp.expand(constraint)


def linear_constraints_from_polynomial(poly_expr: sp.Expr) -> list:
    """Extract one linear-in-biases constraint per (a, b) monomial coefficient."""
    if poly_expr == 0:
        return []
    poly = sp.Poly(poly_expr, a, b)
    constraints = []
    for _monom, coef in poly.terms():
        constraints.append(sp.expand(coef))
    return constraints


def check_pair(outputs_table: dict, k_N: int, k_D: int):
    """Return (feasible: bool, solution: dict or None, n_constraints, rank)."""
    all_constraints = []
    for op_name, outputs_for_op in outputs_table.items():
        constraint_poly = constraint_polynomial(op_name, outputs_for_op, k_N, k_D)
        all_constraints.extend(linear_constraints_from_polynomial(constraint_poly))

    if not all_constraints:
        return True, {}, 0, 0  # trivially feasible (no constraints)

    M = sp.Matrix([[sp.diff(c, e) for e in biases] for c in all_constraints])
    augmented = M.row_join(sp.Matrix([-c.subs({e: 0 for e in biases}) for c in all_constraints]))

    rank_M = M.rank()
    rank_aug = augmented.rank()
    feasible = (rank_M == rank_aug)

    solution = None
    if feasible:
        sol_list = sp.linsolve(all_constraints, biases)
        if sol_list:
            sol_tuple = list(sol_list)[0]
            solution = dict(zip(biases, sol_tuple))

    return feasible, solution, len(all_constraints), int(rank_M)


def verify_solution(solution: dict, outputs_table: dict, k_N: int, k_D: int) -> dict:
    """Plug the solution back in; report per-op residual polynomial."""
    residuals = {}
    for op_name, outputs_for_op in outputs_table.items():
        constraint_poly = constraint_polynomial(op_name, outputs_for_op, k_N, k_D)
        residual = sp.expand(constraint_poly.subs(solution))
        residuals[op_name] = residual
    return residuals


def main():
    out_lines = []
    out_lines.append("v3 ratio-readout joint feasibility — definitive analytical check")
    out_lines.append("=" * 72)
    out_lines.append("Architecture: N=5, op = (op_num, op_den), out_0 = 1.")
    out_lines.append("Op map:  + → (1, 2),  − → (−1, 2),  * → (2, 1),  / → (−2, 1).")
    out_lines.append("Constraint per op:  output[k_N] = target · output[k_D]")
    out_lines.append("                    (for /, b·output[k_N] = a·output[k_D])")
    out_lines.append("")

    outputs_table = precompute_outputs()

    out_lines.append("--- hidden = input · input, per op ---")
    for op_name in ['+', '-', '*', '/']:
        op_n, op_d, target = OP_TARGETS[op_name]
        inp = make_input(op_n, op_d)
        hidden = square(inp)
        out_lines.append(f"\n{op_name} (op_num={op_n}, op_den={op_d}, target={target}):")
        for k, h in enumerate(hidden):
            out_lines.append(f"  hidden[{k}] = {h}")

    out_lines.append("\n" + "=" * 72)
    out_lines.append("Joint feasibility scan over all (k_N, k_D) pairs (k_N ≠ k_D)")
    out_lines.append("=" * 72)

    feasible_pairs = []
    for k_N, k_D in product(range(N), range(N)):
        if k_N == k_D:
            continue
        feasible, solution, n_constraints, rank = check_pair(outputs_table, k_N, k_D)
        status = "FEASIBLE" if feasible else "infeasible"
        out_lines.append(
            f"  (k_N={k_N}, k_D={k_D}): {status}  "
            f"{n_constraints} constraints, rank {rank}"
        )
        if feasible and solution is not None:
            feasible_pairs.append((k_N, k_D, solution))
            out_lines.append(f"      solution: {solution}")
            residuals = verify_solution(solution, outputs_table, k_N, k_D)
            all_zero = all(r == 0 for r in residuals.values())
            out_lines.append(f"      residuals: {residuals}")
            out_lines.append(f"      all zero: {all_zero}")

    out_lines.append("\n" + "=" * 72)
    out_lines.append(f"VERDICT (4-op joint): {len(feasible_pairs)} feasible pair(s)")
    out_lines.append("=" * 72)
    nontrivial_pairs = [(k_N, k_D, s) for k_N, k_D, s in feasible_pairs
                        if any(v != 0 for v in s.values())]
    out_lines.append(f"  Non-trivial (bias ≠ 0) pairs: {len(nontrivial_pairs)}")
    if nontrivial_pairs:
        for k_N, k_D, sol in nontrivial_pairs:
            out_lines.append(f"    (k_N={k_N}, k_D={k_D}) bias = {sol}")
    else:
        out_lines.append("  All feasible pairs are trivial (bias=0 → both outputs = 0 → undefined N/D).")
        out_lines.append("  (B) FAILS for the full 4-op set under this encoding.")

    # Subset diagnosis: which op subsets ARE feasible non-trivially?
    out_lines.append("\n" + "=" * 72)
    out_lines.append("Subset diagnosis: which op-subsets allow a non-trivial bias?")
    out_lines.append("=" * 72)

    from itertools import combinations
    op_subsets = []
    for r in range(1, 5):
        for combo in combinations(['+', '-', '*', '/'], r):
            op_subsets.append(combo)

    for subset in op_subsets:
        outputs_subset = {op: outputs_table[op] for op in subset}
        # For each (k_N, k_D), check rank under just this subset.
        nontrivial_count = 0
        example = None
        for k_N, k_D in product(range(N), range(N)):
            if k_N == k_D:
                continue
            all_constraints = []
            for op_name, outputs_for_op in outputs_subset.items():
                cp = constraint_polynomial(op_name, outputs_for_op, k_N, k_D)
                all_constraints.extend(linear_constraints_from_polynomial(cp))
            if not all_constraints:
                continue
            M = sp.Matrix([[sp.diff(c, e) for e in biases] for c in all_constraints])
            if M.rank() < N:
                # Non-trivial nullspace exists — solve to confirm.
                sol_set = sp.linsolve(all_constraints, biases)
                if sol_set:
                    sol = list(sol_set)[0]
                    if any(s != 0 for s in sol):
                        nontrivial_count += 1
                        if example is None:
                            example = (k_N, k_D, sol)
        if nontrivial_count > 0 and example is not None:
            k_N, k_D, sol = example
            out_lines.append(
                f"  {{{','.join(subset)}}}: {nontrivial_count} non-trivial pair(s); "
                f"example (k_N={k_N}, k_D={k_D}) bias={sol}"
            )
        else:
            out_lines.append(f"  {{{','.join(subset)}}}: NO non-trivial pair")

    text = "\n".join(out_lines)
    print(text)
    Path(__file__).parent.joinpath("v3_ratio_readout_feasibility_results.txt").write_text(
        text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
