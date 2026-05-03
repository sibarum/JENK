"""v2 SWAPPED path-count derivation — div↔in_0 swap.

Hypothesis: the cross-op dead-slot table becomes a clean permutation on N=7
when the op band {div, sub, add, mul} is registered to slots {0, 1, 2, 3} —
the cardinal positions of the Möbius 4-cycle.

Original v2 layout:
  k_0 = in_0(a)   k_1 = op_-   k_2 = op_+   k_3 = op_*
  k_4 = op_/      k_5 = in_1(b)   k_6 = out_0(=1)

Swapped layout (this analysis):
  k_0 = op_/      k_1 = op_-   k_2 = op_+   k_3 = op_*
  k_4 = in_0(a)   k_5 = in_1(b)   k_6 = out_0(=1)

This rotates the op band into correct cardinal registration with no other
changes. inputs stay in their own band (slots 4-6) and never compete for
cardinal slots.
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp


N = 7

a, b = sp.symbols('a b')

# Swapped slot semantics
SLOT_OP_DIV = 0
SLOT_OP_SUB = 1
SLOT_OP_ADD = 2
SLOT_OP_MUL = 3
SLOT_IN_0 = 4
SLOT_IN_1 = 5
SLOT_OUT_0 = 6

OP_INFO = {
    'sub': (SLOT_OP_SUB, a - b),
    'add': (SLOT_OP_ADD, a + b),
    'mul': (SLOT_OP_MUL, a * b),
    'div': (SLOT_OP_DIV, a / b),
}


def basis_product_index(i: int, j: int) -> int:
    if i == j:
        return (i + 1) % N
    return (2 * i + j) % N


def make_input(active_op_slot: int) -> list:
    """Layout: op_/, op_-, op_+, op_*, in_0(a), in_1(b), out_0(=1)."""
    inp = [sp.Integer(0)] * N
    inp[SLOT_IN_0] = a
    inp[active_op_slot] = sp.Integer(1)
    inp[SLOT_IN_1] = b
    inp[SLOT_OUT_0] = sp.Integer(1)
    return inp


def square(vec: list) -> list:
    hidden = [sp.Integer(0)] * N
    for i in range(N):
        for j in range(N):
            hidden[basis_product_index(i, j)] += vec[i] * vec[j]
    return [sp.expand(h) for h in hidden]


def path_count(hidden: list, k: int) -> list:
    pc = [sp.Integer(0)] * N
    for j in range(N):
        for i in range(N):
            if basis_product_index(i, j) == k:
                pc[j] += hidden[i]
    return [sp.expand(c) for c in pc]


def is_zero(expr) -> bool:
    return sp.simplify(expr) == 0


def main():
    out_lines = []
    out_lines.append("v2 SWAPPED path-count tables (div at k_0, in_0 at k_4)")
    out_lines.append("=" * 72)
    out_lines.append("Slot layout: k_0=op_/  k_1=op_-  k_2=op_+  k_3=op_*  "
                     "k_4=in_0(a)  k_5=in_1(b)  k_6=out_0(=1)")
    out_lines.append("")

    all_results = {}
    for op_name in ['div', 'sub', 'add', 'mul']:
        active_slot, target = OP_INFO[op_name]
        inp = make_input(active_slot)
        hidden = square(inp)
        pcs = [path_count(hidden, k) for k in range(N)]
        all_results[op_name] = {'hidden': hidden, 'pcs': pcs}

    out_lines.append("=" * 72)
    out_lines.append("Cross-op structural deadness: bias slot dead under EVERY op")
    out_lines.append("=" * 72)

    dead_map = {}  # output_slot -> dead bias slot
    for k in range(N):
        dead_everywhere = set(range(N))
        for op_name in ['div', 'sub', 'add', 'mul']:
            pc = all_results[op_name]['pcs'][k]
            live_in_op = {j for j, c in enumerate(pc) if not is_zero(c)}
            dead_everywhere -= live_in_op
        dead_list = sorted(dead_everywhere)
        out_lines.append(f"  output[{k}]: dead-in-all-ops bias slots = {dead_list}")
        if len(dead_list) == 1:
            dead_map[k] = dead_list[0]
        elif len(dead_list) == 0:
            dead_map[k] = None
        else:
            dead_map[k] = dead_list  # multiple

    out_lines.append("")
    out_lines.append("=" * 72)
    out_lines.append("Permutation check")
    out_lines.append("=" * 72)

    # Check: each output kills exactly one bias, each bias killed by exactly one output
    output_to_bias = {}
    bias_killed_by = {}
    multi_or_zero = []
    for k, dead in dead_map.items():
        if isinstance(dead, list):
            multi_or_zero.append(f"output[{k}] kills MULTIPLE biases: {dead}")
        elif dead is None:
            multi_or_zero.append(f"output[{k}] kills NOTHING")
        else:
            output_to_bias[k] = dead
            bias_killed_by.setdefault(dead, []).append(k)

    out_lines.append("Output → killed bias map:")
    for k in range(N):
        out_lines.append(f"  output[{k}] → {dead_map[k]}")

    out_lines.append("")
    out_lines.append("Bias → killed-by-output map:")
    for j in range(N):
        killers = bias_killed_by.get(j, [])
        out_lines.append(f"  bias[{j}] killed by outputs {killers if killers else '— (alive everywhere)'}")

    out_lines.append("")
    if multi_or_zero:
        out_lines.append("NOT a clean permutation:")
        for issue in multi_or_zero:
            out_lines.append(f"  {issue}")
    else:
        # Each output kills exactly one bias; check uniqueness
        biases_killed = list(output_to_bias.values())
        if len(set(biases_killed)) == len(biases_killed) and len(biases_killed) == N:
            out_lines.append("✓ CLEAN PERMUTATION: each output kills a unique bias, all biases killed.")
        else:
            multi_killed = {b: outs for b, outs in bias_killed_by.items() if len(outs) > 1}
            never_killed = [j for j in range(N) if j not in bias_killed_by]
            out_lines.append("Each output kills a single bias, but not a permutation:")
            if multi_killed:
                out_lines.append(f"  bias slots killed by multiple outputs: {multi_killed}")
            if never_killed:
                out_lines.append(f"  bias slots killed by no output: {never_killed}")

    text = "\n".join(out_lines)
    print(text)
    Path(__file__).parent.joinpath("v2_swapped_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
