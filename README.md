Jεηk
====

A research-prototype implementation of Traction theory for neural-net
primitives. The goal is a single learnable parameter `s` that lets one
layer operate in any of four computation modes (hyperbolic, parabolic,
elliptic, projective) without branching — and a substrate where the
framework's information-conservation discipline is operational, not just
notational.

Status: experimental. No PyTorch / GPU layer yet; everything below is
sympy-based reference and diagnostic code. 219 tests passing. See
[Findings](#findings) for what's been characterized so far.


The four modes
--------------

|   | Mode       | Traction | δ |
|---|------------|----------|---|
| j | Hyperbolic | 0^(0/2)  | +1|
| ε | Parabolic  | 0^(1/2)  |  0|
| η | Elliptic   | 0^(ω/2)  | −1|
| k | Projective | 0^(-1/2) | ω |

A single parameter `s` selects the mode. The framework's claim is that
all four modes share the *same math* — the only thing that changes is
the value of δ. That makes one piece of hardware enough to compute any
mode, and lets the mode itself be learnable per input.


Traction in 60 seconds
----------------------

Traction is the operational layer of Constructive Operational Type
Theory (COTT). The core principle, paraphrased from COTT:
**zero erases only when the result is invariant under the surrounding
operation.** Annihilation is *deferred* rather than eager. ω is the
reciprocal of zero — not an infinity, more like the closing-the-loop
counterpart that makes division total.

Headline identities:

    0·ω = 1                     ω = -0
    1/0 = ω                     0^0 = 1
    0^1 = 0                     0^(-1) = ω
    0^ω = -1                    ω^ω = -1
    0^(0^x) = x                 0^(ω^x) = -x

Subtraction `a − a` does NOT collapse to numeric zero in the framework
— it produces ∅ (Null), the erasure element. The distinction matters:
a numeric zero may legitimately arise from `1 − 1`; a Null is a record
that information was discarded.

For the source theory, see https://sibarum.github.io/cott/intro/ and
https://sibarum.github.io/cott/theory/reference/.


What's in this repo
-------------------

```
jenk/
  graded.py          (a, b) graded carrier — represents a·0^b. The older
                     free-symbol model for ω. Used by closure_explorer.py.
                     Eventual cleanup: migrate onto the atoms in traction.py.
  traction.py        Traction algebra primitives as sympy Expr subclasses:
                     Zero, Omega, Null. Framework identities fire
                     automatically — z*w → 1, z**w → -1, etc., with no
                     manual reduction call.
  ring.py            Real-s Chebyshev ring  Q[s][g]/(g²−sg+1).  Float arith.
  lifted_ring.py     Sympy-valued variant — s, a, b can be symbolic
                     (may contain ω). Exact arithmetic.
  symbolic_neuron.py Matrix-recipe neuron: c, d, s as sympy expressions;
                     forward is the regular representation of c + d·g.
                     Uses the older free-symbol ω, does NOT auto-fire
                     framework identities. Kept as a baseline for
                     comparison.
  graded_neuron.py   Traction-native neuron. Same forward formula as
                     symbolic_neuron, but parameters can carry the
                     z/w/null atoms; framework identities fire
                     automatically as the forward pass evaluates.
                     Each step recordable into a Trace.
  trace.py           Diagnostic recorder. Captures raw and
                     traction_simplify-reduced forms at every step,
                     flags null/zero/omega appearances, dumps to
                     JSON in tmp/diagnostics/.

experiments/         Research scripts (Stage-3 path-B benchmarks,
                     Stage-4a diagnostic demo).

tests/               Pytest suite.  219 tests as of 2026-04-30:
                     62 ring + 71 lifted_ring + 23 symbolic_neuron
                   + 45 traction + 18 graded_neuron.

closure_explorer.py  Symbolic explorer using jenk.graded — probes
                     cardinals, closure, products.
```


Quickstart
----------

Requires the `traction` conda env (Python 3.12, sympy, gmpy2, numpy):

    # Run the test suite
    PYTHONIOENCODING=utf-8 conda run -n traction python -m pytest tests/ -q

    # Run the diagnostic demo and inspect a trace
    PYTHONIOENCODING=utf-8 PYTHONPATH=. conda run -n traction \
        python experiments/graded_neuron_demo.py

The demo writes `tmp/diagnostics/demo_single.json` and `demo_chain.json`
— each is a step-by-step record of the forward pass, with raw and
simplified expressions side by side.

`PYTHONIOENCODING=utf-8` is needed on Windows because conda's stdout
wrapper uses cp1252 by default and chokes on the framework's unicode
symbols (ω, ∅, etc.).


The two layer designs
---------------------

Both `SymbolicNeuron` and `GradedNeuron` implement the same forward
formula — the regular representation of the ring element `c + d·g`
acting on a 2D input:

    out_a = c·x_a − d·x_b              + bias_a
    out_b = d·x_a + (c + d·s)·x_b      + bias_b

The difference is what kind of arithmetic happens underneath:

**SymbolicNeuron (matrix-recipe).**  c, d, s are sympy expressions;
ω is a free sympy symbol with a positivity hint. The framework
identities (z·w = 1, z^w = −1, etc.) do not fire automatically. To
apply them you must call into the older `Graded.reduced()` machinery
manually. This matches what most "graded carrier" implementations do
elsewhere and is useful as a baseline.

**GradedNeuron (Traction-native).**  c, d, s, biases can carry the
`Zero`, `Omega`, `Null` sympy atoms from `jenk/traction.py`. Framework
identities are built into `_eval_power` and `__mul__` on those atoms,
so they fire automatically as the expression is constructed. There
is no manual reduction step. Every intermediate is recordable into a
`Trace`, which dumps to JSON for inspection.

Concrete demonstration: at `s = ω` with `x_b = z` (the framework zero),
`m6 = (c + d·s)·x_b = w·z` reduces to `1` *inside the forward pass*.
The matrix-recipe variant cannot reach this — `w·z` is just two free
symbols multiplied, and stays as `omega·0 = 0`.


Findings
--------

The early experiments (in `experiments/path_b_*.py`) put the
matrix-recipe `SymbolicNeuron` head-to-head against `FreeLinear`,
`Complex1L`, and `MLP-tanh` on five 2×2 targets including
`M_proj = [[0, 0], [0, 1]]`. Headline results:

- The 1-param SymbolicNeuron beats free-2×2 by up to 1000× under noise
  + limited data — strong inductive bias when the target is in the
  ring image.
- A sinh re-parametrization of `s` fails (gradient explosion).
- Generator-only chains are locked to `(0, 0) = −1` (in the graded
  carrier) and only homogeneous chains recover the omega-limit
  projection cleanly.
- **`M_proj` is provably outside the chain image.** Each layer is
  `c·I + d·M(s)` with det = `c² + cds + d²` (the ring norm), so the
  chain composes det-preserving maps up to scaling and lands in
  SL(2)-flavored territory. The empirical constraint
  `d1·d2·(s2 − s1) = 0` is necessary but not sufficient — the rank-1
  target is genuinely unreachable.
- In framework-native terms, `M_proj = M(ω)·(1/ω) = M(ω)·0` using
  `1/ω = 0` and `ω·0 = 1`. As an operator it is left-multiplication by
  `0·g` at `s = ω` — a Traction-native action whose presence in the
  layer requires *graded coefficients*, not just graded `s`.
- The matrix-recipe layer borrows structure: at `s1 ≠ s2` the chain
  buys expressivity by composing across distinct rings, which is not
  a Traction operation. That extra capacity is in the matrix algebra,
  not in the framework. With 6 params the chain has only ~4
  framework-faithful degrees of freedom.

These findings motivated `jenk/graded_neuron.py`: a layer where the
coefficients themselves can carry framework atoms, so the `0·ω = 1`
identity fires inside the forward pass. The diagnostic JSON makes
that firing visible step by step.


Layout
------

The recommended path through the code if you want to understand it:

1. `jenk/traction.py` — the algebra atoms.
2. `jenk/graded_neuron.py` — what a single forward pass looks like.
3. `jenk/trace.py` — how diagnostics are captured.
4. `experiments/graded_neuron_demo.py` — a runnable example.
5. `experiments/path_b_four_way.py` — the matrix-recipe benchmark and
   the M_proj reachability question.

`SymbolicNeuron` (`jenk/symbolic_neuron.py`) and the older `Graded`
carrier (`jenk/graded.py`) are kept for comparison and historical
reference; new work should build on `traction.py` and `graded_neuron.py`.


What's deliberately out of scope (for now)
------------------------------------------

- PyTorch / GPU layer.  Sympy reference first; performance later.
- The `0 + ω = 0` and similar cross-grade addition rules from the
  README of the framework. Treated as not-yet-trusted; cross-grade
  additions are kept as deferred sympy expressions in the diagnostic
  layer rather than collapsed by an arbitrary rule.
- The Z-action (`GradedElement` / `Z_n(x)`) defined in
  `jenk/traction.py`. Present in the file, unused everywhere else.
- A visualizer for trace JSON. Probably next, once we've stared at
  enough traces to know what's worth showing.
