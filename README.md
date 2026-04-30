Jεηk
====

A research-prototype implementation of Traction theory for neural-net
primitives. The goal is a single learnable parameter `s` that lets one
layer operate in any of four computation modes (hyperbolic, parabolic,
elliptic, projective) without branching — and a substrate where the
framework's information-conservation discipline is operational, not just
notational.

Status: experimental. No PyTorch / GPU layer — and that is now a
deliberate design choice, not a deferral. The reference implementation
stays sympy-throughout because past attempts to embed Traction
semantics in a numerical layer kept losing framework atoms in
translation. 256 tests passing. See [Findings](#findings) for what's
been characterized so far.


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
  learnable_neuron.py Symbolic-first trainable layer.
                     LearnableParameter wraps a sympy Symbol with a
                     mutable numeric value. LearnableNeuron registers
                     which fields are learnable; forward stays
                     symbolic. SymbolicOptimizer uses sp.diff +
                     sp.lambdify for fast repeated gradient
                     evaluation while the underlying expressions
                     never get coerced to floats.
  free_linear.py     Unconstrained 2×2 baseline. Four learnable
                     entries m11, m12, m21, m22 with no structural
                     bias. Same protocol as LearnableNeuron, so
                     SymbolicOptimizer accepts it via duck typing.
                     Used in path_b_inductive_bias.py to measure
                     the framework's inductive-bias advantage.
  chain_learnable.py Composes N learnable layers into one trainable
                     unit. Combined parameter list is the
                     concatenation of each layer's; same protocol
                     as a single layer. Used in
                     path_b_chain_image.py for the depth sweep.

experiments/         Research scripts (Stage-3 path-B benchmarks,
                     Stage-4a diagnostic demo).

tests/               Pytest suite.  256 tests as of 2026-04-30:
                     62 ring + 71 lifted_ring + 23 symbolic_neuron
                   + 45 traction + 18 graded_neuron + 23 learnable_neuron
                   + 8 free_linear + 6 chain_learnable.

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
  + limited data on the Stage-3 setup. The Stage-5b re-measurement
  with the new symbolic-learnable infrastructure (see below) puts the
  effect at a more conservative ~10× — still a robust win, less
  dramatic than the prior number suggested.
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

Stage-4b probed M_proj reachability under graded coefficients
(`experiments/path_b_graded_mproj.py`). For the framework recipe
`c=0, d=z, s=ω`, a generic input `(a, b)` produces output
`(-b·0, a·0 + b)` — the `b` survives in the second slot because
`d·s = 0·ω → 1` fires automatically, while the matrix-recipe
baseline (with `0` treated as numeric zero) collapses to the all-zero
matrix and loses `b` entirely. The `−b·0` and `a·0` terms are
deferred-annihilation IOUs in framework-native form; they are *not*
plain numeric zero.

Inputs containing `ω` themselves — `(a, ω)`, `(ω, b)`, etc. — break
the recipe in non-IOU ways: bare `±1` terms appear in the wrong
slots because the same identity (`0·ω → 1`) that the recipe
exploits at `d·s` also fires at `d·x_a` or `d·x_b`. James's reading
of this: when the framework eventually carries an "accumulation"
primitive, the `ω` injection / absorption pattern looks structurally
analogous to spike behavior in biological neurons. Not literally a
spike, but the same place in the algebra.


Symbolic gradient training (Stage 5)
------------------------------------

Past attempts to embed Traction semantics in a numerical layer
(PyTorch / numpy float-from-the-start) ran into "lost in translation"
pathologies — framework identities that fire automatically in sympy
silently disappear when atoms are coerced to floats. So the
trainable layer keeps the entire forward pass and gradient
expressions in sympy. Floats only appear at two narrow edges:

- The **current numeric value** of each `LearnableParameter`.
- **Gradient evaluation at a point**, via `sp.lambdify` of the
  symbolic gradient expression.

Atoms (Zero, Omega, Null) in fixed parameters pass through the
symbolic forward as usual; they get projected to ℂ via the
framework's own `project_complex` only at the moment a numeric
gradient or loss value is needed. They are never silently coerced.

Training loop:

1. Declare which fields are learnable via `LearnableParameter`.
2. Build the loss as a sympy expression (e.g. squared error between
   `forward()` output and a target).
3. `SymbolicOptimizer` builds the gradient expressions once via
   `sp.diff`, lambdifies them for fast repeated evaluation, and
   updates each parameter's `.value` in place.

The `experiments/learnable_demo.py` script trains one neuron's `s`
to converge to each of the three real cardinals (0 hyperbolic,
1 parabolic, −1 elliptic). All three converge to machine precision
in ~40 steps. Every step is recorded into a Trace JSON including
the unchanging symbolic gradient expression — gradient descent
itself is fully inspectable.

ω as a fourth cardinal is *not* reachable by smooth descent on a
real-valued `s` (proven by the M_proj chain-image work), and is
deferred. When eventually added, it'll be a discrete substitution,
not a smooth limit — which lines up with the spike hypothesis above.


Inductive-bias test (Stage 5b)
------------------------------

`experiments/path_b_inductive_bias.py` pits LearnableNeuron (one
learnable parameter, `s`, with `c=0, d=1` fixed) against
FreeLinearLayer (four learnable parameters — an arbitrary 2×2
matrix) on noisy Möbius-flavored data drawn from `M(c=0, d=1, s=0.7)`.
Both are trained with the same SymbolicOptimizer (duck-typed protocol).
Reporting median over 5 seeds across 4 data sizes × 4 noise levels.

Headline: **LearnableNeuron generalizes ~10× better than the free 2×2
baseline at every cell where noise > 0.** The L/F test-loss ratio
sits around 0.05–0.10 across the entire grid, regardless of how much
training data is thrown at the problem. Structural recovery is also
better — `s_err` stays ≤ 0.27 even at noise=0.5; the free matrix is
off by ~0.8 Frobenius at the same noise.

The clearest single cell — 3 pairs, zero noise:

- LearnableNeuron: test loss `6.35e-27`, `|recovered s - target s|` = `1.4e-13`.
- FreeLinear: test loss `2.18e-04`, matrix off by `0.026` Frobenius.

This is the inductive bias in its purest form. The training set has
6 equations for FreeLinear's 4 parameters, but all 6 lie in a
1-parameter family `M(s)`, so the solution space has slack —
FreeLinear can fit the training set while landing on the wrong
matrix. LearnableNeuron's single parameter is uniquely determined.

Honest correction to the README's earlier "up to 1000×" claim from
Stage 3: in this measurement the inductive-bias advantage is a
robust **~10×, not 1000×**. The Stage 3 setup used the older
SymbolicNeuron with a different gradient scheme, so the numbers
aren't directly comparable, but the new measurement is what to
quote going forward.


Empirical chain-image test (Stage 5c)
-------------------------------------

`experiments/path_b_chain_image.py` puts the Stage-3 theoretical
finding ("M_proj outside the 2-layer chain image") on a measurement
scale, then sweeps depth to see when the floor breaks.

Three parts, all with `ChainLearnable` student:

- **Part A — in-image teacher, depth 2:** student converges to
  loss `5.7e-5`. Theory predicts reachable; confirmed.
- **Part B — M_proj target, depth 2:** student floors at loss
  `1.15e-1`. Theory predicts unreachable; floor matches.
- **Part C — depth sweep against M_proj:** loss drops 10× between
  depth 2 and depth 3, then only modestly between depth 3 and 4.

| Depth | Final loss | Per-pair |
| ----- | ---------- | -------- |
| 1     | 1.42e-1    | 2.83e-2  |
| 2     | 1.15e-1    | 2.30e-2  |
| 3     | **9.93e-3**| 1.99e-3  |
| 4     | 4.04e-3    | 8.08e-4  |

The break at depth 3 is the new empirical finding — the existing
theorem only covers depth 2, and now we have evidence that adding
one more layer is enough for the chain image to reach (or come very
close to) M_proj. The mechanism is plausibly that deeper chains
have more chances to slide one layer to a det=0 setting (which
requires `s² ≥ 4`, outside the cardinal disk), and M_proj being
det=0 / rank-1 means at least one such layer is required.

Per-depth learning rates were used (lr=0.05 for depth ≤ 2, lr=0.02
for depth 3, lr=0.01 for depth 4) — gradients grow non-linearly
with chain length so a constant lr that works at depth 2 overflows
deeper.

Caveats still open:
- Single seed per depth; depth-3 break needs multi-seed verification.
- Depth 3 floor at 9.93e-3 isn't zero. Could be near-reach rather
  than full inclusion; depth 4 only modestly improving (4e-3)
  doesn't disambiguate.


Layout
------

The recommended path through the code if you want to understand it:

1. `jenk/traction.py` — the algebra atoms.
2. `jenk/graded_neuron.py` — what a single forward pass looks like.
3. `jenk/trace.py` — how diagnostics are captured.
4. `experiments/graded_neuron_demo.py` — a runnable example.
5. `experiments/path_b_four_way.py` — the matrix-recipe benchmark and
   the M_proj reachability question.
6. `experiments/path_b_graded_mproj.py` — the M_proj recipe scan
   (Stage 4b) showing where graded coefficients win and where they
   produce non-IOU residues.
7. `jenk/learnable_neuron.py` — the trainable layer (Stage 5).
8. `experiments/learnable_demo.py` — convergence to the three real
   cardinals with full symbolic-gradient traces.

`SymbolicNeuron` (`jenk/symbolic_neuron.py`) and the older `Graded`
carrier (`jenk/graded.py`) are kept for comparison and historical
reference; new work should build on `traction.py` and `graded_neuron.py`.


What's deliberately out of scope (for now)
------------------------------------------

- **PyTorch.** Ruled out for the symbolic phase, not just deferred.
  PyTorch tensors don't understand custom sympy `Expr` subclasses,
  so making Zero/Omega/Null tensorial means coercing them to floats
  — exactly the lost-in-translation problem this implementation is
  designed to avoid. NumPy is acceptable at the narrow edges
  (parameter storage, lambdify backend); PyTorch is for a separate
  later stage that must be validated against this reference.
- **Reaching ω by gradient descent.** Real-valued `s` covers three
  cardinals; ω is a discrete substitution. When ω-as-spike becomes
  the active question, the mechanism will be a discrete switch, not
  a smooth approach.
- The `0 + ω = 0` and similar cross-grade addition rules from the
  framework's README. Treated as not-yet-trusted; cross-grade
  additions are kept as deferred sympy expressions in the diagnostic
  layer rather than collapsed by an arbitrary rule.
- The Z-action (`GradedElement` / `Z_n(x)`) defined in
  `jenk/traction.py`. Present in the file, unused everywhere else.
- A visualizer for trace JSON. Probably worthwhile once enough traces
  exist to know what's worth showing — the learnable demo's traces
  (606 records each) start to make the case.
