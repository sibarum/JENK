# v1 Archive: Readout-Specific Degeneracy in a Bilinear Routing Architecture

## What this document is

A consolidated record of the v1 investigation. Findings are organized by what was established, not by the order in which they were established. The chronological narrative lives in the chat log; this is the spine.

Scope: single architecture, single encoding (`hidden = input · input`, `op=0`, `input = (a, 0, b, 0)`), four candidate output readouts. Everything below is specific to this setting unless explicitly marked otherwise.

---

## Headline finding

**The training degeneracy observed at convergence is a property of the readout slot, not of the architecture.**

The same forward computation, same hidden features, same loss shape, and same parameter space exhibit qualitatively different training geometry depending on which output slot the loss reads. Two readouts (outputs 0 and 2) produce well-conditioned, init-independent convergence. Two readouts (outputs 1 and 3) produce a 1-D degenerate minimum where the orthogonal-to-gradient component of the init is preserved unchanged through training.

This inverts the natural reading of the v1 raw result, in which the "live" output slot looked like the load-bearing one and the "wasted" slots looked like overhead. The well-conditioned readouts are the wasted-looking ones. The load-bearing readout is the degenerate one.

---

## Architecture and notation

- Inputs: `a`, `b` (encoding `input = (a, 0, b, 0)` at `op=0`)
- Hidden: `hidden = input · input` element-wise outer / routing → `hidden = (a·b, a², a·b, b²)`
- Bias parameters: `(e, f, g, h)` — four scalars, applied at the readout
- Output: read from one of four candidate slots `output[0..3]`, each a fixed bilinear form in `(hidden, bias)`
- Target: predict `a · b`
- Loss: MSE between readout and target

The readout slot is treated as a discrete architectural hyperparameter, not a learned weight.

---

## Path-count table (full v1 encoding, op=0)

For each output slot, the gradient on `(e, f, g, h)` per sample factors as a vector ("path-count vector") whose components are simple polynomials in `a, b`. This is the central tool of the analysis — everything else follows from it.

| slot      | path-count vector `(e, f, g, h)` | structurally dead | (g, h) projection | direction in (g, h) |
|-----------|----------------------------------|-------------------|-------------------|---------------------|
| output[0] | `(a·b, 0, a²+b², b²)`            | f                 | `(a²+b², b²)`     | varies with (a, b)  |
| output[1] | `(a·b, 2·a·b, 0, a²)`            | g                 | `(0, a²)`         | `(0, 1)`            |
| output[2] | `(a²+b², a², a·b, 0)`            | h                 | `(a·b, 0)`        | `(1, 0)`            |
| output[3] | `(0, b², a·b, 2·a·b)`            | e                 | `(a·b, 2·a·b)`    | `(1, 2)`            |

**Cyclic dead-slot pattern:** `output[k]` has a structurally dead bias coefficient at index `(k + 1) mod 4`. This is a property of the specific routing rule, not a deep architectural invariant — a different multiplication table would shift the cycle.

---

## Twin structure

The four readouts pair into two structural twins:

**Degenerate twins: outputs 1 and 3.**
Both have a per-sample gradient direction that is *rigidly fixed* — `(1, 2)` for output[3] in `(g, h)`, `(1, 2)` for output[1] in `(e, f)`. The gradient never points anywhere else, on any sample. This produces a rank-1 per-sample Hessian and a 1-D degenerate minimum.

**Well-conditioned twins: outputs 0 and 2.**
Both have a per-sample gradient direction that *varies with the input* in two of three live coordinates. Over the data distribution, the gradient spans the full live subspace. No frozen orthogonal axis.

Same architecture, two qualitatively different training regimes, selected by readout choice.

---

## The degenerate case (output[3]): exact characterization

### Per-sample Hessian

The `(g, h)` block of the per-sample loss Hessian factors as:

```
2 · (a·b)² · (1, 2)(1, 2)ᵀ
```

This is rank 1 on every sample, not just in expectation. Eigendecomposition:

| eigenvalue        | eigenvector       |
|-------------------|-------------------|
| `10 · (a·b)²`     | `(1, 2)/√5`       |
| `0` (exact)       | `(2, −1)/√5`      |

Numerically verified: at `(a, b) = (3, 4)`, eigenvalues `0.000` and `1440.0 = 10·144`, eigenvectors aligned to predictions to <1e-2.

### Closed form for convergence

The minimum is the line `g + 2h = 1`. From any init `(g₀, h₀)`, training converges to:

```
(g, h) = (g₀, h₀) + t* · (1, 2)
t* = (1 − g₀ − 2h₀) / 5
```

The `5` is `‖(1, 2)‖² = 1² + 2²`. This is the Gauss-Newton step along the gradient direction, which equals the SGD limit because the off-axis curvature is exactly zero (no drift possible in the null direction).

### Empirical confirmation (four-seed sweep)

| init `(g₀, h₀)` | final `(g, h)` | `g + 2h` | `Δh / Δg` | predicted `t*` | actual `t` |
|-----------------|----------------|----------|-----------|----------------|------------|
| `(0.0, 0.0)`    | `(0.20, 0.40)` | `1.0`    | `2.0`     | `+0.200`       | `+0.200`   |
| `(0.5, 0.5)`    | `(0.40, 0.30)` | `1.0`    | `2.0`     | `−0.100`       | `−0.100`   |
| `(−0.3, 0.7)`   | `(−0.32, 0.66)`| `1.0`    | `2.0`     | `−0.020`       | `−0.020`   |
| `(0.4, −0.1)`   | `(0.56, 0.22)` | `1.0`    | `2.0`     | `+0.160`       | `+0.160`   |

All four land on the manifold to four decimals, with rigid `Δh/Δg = 2`, matching the closed form to numerical precision.

### On the rationality of converged values

The exact-rational landings (`0.2`, `0.4`, `0.3`, etc.) are not coincidental. They follow from:

1. The path-count vector having integer entries (path counts are integers by construction)
2. The denominator `‖v‖² = 5` being a small integer
3. Rational init → rational landing via a linear closed form

The rationality is a *fingerprint of the discrete combinatorial structure of the routing rule*. Irrational landings would be surprising; rational ones are forced.

---

## The well-conditioned case (output[0]): separation experiment

Predicting `a · b` by reading `output[0]` instead of `output[3]` — same architecture, same loss shape, only the readout slot changes.

### Predictions (from path-count table)

- `(e, g, h)` are live; `f` is structurally dead (path-count component is 0 for every sample)
- Optimum: `(e, g, h) = (1, 0, 0)`, `f = f₀` (init-preserved)
- No frozen orthogonal axis in the live subspace

### Empirical results (five-seed sweep)

| init test                     | converged `(e, g, h)`     | result        |
|-------------------------------|---------------------------|---------------|
| zero init                     | `(1, 0, 0)` ± 0.05        | as predicted  |
| `e₀ = −1` (sign flip required)| `(1, 0, 0)` ± 0.05        | sign-flip OK  |
| `f₀ = 0.7`                    | `(1, 0, 0)`, `f = 0.7`    | f frozen      |
| all-active `(0.5, 0.5, 0.5, 0.5)` | `(1, 0, 0)` ± 0.05    | as predicted  |
| (additional seed)             | `(1, 0, 0)` ± 0.05        | as predicted  |

Pairwise agreement on `(e, g, h)`: <0.05 element-wise.

### Hessian at the optimum

Smallest eigenvalue ≈ 0 with eigenvector aligned to the f-axis (>0.99 cosine). Three other eigenvalues all >100. Full rank in `(e, g, h)`.

The v1 zero-curvature direction is **readout-specific**, not architectural.

---

## Generalization criterion

A readout configuration breaks training degeneracy iff its per-sample path-count vectors *span the live bias subspace* over the data distribution.

For a single readout, this means: the path-count vector must vary with the input across at least as many independent directions as there are live (non-structurally-dead) bias coefficients. Output[0] satisfies this (`(a·b, *, a²+b², b²)` varies over a 2-D subspace in the `(e, g, h)` live coordinates). Output[3] does not (`(0, *, a·b, 2·a·b)` is rigidly proportional to `(0, *, 1, 2)` for every sample).

For multi-readout, the spanning condition is over the *joint* gradient contributions from all readouts, weighted by their per-sample feature values.

This criterion is the design tool: derive path-count vectors for any candidate architecture/encoding/readout combination, check spanning, predict degeneracy structure before running training.

---

## Methodology (transferable to v2 and beyond)

1. **Write the routing table.** For the candidate architecture, enumerate which `(i, j)` hidden-pair → bias-coefficient → output-slot routes exist.
2. **Derive path-count vectors per output slot** as polynomials in the input.
3. **Identify structurally dead bias coefficients** (path-count component identically zero across the data distribution).
4. **Check the spanning condition** for the chosen readout configuration over the joint input distribution. For multi-op encodings, check op-by-op as well as marginally — partial degeneracy across op-slices is a real failure mode.
5. **Predict** which axes are frozen, which are live, the convergence direction(s), and the closed-form landing point.
6. **Run training** and verify against predictions. Hessian eigendecomposition at the optimum is the strongest structural confirmation.

---

## Open items / next steps

### Closing v1

- Outputs 1 and 2 single-readout experiments with predictions baked in (currently in progress). Predictions:
  - **Output[1]:** `(e, f)` degeneracy along `(1, 2)` mirroring output[3]'s `(g, h)` structure exactly. Same `‖v‖² = 5` denominator. `g` structurally dead. `h` converges to 0. Closed form `t* = (1 − e₀ − 2f₀)/5`.
  - **Output[2]:** Well-conditioned in `(e, f, g)`, mirroring output[0]'s `(e, g, h)` structure. Optimum `(e, f, g) = (0, 0, 1)`. `h` structurally dead.
- If outputs 1 and 2 fire as predicted, v1 is closed: framework predictive on every readout, twin structure confirmed, cyclic dead-slot pattern confirmed.

### v2 considerations

- **Encoding change is non-trivial.** v2 introduces `out_0 = 1` (vs. `0` in v1) and one-hot op encoding (N grows to 6 or 7). Both change the hidden values that propagate through `input · input`, which changes path-count vectors.
- **Re-derive the path-count table for v2 before running training.** The output[0]/output[3] separation may not transfer — the well-conditioned slot at v1's encoding may be degenerate at v2's, or vice versa.
- **Spanning over joint `(a, b, op)` distribution.** Check op-by-op as well as marginally. Partial degeneracy across op-slices is a detectable failure mode.
- **Multi-readout vs. single-readout choice.** With the spanning criterion in hand, the question for v2 is which readout configuration (single slot vs. vector-of-ops) gives spanning over the joint distribution. This is now an analytical question answerable from the path-count table, not an empirical question requiring training runs.

### Theoretical loose ends

- The cyclic dead-slot pattern is a property of v1's specific routing rule. Does it survive small perturbations of the rule, or is it special to this multiplication table? Worth checking on one or two routing variants.
- The rationality-as-fingerprint observation generalizes: any architecture with integer-valued path counts and rational init should produce rational landings at convergence, with denominators built from path-count norms. This is testable on v2 as an additional sanity check on the framework.

---

## Reading order if returning cold

1. Headline finding (above)
2. Path-count table — central object
3. Twin structure
4. Output[3] characterization (the degenerate case, fully worked)
5. Output[0] separation experiment (the readout-specificity proof)
6. Generalization criterion
7. Methodology
8. Open items
