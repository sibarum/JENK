# Projective NN — Outstanding Issues & Decisions

Status as of conversation context-pause. This document captures every known issue, structural concern, code-hygiene problem, and forward-looking design decision that was raised during the projective-encoded NN work but not yet resolved. Read this first before making any further edits to `symbolic/projective.py` or related code.

---

## RESCUED: v2/v3/v4 work moved to `analysis/`

The substantive output of the v2/v3/v4 thread now lives in `analysis/` (tracked). The original location `tmp/diagnostics/` is gitignored and would have lost the files on any clean. Inventory:

| File | What it is |
|---|---|
| `analysis/v2_path_count_analysis.py` + `v2_path_count_results.txt` | v2 cross-op dead-slot derivation; identified `*` degeneracy and polynomial-vs-rational issue for `/` |
| `analysis/v2_swapped_path_count_analysis.py` + `v2_swapped_results.txt` | Cardinal-aligned (div ↔ in_0 swap) version of above; showed dead-slot pattern is INVARIANT under encoding swap |
| `analysis/v3_ratio_readout_feasibility.py` + `v3_ratio_readout_feasibility_results.txt` | Symbolic feasibility check for ratio readouts (option B from Möbius discussion); ALL infeasible (only trivial bias=0 satisfies) |
| `analysis/v4_complex_algebra_feasibility.py` + `v4_results.txt` | Complex algebra (i factor on off-diagonal) feasibility; `{add, sub}` jointly feasible, `*` and `/` not |
| `analysis/v2_train.py` | Multi-op training script (refactored to ops-agnostic with all 4 ops). Run output stays in `tmp/diagnostics/v2_train.log` (gitignored — logs are too big and aren't repo content). |
| `analysis/v2_per_op_train.py` | Per-op vs multi-op comparison (refactored, includes baseline + all 4 ops) |
| `analysis/v2_per_op_train_swapped.py` | Same comparison under div ↔ in_0 swap |
| `analysis/visualize_op_function_classes.py` + `op_function_classes.png` | The picture showing polynomial-fit residual concentrates entirely in the b≈0 strip for division |

---

## SUSPECT 1: Mixed-product rule is unjustified

**File:** `symbolic/projective.py`, lines 41-45 (`basis_product_index`)

The diagonal rule `k_i · k_i = k_{(i+1) mod N}` came directly from James's spec. The off-diagonal rule `k_i · k_j = k_{(2i + j) mod N}` for `i ≠ j` was invented to make the algebra closed and non-commutative. **It is not derived from anything framework-native.** It does not correspond to:
- `traction.py`'s defined identities (`0·ω = 1`, etc.)
- Any natural algebra structure (Clifford, group ring, hypercomplex)
- The Riemann sphere + 90° Möbius framework James actually uses (see Suspect 2)

The empirical symptom: cross-op dead-slot table is asymmetric — `output[5]` kills no bias, `bias[4]` is alive at every output. The "k_4 and k_6 flipped" observation traced this to the multiplication rule, not the encoding (the encoding swap left the dead-slot pattern unchanged).

**What needs to happen:** replace `basis_product_index` with a rule derived from `traction.py` identities, using `traction_simplify()` to compute the actual multiplication table for the framework's basis elements `{1, 0, −1, ω}` (or whichever 4-cycle / 6-point structure the framework intends).

---

## SUSPECT 2: Basis doesn't loop meaningfully

The cyclic squaring rule `k_n² = k_{n+1 mod N}` is a *symbolic* loop — basis indices cycle by name. But the framework's actual loop is the **90° Möbius rotation `M(z) = (z+i)/(iz+1)`** on the Riemann sphere, which cycles `{0, i, ∞, −i}` with `±1` as fixed points. Our algebra implements neither this Möbius nor the Riemann-sphere structure; it just labels indices cyclically.

The squaring rule should really be `M(k_i) = k_{(i+1) mod N}` where `k_i` is an actual point on the sphere — not a symbolic index. The current implementation treats slot identity as bookkeeping, when it should encode actual sphere positions.

**What needs to happen:** decide whether the architecture should:
1. Anchor `k_0..k_3` to specific Riemann-sphere points (`{0, i, ∞, −i}`) and implement the Möbius literally as the squaring/shift operation.
2. Stay symbolic but at least anchor multiplication to `traction_simplify()` (Suspect 1's fix).
3. Continue with the symbolic approximation but explicitly document it as such.

---

## OUT-OF-SYNC: `V2_OUT_0` is a misleading name

**File:** `symbolic/projective.py`, line 107: `V2_OUT_0 = 6`

This constant is named "OUT" but slot 6 is **not** where the prediction is read from. Slot 6 is where `encode_problem_v2` puts a constant `1.0` as a free input feature (per James's plan to break the wasted-slot issue from v0). The actual readout slot in every v2 training script is `readout_slot=0`.

The same misleading name appears in `analysis/v2_per_op_train_swapped.py` line 38: `SLOT_OUT_0 = 6`.

**What needs to happen:**
- Rename `V2_OUT_0` to something accurate: `V2_INIT_1_SLOT`, `V2_BIAS_INPUT_SLOT`, `V2_FREE_FEATURE_SLOT`, or similar.
- Same in the swapped script.
- Document that the actual readout location is `readout_slot=0` (set on the neuron object) and explain *why* slot 0 was chosen (path-count analysis identified it as well-conditioned for the original v2 layout).

---

## OUT-OF-SYNC: `readout_slot=0` is unjustified after the swap

**File:** `analysis/v2_per_op_train_swapped.py`, lines 73, 95

`readout_slot=0` was chosen for the **original** v2 layout (slot 0 = in_0) based on path-count analysis identifying it as well-conditioned for that specific arrangement. After the div ↔ in_0 swap, slot 0 holds `op_/` (the division indicator). The path-count properties under the swapped layout were never re-derived. We just kept `readout_slot=0` mechanically.

**What needs to happen:**
- Re-run the v2 path-count analysis under the swapped layout to determine which output slot is well-conditioned.
- Use that as the readout slot for the swapped neuron, instead of blindly using 0.
- Or document explicitly: "readout_slot=0 in the swapped layout reads from the cardinal-pit position (`op_/`'s slot), as a deliberate test of the cardinal-alignment hypothesis."

---

## OUT-OF-SYNC: Two source-of-truth definitions for `op_/`'s slot

**Files:**
- `symbolic/projective.py` line 105: `V2_OP_DIV = 4`
- `analysis/v2_per_op_train_swapped.py` line 32: `SLOT_OP_DIV = 0`

These intentionally disagree (one is the production layout, one is the hypothesis-test layout) but the conflict is buried in a tmp file. There's no documentation in `projective.py` indicating that the swapped variant exists or what it tests.

**What needs to happen (after deciding the swap question):**
- Either promote the swap into `projective.py` (modify `V2_OP_DIV = 0`, `V2_IN_0 = 4`, update all dependent code in tests and analysis scripts).
- Or delete the swapped script if the experiment is done.
- Don't leave both around without cross-references.

---

## REMAINING DUPLICATIONS (single-source-of-truth violations)

After refactoring `v2_train.py` and `v2_per_op_train.py` to use a single `TARGETS`/`OPS` definition, these files still hardcode op lists in multiple places:

| File | Lines | Pattern |
|---|---|---|
| `analysis/v2_path_count_analysis.py` | 115, 133, 147, 159 | `for op_slot in [1, 2, 3, 4]:` (4 occurrences of op-slot indices) |
| `analysis/v2_swapped_path_count_analysis.py` | 94, 108 | `for op_name in ['div', 'sub', 'add', 'mul']:` (2 occurrences; `OP_INFO` exists at L40-45 as the right single source) |
| `analysis/v3_ratio_readout_feasibility.py` | 176 | `for op_name in ['+', '-', '*', '/']:` (should be `OP_TARGETS.keys()`) |
| `analysis/v4_complex_algebra_feasibility.py` | 130-136 | Subset enumeration with string literals; should reference `OP_INFO.keys()` |

These are code smell, not bugs (none exclude division). But they're the same pattern that bit us in `v2_train.py` originally — when assumptions are duplicated, they get out of sync silently.

---

## METHODOLOGICAL: Training loss vs evaluation loss

**Symptom in conversation:** I claimed division was "32× worse than baseline" based on training loss over the last 5000 steps of `v2_train.py`. Re-running with proper held-out evaluation showed division at **2× baseline** under `{+,−,*,/}` joint training — much less catastrophic.

**Root cause:** training loss includes the loss measured at each gradient step, with the bias still oscillating. Evaluation loss measures the *final* bias on a *held-out* sample. They can differ by 10× or more when the bias is in a noisy regime.

**What needs to happen:** any future training script must report **both**:
- Training loss (windowed average over last N steps) — diagnostic for whether the optimization stabilized.
- Held-out evaluation loss on a fresh distribution — the actual performance number.

The refactored `v2_per_op_train.py` does this correctly (see `evaluate_on_all_ops` and `baseline_per_op`); the older `v2_train.py` still only reports training loss.

---

## EMPIRICAL FINDINGS (durable, should not be re-derived)

These are the substantive results from the conversation. They should be preserved somewhere version-controlled.

### Function-class limitations

The architecture `output = (input · input) · bias` is **degree-2 polynomial** in `(a, b)`. Confirmed by `op_function_classes.png`: the closed-form L²-best degree-2 polynomial fits `+`, `−`, `*` exactly (zero residual on the visualisation grid) but cannot represent `a/b`. The residual for `/` concentrates entirely in the strip near `b = 0` (max ≈ 287 in the figure).

### v2 path-count summary (original layout)

Cross-op dead-slot map (each output → bias slot dead under EVERY op):
```
output:    0  1  2  3  4  5  6
killed:    0  5  3  1  6  -  2     (bias 4 never killed; output 5 kills nothing)
```
This map is **invariant under the div ↔ in_0 swap** — verified empirically. The asymmetry comes from the multiplication rule, not the encoding.

Per-op spanning verdicts at output[0]:
- `{+}`: well-conditioned (rank 6/6)
- `{−}`: well-conditioned
- `{*}`: 1-D init-frozen direction (rank 5/6) — same multiplicative-interference pattern
- `{/}`: structurally unfittable (polynomial vs rational)

### Empirical training results (held-out eval, ratio to predict-0 baseline)

| trained on | `+` | `−` | `*` | `/` | layout |
|---|---|---|---|---|---|
| `{+}` | 0.00× | 1.29× | 1.02× | 3.23× | original |
| `{−}` | 0.31× | 0.00× | 1.06× | 5.32× | original |
| `{*}` | 18.10× | 18.64× | 0.04× | 243.35× | original |
| `{/}` | 1.02× | 0.99× | 0.97× | 0.87× | original |
| `{+,−}` | 0.13× | 0.05× | 1.12× | 3.78× | original |
| `{+,−,*}` | 2.02× | 1.61× | 0.75× | 22.63× | original |
| `{+,−,*,/}` | 0.16× | 0.26× | 0.90× | 2.04× | original |
| `{+,−,*}` | 0.34× | 0.38× | 1.07× | 6.66× | swapped |
| `{+,−,*,/}` | 0.06× | 0.32× | 0.99× | 3.37× | swapped |

Key empirical takeaways:
1. **`{/}` alone trains to bias ≈ 0**, achieving polynomial L²-floor (just below baseline). Confirms polynomial-vs-rational ceiling for single-op fit.
2. **Adding `/` to multi-op training improves `+` and `−` significantly** (e.g., `{+,−,*}` → `{+,−,*,/}`: + goes 2.02× → 0.16×, − goes 1.61× → 0.26×). Division acts as a regulariser.
3. **`{*}` alone is catastrophic for every other op** (243× baseline for `/`).
4. **Cardinal swap helps polynomial-fittable ops jointly** but does not lift the polynomial-vs-rational ceiling for `/`.

### Joint-feasibility findings (analytical)

- **v3 ratio readout (option B):** infeasible at every (k_N, k_D) pair — only trivial bias=0 solves the constraint system, regardless of op subset chosen. Cause: polynomial-degree mismatch (LHS degree 2, RHS degree 3+ for non-constant target).
- **v4 complex algebra (off-diagonal i factor):** `{add, sub}` jointly feasible (dim 5 nullspace at most readouts); `{add, sub, mul}` infeasible at every readout; `{add, sub, mul, div}` infeasible at every readout. Complex extension resolves inverse-pair joint feasibility but doesn't fix mul or div.

---

## FORWARD-LOOKING DESIGN NOTES (from James, not yet implemented)

These are captured in `~/.claude/projects/.../memory/project_projective_nn_plans.md` and `project_projective_nn_readout_finding.md`. Listed here for visibility:

1. **Op enum reorder when expanding past +:** `1=−, 2=+, 3=*, 4=/` (adjacent indices are inverse pairs). Used for v2 encoding.
2. **Learnable `basis_product_index`** — make the multiplication rule itself a learnable tensor instead of a fixed analytic function. Sibling to "the Graph" architecture-search idea.
3. **`out_0 = 1` init** — done in v2; closes the wasted-slot issue from v0 where `input[OUT_0] = 0` killed any gradient through `f`.
4. **ReLU-activated shuffle** — some kind of nonlinear activation involving slot permutation between layers. Concrete shape TBD.
5. **Dead-slot resurrection chain** — when a slot's gradient is dead: (a) edge mutation in learnable basis, (b) inverted sibling copy if mutation fails, (c) per-sibling component recombination on biased dataset.
6. **Möbius transform integration** — three candidate locations:
   - At the readout: `prediction = M(output[k])`
   - Between hidden and output: apply M to hidden values before bias multiplication
   - As the algebra's shift rule itself: replace `k_i² = k_{i+1}` with literal Möbius application
7. **Split-quaternion / coquaternion / SU(1,1) interpretation** — alternative to Riemann-sphere SU(2). Idempotent basis `e± = ½(1 ± j)` diagonalizes; algebra splits into independent scaling on null lines. James indicated SU(2) sphere first, hyperboloid as fallback.
8. **Different op-encoding values** — instead of one-hot `{0, 1}`, possibilities include `{−1, +1}`, `{1/2, −1/2, 2/1, −2/1}` (matched to Möbius-cycle positions), or learned embeddings.
9. **Ratio architecture** — bias has Re and Im components; Im components encode division pathway. Tested analytically (v3) and shown infeasible under the current rule, but could be revisited with the framework-derived rule.

---

## TESTS NOT WRITTEN

- **No tests exist for the v2 architecture** in `tests/test_projective_neuron.py`. The v0 (addition) and v1 (multiplication, plus readout-swap experiments) are well-tested. v2 was attempted via heredoc append but the change was rejected; the test class `TestV2OneHotEncoding` was never added to the file.
- **The path-count analyses are not pytest-integrated.** They live as standalone scripts in `analysis/` with text-output. Promoting them to pytest tests with assertions would document the expected findings durably.
- **The visualisation script** `visualize_op_function_classes.py` produces an image but has no test verifying the polynomial fit residual lands in the expected places.

---

## MEMORY (durable across sessions)

Saved during the conversation:
- `project_projective_nn_plans.md` — forward-looking design notes (the list above)
- `project_projective_nn_readout_finding.md` — v1 finding that readout-slot choice (not architecture) determines convergence geometry; the v1 path-count table for all 4 readouts

These are in `~/.claude/projects/.../memory/` and survive across Claude Code sessions.

---

## NEXT STEPS, IN PRIORITY ORDER

1. ~~**`git mv tmp/diagnostics/* analysis/`** + commit.~~ DONE — files now in `analysis/` and tracked.
2. **Decide the suspect-1 question:** anchor multiplication to `traction.py` (real fix) vs leave the made-up rule (status quo with known limits). If the former, derive the multiplication table for `{1, 0, −1, ω}` from `traction_simplify()` and rewrite `basis_product_index`.
3. **Decide the suspect-2 question:** implement the actual 90° Möbius (real fix) vs keep symbolic squaring (approximation).
4. **Rename `V2_OUT_0`** to something accurate, document `readout_slot=0`'s justification.
5. **Promote v2 architecture into tests** — add `TestV2OneHotEncoding` (and its swapped sibling) to `tests/test_projective_neuron.py` with assertions on the empirical findings table above.
6. **DRY up the remaining duplicated op-lists** in the four files listed in "REMAINING DUPLICATIONS."
7. **Decide on Möbius integration location** (readout vs middle vs algebra-core), then build it.
8. **Re-derive path-counts under the framework-derived multiplication rule** — once that rule replaces the made-up one, the dead-slot pattern, joint-feasibility verdicts, and choice of well-conditioned readout all need to be recomputed.
