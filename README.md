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
translation.


The four modes
--------------

|   | Mode       | g(s)     | δ |
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
