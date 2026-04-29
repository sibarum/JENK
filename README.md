
Jεηk: Computational Flexibility
====

Jenk is a 4-mode 2-parameter mapping space.

|   | Mode       | Traction | δ  |
|---|------------|----------|----|
| j | Hyperbolic | 0^(0/2)  | +1 |
| ε | Parabolic  | 0^(1/2)  | 0  |
| η | Elliptic   | 0^(ω/2)  | −1 |
| k | Projective | 0^(-1/2) | ω  |

The rational numbers occupy the horizontal axis
and the Traction numbers occupy the vertical axis.
Note that this diagram is distorted to illustrate the
"point at zero" as a Traction unit circle, with erasure
(⊘) as the origin.

Erasure is: x-x=⊘

                         k
                         ↑
                   Multiplicative
                       (ω^2)
                         |
                      _(ω^1)_
       Additive      /   |   \
     η ←--(-2)---(-1)---(⊘)---(+1)---(+2)--→ j
                     \   |   / Unit
                      -(0^1)- Circle
                         |
                       (0^2)
                         |
                         ↓
                         ε

The mode of an element isn't a property of the element's location - 
it's a property of the direction by which you approach the unit.
Two paths to the same value can have different modes,
and the mode tells you about the path, not the destination.

Traction Theory
===============

Identities from Traction Theory:

- 1/0=ω
- 0ω=1
- 0-0=0
- ω-ω=0
- 0+ω=0
- ω=-0
- 0^0=1
- 0^1=0
- 0^(-1)=ω
- 0^ω=-1
- ω^0=1
- ω^1=ω
- ω^(-1)=0
- ω^ω=-1

ω is not an infinity-like quantity.
Think of it more like 2*pi: it's a full cycle.
Or -0: the zero beyond the singularity that becomes real somehow.
Better to call it omega, not infinity.

Implementation
==============

Neural Jεηk
-----------

A high-performance PyTorch primitive for complex 2D and 4D
equivariance with learnable parameter computation mode parameter "s".

Approximate trig, compressible weights, and learned computation mode
makes this architecture highly effective at learning a broad variety of tasks
without sacrificing efficiency.

Every mode uses exactly the same math, making hardware efficiency native to the technique.
The only difference is the value for δ, no branching
even if the mode changes for every input.

The "s" parameter corresponds with the square of a value on the traction
complex surface:

0^(s/2)

This enables a continuum of complex generators to be learned by the network.

Notes
-----

The Jenk construction is not a closed form. There are some expressions that can't be reduced:

- 0^((w+1)/2)