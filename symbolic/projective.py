"""Graded projective basis with cyclic squaring rule, plus a single neuron.

Basis  k_0, ..., k_{N-1}  with multiplication rules

    k_i · k_i  =  k_{(i + 1) mod N}     (squaring; James's rule)
    k_i · k_j  =  k_{(2i + j) mod N}    (mixed products, i ≠ j)

The mixed-product rule is asymmetric — k_0·k_1 = k_1 but k_1·k_0 = k_2 — so
the algebra is non-commutative.  Other rules respecting the squaring identity
exist; this is the simplest non-commutative one that scales to any N.

A ``Projective(N)`` value is a length-N coefficient vector over the reals.
All arithmetic stays in raw numpy / python — sympy does not cross this
boundary.

v0 encoding
-----------
Four slots ``(in_0, op_0, in_1, out_0) ↦ (k_0, k_1, k_2, k_3)``.
``in_0`` and ``in_1`` are the two operands; ``op_0`` is an integer enum
(currently constant at 2 = ``+``, the only op v0 trains on); ``out_0`` is
initialised to 0 in inputs and is read out from the network output as the
prediction.

The neuron has one bias (also a 4-slot Projective).  Forward is
``output = input · bias``; prediction is ``output[OUT_0]``; loss is squared
error; gradients are computed analytically from the bilinear product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


# Slot conventions for the v0 four-slot encoding
IN_0, OP_0, IN_1, OUT_0 = 0, 1, 2, 3


def basis_product_index(i: int, j: int, n: int) -> int:
    """Index k where  k_i · k_j = k_k  under the cyclic-squaring algebra."""
    if i == j:
        return (i + 1) % n
    return (2 * i + j) % n


@dataclass
class Projective:
    """A coefficient vector in the N-dimensional graded projective basis."""
    coeffs: np.ndarray

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=float)

    @property
    def n(self) -> int:
        return len(self.coeffs)

    @staticmethod
    def zeros(n: int) -> "Projective":
        return Projective(np.zeros(n))

    def __getitem__(self, idx: int) -> float:
        return float(self.coeffs[idx])

    def __add__(self, other: "Projective") -> "Projective":
        return Projective(self.coeffs + other.coeffs)

    def __sub__(self, other: "Projective") -> "Projective":
        return Projective(self.coeffs - other.coeffs)

    def __mul__(self, other: Union["Projective", float, int]) -> "Projective":
        if isinstance(other, Projective):
            assert self.n == other.n, f"basis size mismatch: {self.n} vs {other.n}"
            n = self.n
            out = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    k = basis_product_index(i, j, n)
                    out[k] += self.coeffs[i] * other.coeffs[j]
            return Projective(out)
        return Projective(self.coeffs * float(other))

    def __rmul__(self, other) -> "Projective":
        return Projective(self.coeffs * float(other))


def encode_problem(a: float, op: int, b: float, n: int = 4) -> Projective:
    """Encode  (a, op, b)  into a 4-slot Projective with ``out_0`` set to 0."""
    coeffs = np.zeros(n)
    coeffs[IN_0] = a
    coeffs[OP_0] = op
    coeffs[IN_1] = b
    return Projective(coeffs)


@dataclass
class ProjectiveNeuron:
    """Single-layer neuron over the projective algebra.

    ``output = input · bias``; prediction is the ``out_0`` (k_3) coefficient
    of the output.  Loss is squared error; SGD updates the 4 bias scalars
    using analytic gradients.
    """
    bias: Projective

    @staticmethod
    def zeros(n: int = 4) -> "ProjectiveNeuron":
        return ProjectiveNeuron(bias=Projective.zeros(n))

    def feedforward(self, input_: Projective) -> Projective:
        return input_ * self.bias

    def predict(self, input_: Projective) -> float:
        return self.feedforward(input_)[OUT_0]

    def gradients(self, input_: Projective, target: float) -> np.ndarray:
        """``dL / d(bias_j)`` for ``L = (output[OUT_0] − target) ** 2``."""
        n = input_.n
        residual = self.predict(input_) - target
        d_pred = np.zeros(n)
        for j in range(n):
            for i in range(n):
                if basis_product_index(i, j, n) == OUT_0:
                    d_pred[j] += input_.coeffs[i]
        return 2.0 * residual * d_pred

    def update(self, grad: np.ndarray, lr: float) -> None:
        self.bias.coeffs -= lr * grad

    def step(self, input_: Projective, target: float, lr: float) -> float:
        pred = self.predict(input_)
        grad = self.gradients(input_, target)
        self.update(grad, lr)
        return pred
