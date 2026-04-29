"""Chebyshev ring  Q[s][g] / (g^2 - s*g + 1)  with real-valued s.

Faithful Python port of the Java CliffordRingElement. Each element is
stored as (a, b, s) representing  a + b*g  in the ring with trace
parameter s. Multiplication uses  g^2 = s*g - 1 :

    (a1 + b1*g)(a2 + b2*g)
        = a1*a2 + a1*b2*g + b1*a2*g + b1*b2*g^2
        = a1*a2 + (a1*b2 + b1*a2)*g + b1*b2*(s*g - 1)
        = (a1*a2 - b1*b2) + (a1*b2 + b1*a2 + b1*b2*s) * g.

The trace s smoothly parameterizes the signature of the algebra:

    |s| < 2     elliptic    g is a root of unity, |g| = 1, rotation-like
    |s| = 2     parabolic   double root g = +/-1, shear / phase boundary
    |s| > 2     hyperbolic  g real with two distinct roots, boost-like

Two operands must share the same s to compose - different traces are
different rings, with no canonical product. Mixed-s composition raises
ValueError.

This module is intentionally float-only (no graded-s lift). The lifted
variant lives in jenk.lifted_ring.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tolerance band around |s| = 2 for the parabolic classification. Matches
# the Java port's 1e-9 epsilon in signatureRegion.
_PARABOLIC_EPS = 1e-9

# Norm threshold below which inverse() raises. The Java port checks
# n == 0.0 exactly; under float arithmetic, light-cone elements at |s| > 2
# have norms of order 1e-16, and the resulting inverse blows up by 1e16.
# Treating those as singular is more useful than letting them through.
_INVERSE_NORM_EPS = 1e-12


def _require_same_s(left: 'RingElement', right: 'RingElement', op: str) -> None:
    if left.s != right.s:
        raise ValueError(
            f"Ring {op} requires matching s; got {left.s} and {right.s}"
        )


@dataclass(frozen=True)
class RingElement:
    """Element  a + b*g  in  Q[s][g] / (g^2 - s*g + 1)  with real s.

    Stored as three floats: coefficient `a` (scalar part), coefficient `b`
    (g-component), and trace `s`. The triple `(a, b, s)` is enough to
    recover all algebra operations via the formulas at the top of the
    module.
    """
    a: float
    b: float
    s: float

    # --- Constructors ------------------------------------------------------

    @staticmethod
    def identity(s: float) -> 'RingElement':
        """The multiplicative identity at trace s: 1 + 0*g."""
        return RingElement(1.0, 0.0, s)

    @staticmethod
    def generator(s: float) -> 'RingElement':
        """The pure generator at trace s: 0 + 1*g."""
        return RingElement(0.0, 1.0, s)

    # --- Additive structure -----------------------------------------------

    def add(self, other: 'RingElement') -> 'RingElement':
        _require_same_s(self, other, 'addition')
        return RingElement(self.a + other.a, self.b + other.b, self.s)

    def sub(self, other: 'RingElement') -> 'RingElement':
        _require_same_s(self, other, 'subtraction')
        return RingElement(self.a - other.a, self.b - other.b, self.s)

    def negate(self) -> 'RingElement':
        return RingElement(-self.a, -self.b, self.s)

    def scale(self, k: float) -> 'RingElement':
        """Scalar multiplication on (a, b); s unchanged."""
        return RingElement(self.a * k, self.b * k, self.s)

    # --- Multiplicative structure -----------------------------------------

    def mult(self, other: 'RingElement') -> 'RingElement':
        """Ring product via g^2 = s*g - 1."""
        _require_same_s(self, other, 'multiplication')
        a1, b1, s = self.a, self.b, self.s
        a2, b2 = other.a, other.b
        new_a = a1 * a2 - b1 * b2
        new_b = a1 * b2 + b1 * a2 + b1 * b2 * s
        return RingElement(new_a, new_b, s)

    def conjugate(self) -> 'RingElement':
        """Conjugation via  g -> g^{-1} = s - g :
                conj(a + b*g) = (a + b*s) - b*g.
        """
        return RingElement(self.a + self.b * self.s, -self.b, self.s)

    def norm(self) -> float:
        """Norm  N(a + b*g) = a^2 + a*b*s + b^2 .

        Equals the scalar component of  x * conj(x); the g-component of
        that product is identically zero by the conjugation rule. Sign
        convention matches the Java port: positive-definite for |s| <= 2,
        indefinite for |s| > 2 (with zero divisors on the light cone).
        """
        return self.a * self.a + self.a * self.b * self.s + self.b * self.b

    def inverse(self) -> 'RingElement':
        """Multiplicative inverse  x^{-1} = conj(x) / N(x).

        Raises ZeroDivisionError when |N(x)| < _INVERSE_NORM_EPS (zero
        divisor on the |s| = 2 edge or on the light cone for |s| > 2).
        The tolerance differs from the Java port's exact == 0.0 check;
        see _INVERSE_NORM_EPS at module scope.
        """
        n = self.norm()
        if abs(n) < _INVERSE_NORM_EPS:
            raise ZeroDivisionError(
                f"RingElement has near-zero norm "
                f"(s={self.s}, a={self.a}, b={self.b}, n={n})"
            )
        return self.conjugate().scale(1.0 / n)

    # --- Classification ---------------------------------------------------

    def signature_region(self) -> str:
        """Return 'elliptic' / 'parabolic' / 'hyperbolic' for the current s.

        Uses an absolute-value tolerance of 1e-9 around |s| = 2 to match
        the Java port. The classification depends only on s, so it is
        independent of (a, b).
        """
        abs_s = abs(self.s)
        if abs_s < 2.0 - _PARABOLIC_EPS:
            return 'elliptic'
        if abs_s > 2.0 + _PARABOLIC_EPS:
            return 'hyperbolic'
        return 'parabolic'

    # --- Operator overloads (sugar over the named methods) ----------------

    def __add__(self, other: 'RingElement') -> 'RingElement':
        return self.add(other)

    def __sub__(self, other: 'RingElement') -> 'RingElement':
        return self.sub(other)

    def __neg__(self) -> 'RingElement':
        return self.negate()

    def __mul__(self, other):
        if isinstance(other, RingElement):
            return self.mult(other)
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"RingElement({self.a:+.4f} {self.b:+.4f}*g | "
            f"s={self.s:+.4f}, {self.signature_region()})"
        )
