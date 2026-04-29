"""Tests for jenk.ring (stage 1: real-valued s)."""

import math

import pytest

from jenk.ring import RingElement


# Tolerance for float-equality checks. The norm sometimes pulls things
# into machine epsilon territory; 1e-9 is comfortable.
TOL = 1e-9


def approx_equal(x: RingElement, y: RingElement, tol: float = TOL) -> bool:
    return (
        x.s == y.s
        and abs(x.a - y.a) < tol
        and abs(x.b - y.b) < tol
    )


# --- Construction ---------------------------------------------------------

class TestConstructors:
    def test_identity(self):
        e = RingElement.identity(0.5)
        assert e.a == 1.0
        assert e.b == 0.0
        assert e.s == 0.5

    def test_generator(self):
        g = RingElement.generator(0.5)
        assert g.a == 0.0
        assert g.b == 1.0
        assert g.s == 0.5

    def test_direct(self):
        x = RingElement(2.0, -3.0, 1.5)
        assert (x.a, x.b, x.s) == (2.0, -3.0, 1.5)


# --- Multiplication: the load-bearing formula -----------------------------

class TestMultiplication:
    def test_identity_is_left_identity(self):
        s = 1.7
        e = RingElement.identity(s)
        x = RingElement(0.4, -1.2, s)
        assert approx_equal(e.mult(x), x)

    def test_identity_is_right_identity(self):
        s = -0.3
        e = RingElement.identity(s)
        x = RingElement(2.0, 0.5, s)
        assert approx_equal(x.mult(e), x)

    @pytest.mark.parametrize("s", [-3.0, -1.5, 0.0, 0.5, 1.999, 2.0, 3.0])
    def test_generator_squared_equals_sg_minus_1(self, s):
        """Defining identity of the ring:  g^2 = s*g - 1.

        Equivalently, generator * generator = (-1) + s*g.
        """
        g = RingElement.generator(s)
        gg = g.mult(g)
        assert approx_equal(gg, RingElement(-1.0, s, s))

    def test_explicit_formula(self):
        """Spot-check the bilinear formula at one point."""
        s = 1.3
        x = RingElement(2.0, 3.0, s)
        y = RingElement(-1.0, 4.0, s)
        # (a1*a2 - b1*b2, a1*b2 + b1*a2 + b1*b2*s)
        expected_a = 2.0 * -1.0 - 3.0 * 4.0
        expected_b = 2.0 * 4.0 + 3.0 * -1.0 + 3.0 * 4.0 * s
        assert approx_equal(x.mult(y), RingElement(expected_a, expected_b, s))

    def test_commutativity(self):
        """The ring is commutative: x*y = y*x."""
        s = 0.7
        x = RingElement(1.2, -0.8, s)
        y = RingElement(0.3, 2.5, s)
        assert approx_equal(x.mult(y), y.mult(x))

    def test_associativity(self):
        s = 1.1
        x = RingElement(0.5, 1.0, s)
        y = RingElement(-0.3, 0.7, s)
        z = RingElement(2.0, -1.5, s)
        assert approx_equal(x.mult(y).mult(z), x.mult(y.mult(z)))

    def test_distributivity(self):
        s = -0.5
        x = RingElement(1.0, 0.5, s)
        y = RingElement(2.0, -1.0, s)
        z = RingElement(0.5, 0.25, s)
        # x*(y+z) = x*y + x*z
        lhs = x.mult(y.add(z))
        rhs = x.mult(y).add(x.mult(z))
        assert approx_equal(lhs, rhs)


# --- Conjugation: involution; defines the norm ----------------------------

class TestConjugate:
    @pytest.mark.parametrize("s", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_involution(self, s):
        """conj(conj(x)) = x for any x."""
        x = RingElement(0.7, -1.3, s)
        assert approx_equal(x.conjugate().conjugate(), x)

    def test_formula(self):
        """conj(a + b*g) = (a + b*s) - b*g."""
        s = 1.7
        x = RingElement(2.0, -1.0, s)
        c = x.conjugate()
        assert approx_equal(c, RingElement(2.0 + (-1.0) * s, 1.0, s))


# --- Norm: equals scalar part of x*conj(x); zero only on the light cone ---

class TestNorm:
    @pytest.mark.parametrize("s", [-2.5, -1.5, 0.0, 1.5, 2.5])
    def test_norm_matches_x_times_conj_scalar(self, s):
        x = RingElement(0.6, -0.4, s)
        prod = x.mult(x.conjugate())
        # The g-component of x*conj(x) should vanish by construction.
        assert abs(prod.b) < TOL
        assert abs(prod.a - x.norm()) < TOL

    @pytest.mark.parametrize("s", [0.0, 1.0, -1.0])
    def test_norm_positive_for_elliptic(self, s):
        """At |s| < 2 the form is positive-definite (non-zero element -> positive norm)."""
        x = RingElement(0.5, 0.7, s)
        assert x.norm() > 0

    def test_norm_can_vanish_on_light_cone_hyperbolic(self):
        """For |s| > 2 there exist non-zero elements with norm zero
        (zero divisors on the light cone). With s=3 and (a, b) = (1, 1),
        the discriminant suggests g_1 = (3 - sqrt(5))/2; here we just pick
        a non-zero (a, b) that makes the explicit norm a^2 + ab*s + b^2
        evaluate to a small enough value to verify the form is indefinite.
        """
        s = 3.0  # hyperbolic
        # Solve a^2 + ab*s + b^2 = 0 for (a/b): it's quadratic, with
        # roots r = (-s +/- sqrt(s^2 - 4)) / 2. For s = 3, r = (-3 + sqrt(5))/2.
        r = (-s + math.sqrt(s * s - 4)) / 2
        x = RingElement(r, 1.0, s)
        assert abs(x.norm()) < TOL


# --- Inverse: round-trips when defined; raises on zero norm --------------

class TestInverse:
    @pytest.mark.parametrize("s", [-1.5, 0.0, 1.0, 1.7])
    def test_inverse_round_trip_elliptic(self, s):
        x = RingElement(0.7, -1.1, s)
        ident = x.mult(x.inverse())
        assert approx_equal(ident, RingElement.identity(s))

    @pytest.mark.parametrize("s", [-3.0, 2.5, 4.0])
    def test_inverse_round_trip_hyperbolic_off_light_cone(self, s):
        # Pick (a, b) such that norm = a^2 + ab*s + b^2 is non-zero. With
        # b = 0 and a = 1, norm = 1 (always invertible).
        x = RingElement(1.0, 0.0, s)
        ident = x.mult(x.inverse())
        assert approx_equal(ident, RingElement.identity(s))

    def test_zero_norm_raises(self):
        s = 3.0
        r = (-s + math.sqrt(s * s - 4)) / 2
        x = RingElement(r, 1.0, s)
        # confirm norm is small enough that inverse should reject it
        assert abs(x.norm()) < TOL
        with pytest.raises(ZeroDivisionError):
            x.inverse()


# --- Signature region: classification by |s| -----------------------------

class TestSignatureRegion:
    @pytest.mark.parametrize("s", [-1.999, -1.0, 0.0, 1.0, 1.999])
    def test_elliptic(self, s):
        assert RingElement(1.0, 0.0, s).signature_region() == 'elliptic'

    @pytest.mark.parametrize("s", [-2.0, 2.0])
    def test_parabolic_exact(self, s):
        assert RingElement(1.0, 0.0, s).signature_region() == 'parabolic'

    @pytest.mark.parametrize("s", [-2.0 + 1e-10, 2.0 - 1e-10])
    def test_parabolic_within_tolerance(self, s):
        """Within 1e-9 of |s|=2, classify as parabolic (matches Java)."""
        assert RingElement(1.0, 0.0, s).signature_region() == 'parabolic'

    @pytest.mark.parametrize("s", [-3.0, -2.001, 2.001, 3.0, 100.0])
    def test_hyperbolic(self, s):
        assert RingElement(1.0, 0.0, s).signature_region() == 'hyperbolic'


# --- Mismatched-s rejection: different traces are different rings -------

class TestMismatchedS:
    @pytest.mark.parametrize("op", ['add', 'sub', 'mult'])
    def test_rejects_mismatched_s(self, op):
        x = RingElement(1.0, 0.0, 0.5)
        y = RingElement(1.0, 0.0, 1.5)
        with pytest.raises(ValueError, match="matching s"):
            getattr(x, op)(y)


# --- Operator overloads: sugar over named methods ------------------------

class TestOperatorOverloads:
    def test_add_operator(self):
        s = 0.4
        x = RingElement(1.0, 2.0, s)
        y = RingElement(3.0, -1.0, s)
        assert approx_equal(x + y, RingElement(4.0, 1.0, s))

    def test_sub_operator(self):
        s = 0.4
        x = RingElement(1.0, 2.0, s)
        y = RingElement(3.0, -1.0, s)
        assert approx_equal(x - y, RingElement(-2.0, 3.0, s))

    def test_neg_operator(self):
        s = 0.4
        x = RingElement(1.0, -2.0, s)
        assert approx_equal(-x, RingElement(-1.0, 2.0, s))

    def test_mul_with_scalar_left(self):
        s = 0.4
        x = RingElement(1.5, -2.5, s)
        assert approx_equal(2.0 * x, RingElement(3.0, -5.0, s))

    def test_mul_with_scalar_right(self):
        s = 0.4
        x = RingElement(1.5, -2.5, s)
        assert approx_equal(x * 2.0, RingElement(3.0, -5.0, s))

    def test_mul_with_ring_element(self):
        s = 0.5
        x = RingElement(1.0, 1.0, s)
        y = RingElement(2.0, 0.0, s)
        # (1 + g)*(2) = 2 + 2g
        assert approx_equal(x * y, RingElement(2.0, 2.0, s))
