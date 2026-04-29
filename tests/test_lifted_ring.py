"""Tests for jenk.lifted_ring (stage 2: sympy-valued s, a, b).

The lifted ring needs to satisfy two banks of properties:

  (1) Numeric reduction: at rational/integer s, a, b it agrees with
      jenk.ring once converted back to floats.
  (2) Symbolic closure: at symbolic s (free symbols, omega, polynomials
      in omega), all ring axioms hold and conjugate/norm/inverse have
      the correct algebraic structure.

We also include cardinal probes at s in {0, 1, omega, -1} to verify the
projective mode stays symbolic and the others reduce as expected.
"""

import math

import pytest
import sympy as sp

from jenk.graded import omega
from jenk.lifted_ring import LiftedRingElement
from jenk.ring import RingElement


# Free symbol used as a generic symbolic trace.
t = sp.Symbol('t', real=True)


# --- Helpers --------------------------------------------------------------

def lifted_equal(x: LiftedRingElement, y: LiftedRingElement) -> bool:
    """Equality via component-wise sp.simplify(diff)==0."""
    return (
        sp.simplify(x.a - y.a) == 0
        and sp.simplify(x.b - y.b) == 0
        and sp.simplify(x.s - y.s) == 0
    )


# --- Constructors and input coercion --------------------------------------

class TestConstructors:
    def test_identity_int_s(self):
        e = LiftedRingElement.identity(0)
        assert e.a == sp.Integer(1)
        assert e.b == sp.Integer(0)
        assert e.s == sp.Integer(0)

    def test_identity_omega(self):
        e = LiftedRingElement.identity(omega)
        assert e.a == sp.Integer(1)
        assert e.b == sp.Integer(0)
        assert e.s == omega

    def test_generator_omega(self):
        g = LiftedRingElement.generator(omega)
        assert g.a == sp.Integer(0)
        assert g.b == sp.Integer(1)
        assert g.s == omega

    def test_direct_with_python_ints(self):
        x = LiftedRingElement(2, -3, 5)
        # Should be sympified.
        assert isinstance(x.a, sp.Expr) and x.a == sp.Integer(2)
        assert isinstance(x.b, sp.Expr) and x.b == sp.Integer(-3)
        assert isinstance(x.s, sp.Expr) and x.s == sp.Integer(5)

    def test_direct_with_rationals(self):
        x = LiftedRingElement(sp.Rational(1, 2), sp.Rational(-3, 4), sp.Rational(7, 5))
        assert x.a == sp.Rational(1, 2)
        assert x.b == sp.Rational(-3, 4)
        assert x.s == sp.Rational(7, 5)

    def test_direct_with_symbolic(self):
        x = LiftedRingElement(omega, 1 - omega, omega**2)
        assert x.a == omega
        assert x.b == 1 - omega
        assert x.s == omega ** 2


# --- Defining identity:  g^2 = s*g - 1  -----------------------------------

class TestGeneratorIdentity:
    @pytest.mark.parametrize("s_val", [
        sp.Integer(0), sp.Integer(1), sp.Integer(-1),
        sp.Rational(3, 2), sp.Integer(3),
        omega, 1 - omega, omega ** 2, t,
    ])
    def test_g_squared_equals_sg_minus_1(self, s_val):
        g = LiftedRingElement.generator(s_val)
        gg = g.mult(g)
        # g^2 should be (-1) + s*g  =>  a = -1, b = s.
        assert lifted_equal(gg, LiftedRingElement(-1, s_val, s_val))


# --- Multiplication: identity, formula, ring axioms ----------------------

class TestMultiplication:
    def test_identity_is_left_identity_symbolic(self):
        s_val = omega
        e = LiftedRingElement.identity(s_val)
        x = LiftedRingElement(omega + 1, omega - 2, s_val)
        assert lifted_equal(e.mult(x), x)

    def test_identity_is_right_identity_symbolic(self):
        s_val = t
        e = LiftedRingElement.identity(s_val)
        x = LiftedRingElement(t + 1, 2 * t, s_val)
        assert lifted_equal(x.mult(e), x)

    def test_explicit_formula_symbolic(self):
        s_val = omega
        x = LiftedRingElement(2, 3, s_val)
        y = LiftedRingElement(-1, 4, s_val)
        expected_a = sp.Integer(2 * -1 - 3 * 4)
        expected_b = sp.Integer(2 * 4 + 3 * -1) + 3 * 4 * omega
        assert lifted_equal(x.mult(y), LiftedRingElement(expected_a, expected_b, s_val))

    def test_commutativity_symbolic(self):
        s_val = omega
        x = LiftedRingElement(omega + 1, -omega, s_val)
        y = LiftedRingElement(2, omega - 3, s_val)
        assert lifted_equal(x.mult(y), y.mult(x))

    def test_associativity_symbolic(self):
        s_val = t
        x = LiftedRingElement(sp.Rational(1, 2), 1, s_val)
        y = LiftedRingElement(-t, 1 + t, s_val)
        z = LiftedRingElement(2, -1, s_val)
        assert lifted_equal(x.mult(y).mult(z), x.mult(y.mult(z)))

    def test_distributivity_symbolic(self):
        s_val = omega
        x = LiftedRingElement(1, omega, s_val)
        y = LiftedRingElement(omega - 1, 2, s_val)
        z = LiftedRingElement(sp.Rational(1, 2), -omega, s_val)
        lhs = x.mult(y.add(z))
        rhs = x.mult(y).add(x.mult(z))
        assert lifted_equal(lhs, rhs)


# --- Conjugate involution -------------------------------------------------

class TestConjugate:
    @pytest.mark.parametrize("s_val", [sp.Integer(0), sp.Integer(2), omega, t])
    def test_involution_symbolic(self, s_val):
        x = LiftedRingElement(omega - 1, sp.Rational(2, 3), s_val)
        assert lifted_equal(x.conjugate().conjugate(), x)

    def test_formula_symbolic(self):
        s_val = omega
        x = LiftedRingElement(2, -1, s_val)
        c = x.conjugate()
        # conj(a + b*g) = (a + b*s) - b*g
        assert lifted_equal(c, LiftedRingElement(2 + (-1) * omega, 1, s_val))


# --- Norm: scalar of x*conj(x), symbolic ---------------------------------

class TestNorm:
    @pytest.mark.parametrize("s_val", [omega, t, sp.Integer(0), 1 - omega])
    def test_norm_matches_x_conj_scalar_symbolic(self, s_val):
        x = LiftedRingElement(omega - 2, sp.Rational(3, 5), s_val)
        prod = x.mult(x.conjugate())
        # g-component of x * conj(x) vanishes identically.
        assert sp.simplify(prod.b) == 0
        assert sp.simplify(prod.a - x.norm()) == 0

    def test_norm_of_generator(self):
        # N(0 + 1*g) = 0 + 0*s + 1 = 1, regardless of s.
        for s_val in [sp.Integer(0), omega, t, sp.Integer(5)]:
            g = LiftedRingElement.generator(s_val)
            assert sp.simplify(g.norm() - 1) == 0


# --- Inverse: round-trip; raises on zero norm ----------------------------

class TestInverse:
    @pytest.mark.parametrize("s_val", [
        sp.Integer(0), sp.Integer(1), sp.Rational(-3, 2),
    ])
    def test_inverse_round_trip_numeric(self, s_val):
        x = LiftedRingElement(sp.Rational(7, 10), sp.Rational(-11, 10), s_val)
        ident = x.mult(x.inverse()).simplified()
        assert lifted_equal(ident, LiftedRingElement.identity(s_val))

    def test_inverse_round_trip_symbolic_omega(self):
        s_val = omega
        # pick (a, b) with non-zero symbolic norm; a=1, b=0 -> N=1.
        x = LiftedRingElement(1, 0, s_val)
        ident = x.mult(x.inverse()).simplified()
        assert lifted_equal(ident, LiftedRingElement.identity(s_val))

    def test_inverse_round_trip_symbolic_general(self):
        s_val = omega
        x = LiftedRingElement(2, omega, s_val)
        # N = 4 + 2*omega*omega + omega^2 = 4 + 3*omega^2
        ident = x.mult(x.inverse()).simplified()
        assert lifted_equal(ident, LiftedRingElement.identity(s_val))

    def test_zero_norm_raises_parabolic_light_cone(self):
        # At s=2, a=1, b=-1: N = 1 + 1*(-1)*2 + 1 = 0.
        s_val = sp.Integer(2)
        x = LiftedRingElement(1, -1, s_val)
        assert sp.simplify(x.norm()) == 0
        with pytest.raises(ZeroDivisionError):
            x.inverse()


# --- Numeric reduction: lifted ring agrees with float ring on rationals --

class TestNumericReduction:
    """At rational s, a, b, the lifted ring should match jenk.ring numerically."""

    @pytest.mark.parametrize("s_q", [
        sp.Rational(0), sp.Rational(1, 2), sp.Rational(-3, 2),
        sp.Rational(2), sp.Rational(5, 2), sp.Rational(-3),
    ])
    def test_mult_matches_float(self, s_q):
        a1, b1 = sp.Rational(7, 10), sp.Rational(-1, 5)
        a2, b2 = sp.Rational(3, 4), sp.Rational(11, 10)

        lifted = LiftedRingElement(a1, b1, s_q).mult(LiftedRingElement(a2, b2, s_q))
        floated = RingElement(float(a1), float(b1), float(s_q)).mult(
            RingElement(float(a2), float(b2), float(s_q))
        )
        assert math.isclose(float(lifted.a), floated.a, abs_tol=1e-12)
        assert math.isclose(float(lifted.b), floated.b, abs_tol=1e-12)
        assert math.isclose(float(lifted.s), floated.s, abs_tol=1e-12)

    @pytest.mark.parametrize("s_q", [sp.Rational(1, 2), sp.Rational(-1, 3)])
    def test_inverse_matches_float(self, s_q):
        a, b = sp.Rational(7, 10), sp.Rational(-1, 5)
        lifted_inv = LiftedRingElement(a, b, s_q).inverse()
        float_inv = RingElement(float(a), float(b), float(s_q)).inverse()
        assert math.isclose(float(lifted_inv.a), float_inv.a, abs_tol=1e-12)
        assert math.isclose(float(lifted_inv.b), float_inv.b, abs_tol=1e-12)


# --- Cardinal probes at s in {0, 1, omega, -1} ---------------------------

class TestCardinalProbes:
    """Verify g^2 takes the README cardinal value at each cardinal s.

    For numeric s, g^2 = (-1, s); we just confirm that. For s=omega the
    point is structural: g^2 stays as (-1, omega) without collapsing.
    """

    def test_hyperbolic_s0(self):
        g = LiftedRingElement.generator(sp.Integer(0))
        assert lifted_equal(g.mult(g), LiftedRingElement(-1, 0, 0))

    def test_parabolic_s1(self):
        g = LiftedRingElement.generator(sp.Integer(1))
        assert lifted_equal(g.mult(g), LiftedRingElement(-1, 1, 1))

    def test_elliptic_s_minus_1(self):
        # README: s=-1 is the projective mode (not s=omega; per memo
        # 'Jenk / Traction theory framework', the four cardinals are
        # delta in {+1, 0, -1, omega} mapping to s=0, 1, omega, -1).
        g = LiftedRingElement.generator(sp.Integer(-1))
        assert lifted_equal(g.mult(g), LiftedRingElement(-1, -1, -1))

    def test_projective_s_omega_stays_symbolic(self):
        g = LiftedRingElement.generator(omega)
        gg = g.mult(g)
        # g^2 = -1 + omega*g, i.e., (a, b) = (-1, omega) at s=omega.
        assert lifted_equal(gg, LiftedRingElement(-1, omega, omega))
        # And it does NOT silently collapse to anything else under simplify.
        assert lifted_equal(gg.simplified(),
                            LiftedRingElement(-1, omega, omega))


# --- Mismatched-s rejection ----------------------------------------------

class TestMismatchedS:
    @pytest.mark.parametrize("op", ['add', 'sub', 'mult'])
    def test_rejects_omega_vs_zero(self, op):
        x = LiftedRingElement(1, 0, omega)
        y = LiftedRingElement(1, 0, sp.Integer(0))
        with pytest.raises(ValueError, match="matching s"):
            getattr(x, op)(y)

    @pytest.mark.parametrize("op", ['add', 'sub', 'mult'])
    def test_rejects_distinct_free_symbols(self, op):
        u = sp.Symbol('u')
        v = sp.Symbol('v')
        x = LiftedRingElement(1, 0, u)
        y = LiftedRingElement(1, 0, v)
        with pytest.raises(ValueError, match="matching s"):
            getattr(x, op)(y)

    def test_accepts_equivalent_under_simplify(self):
        # s = (omega + 1) - 1  should equal  omega  under sp.simplify.
        x = LiftedRingElement(1, 0, (omega + 1) - 1)
        y = LiftedRingElement(2, 3, omega)
        # No raise; addition succeeds.
        result = x.add(y)
        assert sp.simplify(result.a - 3) == 0
        assert sp.simplify(result.b - 3) == 0


# --- subs: partial evaluation of symbolic ring elements -----------------

class TestSubs:
    def test_subs_omega_to_numeric(self):
        # Build at s=omega, then substitute omega -> 0.5.
        s_val = omega
        x = LiftedRingElement(omega + 1, omega, s_val)
        x_at_half = x.subs(omega, sp.Rational(1, 2))
        assert x_at_half.a == sp.Rational(3, 2)
        assert x_at_half.b == sp.Rational(1, 2)
        assert x_at_half.s == sp.Rational(1, 2)

    def test_subs_preserves_mult(self):
        # Multiply at s=omega, then sub omega->t : same as multiplying
        # the substituted operands at s=t.
        a = LiftedRingElement(omega, 1, omega)
        b = LiftedRingElement(2, omega - 1, omega)
        product_then_sub = a.mult(b).subs(omega, t)
        sub_then_product = a.subs(omega, t).mult(b.subs(omega, t))
        assert lifted_equal(product_then_sub, sub_then_product)


# --- Signature region: numeric routes; symbolic returns 'symbolic' -------

class TestSignatureRegion:
    @pytest.mark.parametrize("s_val,expected", [
        (sp.Integer(0), 'elliptic'),
        (sp.Rational(3, 2), 'elliptic'),
        (sp.Integer(2), 'parabolic'),
        (sp.Integer(-2), 'parabolic'),
        (sp.Integer(3), 'hyperbolic'),
        (sp.Rational(-5, 2), 'hyperbolic'),
    ])
    def test_numeric_classification(self, s_val, expected):
        x = LiftedRingElement(1, 0, s_val)
        assert x.signature_region() == expected

    def test_symbolic_returns_symbolic(self):
        for s_val in [omega, t, omega + 1, omega ** 2]:
            x = LiftedRingElement(1, 0, s_val)
            assert x.signature_region() == 'symbolic'


# --- Operator overloads --------------------------------------------------

class TestOperatorOverloads:
    def test_add_operator(self):
        s_val = omega
        x = LiftedRingElement(1, 2, s_val)
        y = LiftedRingElement(3, -1, s_val)
        assert lifted_equal(x + y, LiftedRingElement(4, 1, s_val))

    def test_sub_operator(self):
        s_val = omega
        x = LiftedRingElement(1, 2, s_val)
        y = LiftedRingElement(3, -1, s_val)
        assert lifted_equal(x - y, LiftedRingElement(-2, 3, s_val))

    def test_neg_operator(self):
        s_val = t
        x = LiftedRingElement(1, -2, s_val)
        assert lifted_equal(-x, LiftedRingElement(-1, 2, s_val))

    def test_scalar_mul_left_int(self):
        s_val = omega
        x = LiftedRingElement(sp.Rational(3, 2), -sp.Rational(5, 2), s_val)
        assert lifted_equal(2 * x, LiftedRingElement(3, -5, s_val))

    def test_scalar_mul_right_sympy(self):
        s_val = omega
        x = LiftedRingElement(1, 1, s_val)
        # multiply by a sympy scalar (omega itself)
        assert lifted_equal(x * omega, LiftedRingElement(omega, omega, s_val))

    def test_ring_mul_operator(self):
        s_val = omega
        x = LiftedRingElement(1, 1, s_val)
        y = LiftedRingElement(2, 0, s_val)
        # (1 + g) * 2 = 2 + 2*g
        assert lifted_equal(x * y, LiftedRingElement(2, 2, s_val))
