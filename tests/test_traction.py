"""Tests for jenk.traction — the Traction algebra primitives.

Mirrors and extends the headline identity checks from traction.py's
__main__ block. Covers Zero/Omega/Null atoms, traction_simplify, log0/logw,
and complex projection.

Z_n / GradedElement is intentionally NOT exercised here per scope decision
(the diagnostic neuron does not use it).
"""

from sympy import S, I, pi, exp, sqrt, Integer, Rational, Symbol, Add

from jenk.traction import (
    z, w, null,
    Zero, Omega, Null,
    log0, logw,
    traction_simplify, project_complex,
    set_cheb_theta, get_cheb_theta,
)


# ---------------------------------------------------------------- atoms

class TestAtoms:
    def test_zero_is_distinct_from_sympy_zero(self):
        assert z != S.Zero

    def test_omega_is_distinct_from_sympy_zero(self):
        assert w != S.Zero

    def test_null_is_distinct_from_sympy_zero(self):
        assert null != S.Zero

    def test_atoms_are_singletons_by_equality(self):
        assert Zero() == z
        assert Omega() == w
        assert Null() == null

    def test_zero_is_not_a_sympy_zero(self):
        assert not z.is_zero
        assert not w.is_zero
        assert not null.is_zero


# ---------------------------------------------------------------- power identities

class TestZeroPowers:
    def test_zero_to_the_zero(self):
        assert z**0 == S.One

    def test_zero_to_the_one(self):
        assert z**1 == z

    def test_zero_to_negative_one(self):
        assert z**(-1) == w

    def test_zero_to_omega(self):
        assert z**w == S.NegativeOne

    def test_zero_to_zero_to_x(self):
        x = Symbol('x')
        assert z**(z**x) == x

    def test_zero_to_omega_to_x(self):
        x = Symbol('x')
        assert z**(w**x) == -x

    def test_zero_cubed(self):
        assert z**3 != S.Zero  # crucial: not collapsing to numeric 0


class TestOmegaPowers:
    def test_omega_to_the_zero(self):
        assert w**0 == S.One

    def test_omega_to_the_one(self):
        assert w**1 == w

    def test_omega_to_negative_one(self):
        assert w**(-1) == z

    def test_omega_to_omega(self):
        assert w**w == S.NegativeOne


# ---------------------------------------------------------------- multiplication

class TestMultiplication:
    def test_zero_times_omega_is_one(self):
        assert z * w == S.One

    def test_omega_times_zero_is_one(self):
        assert w * z == S.One

    def test_zero_div_zero_is_one(self):
        assert z / z == S.One

    def test_one_div_zero_is_omega(self):
        assert S.One / z == w

    def test_zero_div_omega_is_zero_squared(self):
        assert z / w == z**2

    def test_simplify_mixed_factors(self):
        assert traction_simplify(2*z*3*w) == Integer(6)

    def test_simplify_combines_zero_powers(self):
        assert traction_simplify(z**2 * z**3) == z**5

    def test_simplify_omega_as_negative_zero_exponent(self):
        # 0^a * w^b -> 0^(a-b)
        assert traction_simplify(z**3 * w**2) == z


# ---------------------------------------------------------------- subtraction / Null

class TestNullCancellation:
    """Cancellation -> Null rule.

    KNOWN LIMITATION: sympy evaluates `x - x` to `S.Zero` at expression-
    construction time, before traction_simplify can intercept. So the rule
    only fires when the Add is preserved (e.g. via `Add(..., evaluate=False)`)
    or when cancellation appears as a sub-expression that traction_simplify
    walks into. Diagnostic code that wants to surface erasure events should
    keep intermediate Adds unevaluated.
    """

    def test_unevaluated_cancellation_simplifies_to_null(self):
        x = Symbol('x')
        expr = Add(x, -x, evaluate=False)
        assert traction_simplify(expr) == null

    def test_unevaluated_zero_minus_zero_is_null(self):
        expr = Add(z, -z, evaluate=False)
        assert traction_simplify(expr) == null

    def test_evaluated_cancellation_returns_sympy_zero(self):
        # Documents the limitation: by the time x - x reaches traction_simplify,
        # it's already S.Zero. We don't reify because S.Zero might be legitimate.
        x = Symbol('x')
        assert traction_simplify(x - x) == S.Zero

    def test_no_cancellation_no_null(self):
        x = Symbol('x')
        y = Symbol('y')
        result = traction_simplify(x + y)
        assert result != null


# ---------------------------------------------------------------- logarithms

class TestLogarithms:
    def test_log0_of_one(self):
        assert log0(S.One) == S.Zero

    def test_log0_of_zero(self):
        assert log0(z) == S.One

    def test_log0_of_omega(self):
        assert log0(w) == S.NegativeOne

    def test_log0_of_negative_one(self):
        assert log0(S.NegativeOne) == w

    def test_logw_of_negative_three(self):
        assert logw(Integer(-3)) == z**3

    def test_log0_of_zero_power(self):
        assert log0(z**5) == Integer(5)


# ---------------------------------------------------------------- complex projection

class TestComplexProjection:
    def setup_method(self):
        # Always reset to default before each test in this class
        set_cheb_theta(pi / 2)

    def teardown_method(self):
        set_cheb_theta(pi / 2)

    def test_default_theta(self):
        assert get_cheb_theta() == pi / 2

    def test_zero_projects_to_i(self):
        assert project_complex(z) == exp(I * pi / 2)

    def test_zero_squared_projects_to_minus_one(self):
        result = project_complex(z**2).simplify()
        assert result == S.NegativeOne

    def test_omega_projects_to_minus_i(self):
        result = project_complex(w).simplify()
        assert result == -I

    def test_zero_to_omega_projects_to_minus_one(self):
        # Independent of theta: 0^w -> -1 always
        assert project_complex(z**w) == S.NegativeOne

    def test_zero_to_omega_over_two_projects_to_i(self):
        result = project_complex(z**(w/2)).simplify()
        assert result == I

    def test_theta_pi_over_four_changes_zero_projection(self):
        set_cheb_theta(pi / 4)
        result = project_complex(z).simplify()
        # Should be e^(i*pi/4), not e^(i*pi/2)
        assert result == exp(I * pi / 4)
        # 0^w still -> -1 regardless of theta
        assert project_complex(z**w) == S.NegativeOne


# ---------------------------------------------------------------- absorption discipline

class TestNoEagerAnnihilation:
    """Multiplication by Zero must not silently collapse non-trivial results.

    These are the 'operational deferral' guarantees: we never want a
    Zero-times-something to disappear into numeric 0 unless that's the
    framework-correct outcome.
    """

    def test_zero_times_one_is_zero(self):
        # z * 1 stays as z (not collapsed to numeric 0)
        assert z * S.One == z

    def test_zero_times_symbol_stays_symbolic(self):
        x = Symbol('x')
        result = z * x
        # Should not be S.Zero — z is not absorbing
        assert result != S.Zero
        assert result.has(z)

    def test_zero_squared_stays_as_power(self):
        assert z**2 != S.Zero
        assert z**2 != S.One

    def test_omega_squared_stays_as_power(self):
        assert w**2 != S.Zero
