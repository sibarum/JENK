"""Tests for jenk.chebyshev_ring — Traction-aware Chebyshev ring Q[t][g]/(g²−tg+1)."""

from __future__ import annotations

import pytest
import sympy as sp

from old.jenk.chebyshev_ring import ChebyshevRing
from old.jenk.traction import Zero, Omega, traction_simplify


# --- Construction ---------------------------------------------------------

class TestConstruction:

    def test_default_ring_has_free_symbol_t(self):
        R = ChebyshevRing()
        assert R.t == sp.Symbol('t')

    def test_ring_with_provided_symbol(self):
        u = sp.Symbol('u')
        R = ChebyshevRing(t=u)
        assert R.t == u

    def test_ring_with_numeric_t(self):
        R = ChebyshevRing(t=sp.Integer(0))
        assert R.t == sp.Integer(0)

    def test_ring_with_traction_atom_t(self):
        R = ChebyshevRing(t=Omega())
        assert R.t == Omega()

    def test_zero_one_generator(self):
        R = ChebyshevRing()
        assert R.zero() == R.element(0, 0)
        assert R.one() == R.element(1, 0)
        assert R.generator() == R.element(0, 1)


# --- Additive group -------------------------------------------------------

class TestAdditive:

    def test_add_associative(self):
        R = ChebyshevRing()
        x = R.element(1, 2)
        y = R.element(3, 4)
        z = R.element(5, 6)
        assert (x + y) + z == x + (y + z)

    def test_add_commutative(self):
        R = ChebyshevRing()
        x = R.element(1, 2)
        y = R.element(3, 4)
        assert x + y == y + x

    def test_zero_is_additive_identity(self):
        R = ChebyshevRing()
        x = R.element(7, sp.Rational(2, 3))
        assert x + R.zero() == x
        assert R.zero() + x == x

    def test_negate_is_additive_inverse(self):
        R = ChebyshevRing()
        x = R.element(7, sp.Rational(2, 3))
        assert x + (-x) == R.zero()

    def test_scalar_multiplication(self):
        R = ChebyshevRing()
        x = R.element(2, 3)
        assert x.scale(5) == R.element(10, 15)
        assert 5 * x == R.element(10, 15)
        assert x * 5 == R.element(10, 15)


# --- Multiplicative group -------------------------------------------------

class TestMultiplicative:

    def test_one_is_multiplicative_identity(self):
        R = ChebyshevRing()
        x = R.element(sp.Symbol('a'), sp.Symbol('b'))
        assert x * R.one() == x
        assert R.one() * x == x

    def test_generator_squared_is_t_g_minus_one(self):
        """g² = t·g − 1 — the defining relation."""
        R = ChebyshevRing()
        g = R.generator()
        # g² should be (-1, t)  i.e.  -1 + t·g
        assert g * g == R.element(-1, R.t)

    def test_generator_squared_at_t_zero_is_minus_one(self):
        R = ChebyshevRing(t=sp.Integer(0))
        g = R.generator()
        # g² = 0·g − 1 = −1
        assert g * g == R.element(-1, 0)

    def test_generator_squared_at_t_two_is_2g_minus_1(self):
        R = ChebyshevRing(t=sp.Integer(2))
        g = R.generator()
        assert g * g == R.element(-1, 2)

    def test_mult_associative(self):
        R = ChebyshevRing()
        x = R.element(1, 2)
        y = R.element(3, 4)
        z = R.element(5, 6)
        assert (x * y) * z == x * (y * z)

    def test_mult_distributes_over_add(self):
        R = ChebyshevRing()
        x = R.element(1, 2)
        y = R.element(3, 4)
        z = R.element(5, 6)
        assert x * (y + z) == x * y + x * z

    def test_mult_commutative_in_chebyshev_ring(self):
        # The Chebyshev ring is commutative because the minimal polynomial
        # is degree 2 over a commutative base.
        R = ChebyshevRing()
        x = R.element(1, 2)
        y = R.element(3, 4)
        assert x * y == y * x


# --- Conjugation / norm / inverse ----------------------------------------

class TestConjugationNormInverse:

    def test_conjugation_is_involution(self):
        R = ChebyshevRing()
        x = R.element(sp.Symbol('a'), sp.Symbol('b'))
        assert x.conjugate().conjugate() == x

    def test_conjugate_of_g_is_t_minus_g(self):
        R = ChebyshevRing()
        g = R.generator()
        # conj(g) = (0 + 1·t) − 1·g = t − g
        assert g.conjugate() == R.element(R.t, -1)

    def test_norm_of_one_is_one(self):
        R = ChebyshevRing()
        assert R.one().norm() == 1

    def test_norm_of_g_is_one(self):
        """N(g) = 0² + 0·1·t + 1² = 1 — every power of g is a unit."""
        R = ChebyshevRing()
        assert R.generator().norm() == 1

    def test_norm_of_g_to_arbitrary_power_is_one(self):
        R = ChebyshevRing()
        g = R.generator()
        for n in range(1, 6):
            assert sp.simplify(g.power(n).norm() - 1) == 0

    def test_norm_is_multiplicative(self):
        R = ChebyshevRing()
        x = R.element(2, 3)
        y = R.element(5, -1)
        # N(xy) − N(x)·N(y) should simplify to 0
        assert sp.simplify((x * y).norm() - x.norm() * y.norm()) == 0

    def test_inverse_of_g_is_t_minus_g(self):
        """g·(t − g) = t·g − g² = t·g − (t·g − 1) = 1."""
        R = ChebyshevRing()
        g = R.generator()
        g_inv = g.inverse()
        # g·g⁻¹ should be the multiplicative identity
        assert g * g_inv == R.one()
        # And g_inv should equal t − g, i.e. (t, −1)
        assert g_inv == R.element(R.t, -1)

    def test_inverse_of_arbitrary_unit(self):
        R = ChebyshevRing()
        x = R.element(2, 3)
        # x · x⁻¹ = 1 (modulo simplification)
        product = x * x.inverse()
        assert sp.simplify(product.a - 1) == 0
        assert sp.simplify(product.b) == 0

    def test_inverse_zero_norm_raises(self):
        # On the parabolic light cone t=2 with (a, b) = (1, -1):
        # N = 1 + 1·(-1)·2 + 1 = 1 − 2 + 1 = 0
        R = ChebyshevRing(t=sp.Integer(2))
        x = R.element(1, -1)
        assert x.norm() == 0
        with pytest.raises(ZeroDivisionError):
            x.inverse()


# --- Power -----------------------------------------------------------------

class TestPower:

    def test_power_zero_is_one(self):
        R = ChebyshevRing()
        x = R.element(sp.Symbol('a'), sp.Symbol('b'))
        assert x.power(0) == R.one()

    def test_power_one_is_self(self):
        R = ChebyshevRing()
        x = R.element(2, 3)
        assert x.power(1) == x

    def test_power_two_matches_self_mult(self):
        R = ChebyshevRing()
        x = R.element(2, 3)
        assert x.power(2) == x * x

    def test_power_negative_uses_inverse(self):
        R = ChebyshevRing()
        g = R.generator()
        assert g.power(-1) == g.inverse()

    def test_g_to_the_six_at_t_zero_is_minus_one(self):
        """At t=0: g²=-1 so g has order 4. g⁶ = g² = -1."""
        R = ChebyshevRing(t=sp.Integer(0))
        g = R.generator()
        assert g.power(6) == R.element(-1, 0)

    def test_g_to_the_four_at_t_zero_is_one(self):
        R = ChebyshevRing(t=sp.Integer(0))
        g = R.generator()
        assert g.power(4) == R.one()


# --- Substitution / specialization ---------------------------------------

class TestSubstitution:

    def test_subs_t_to_numeric(self):
        R = ChebyshevRing()
        g = R.generator()
        gg = g * g  # = (-1, t)
        # specialize t to 0
        gg_at_zero = gg.subs(R.t, sp.Integer(0))
        assert gg_at_zero == ChebyshevRing(t=sp.Integer(0)).element(-1, 0)

    def test_subs_t_to_omega_atom(self):
        """Specialization to the Traction omega atom keeps results symbolic."""
        R = ChebyshevRing()
        g = R.generator()
        gg = g * g  # = (-1, t)
        gg_at_omega = gg.subs(R.t, Omega())
        # New ring has t = ω; element is (-1, ω)
        assert gg_at_omega.ring.t == Omega()
        assert gg_at_omega.a == -1
        assert gg_at_omega.b == Omega()

    def test_subs_preserves_mult_structure(self):
        # Multiply at free t, then specialize: should match specializing
        # operands then multiplying.
        R = ChebyshevRing()
        x = R.element(R.t, 1)
        y = R.element(2, R.t - 1)
        product_then_sub = (x * y).subs(R.t, sp.Integer(3))
        x_sub = x.subs(R.t, sp.Integer(3))
        y_sub = y.subs(R.t, sp.Integer(3))
        sub_then_product = x_sub * y_sub
        assert product_then_sub == sub_then_product


# --- Mismatched trace rejection -------------------------------------------

class TestMismatchedTrace:

    @pytest.mark.parametrize('op_name', ['add', 'sub', 'mult'])
    def test_rejects_different_t(self, op_name):
        R1 = ChebyshevRing(t=sp.Integer(0))
        R2 = ChebyshevRing(t=sp.Integer(1))
        x = R1.one()
        y = R2.one()
        with pytest.raises(ValueError, match='matching trace'):
            getattr(x, op_name)(y)

    def test_accepts_simplify_equivalent_t(self):
        """t = (a + 1) - 1 should structurally match t = a."""
        a_sym = sp.Symbol('a')
        R1 = ChebyshevRing(t=(a_sym + 1) - 1)
        R2 = ChebyshevRing(t=a_sym)
        x = R1.element(1, 2)
        y = R2.element(3, 4)
        # Should not raise
        result = x + y
        assert result.a == 4
        assert result.b == 6


# --- Traction atoms in coefficients --------------------------------------

class TestTractionAtomsInCoefficients:
    """Verify that Traction identities fire when atoms appear in coefficients
    or in the trace t."""

    def test_zero_times_omega_collapses_to_one(self):
        """0·ω = 1 should fire automatically through arithmetic."""
        R = ChebyshevRing()
        x = R.element(Zero(), 0)
        y = R.element(Omega(), 0)
        # x·y has scalar part = Zero·Omega - 0 = 0·ω, which simplifies to 1
        product = x * y
        assert sp.simplify(product.a - 1) == 0

    def test_zero_to_the_omega_collapses_to_minus_one(self):
        """0^ω = -1 should fire automatically."""
        # Place 0^ω in a coefficient
        z_to_w = Zero() ** Omega()
        R = ChebyshevRing()
        x = R.element(z_to_w, 0)
        # The coefficient should simplify to -1
        assert sp.simplify(x.a + 1) == 0

    def test_g_squared_at_t_omega_stays_symbolic(self):
        """At t=ω, g² = ω·g − 1 stays as (−1, ω) — the omega does not collapse
        because there's nothing for it to multiply against (yet)."""
        R = ChebyshevRing(t=Omega())
        g = R.generator()
        gg = g * g
        assert gg.a == -1
        assert gg.b == Omega()

    def test_norm_of_g_at_t_omega_is_one(self):
        """N(g) = 0² + 0·1·ω + 1² = 1, even with ω in t."""
        R = ChebyshevRing(t=Omega())
        g = R.generator()
        assert g.norm() == 1


# --- Sanity: matches classical Chebyshev evaluation ----------------------

class TestClassicalAgreement:
    """For numeric t, the symbolic ring should agree with naive sympy."""

    def test_g_squared_numerical(self):
        # At t=3, g² = 3·g − 1, so (g²).a = -1, (g²).b = 3.
        R = ChebyshevRing(t=sp.Integer(3))
        g = R.generator()
        gg = g * g
        assert gg.a == -1
        assert gg.b == 3

    def test_g_cubed_numerical(self):
        # g³ = g·g² = g·(t·g − 1) = t·g² − g = t·(t·g − 1) − g = (t² − 1)·g − t
        # At t=3: g³ = 8·g − 3, so (a, b) = (-3, 8).
        R = ChebyshevRing(t=sp.Integer(3))
        g = R.generator()
        ggg = g.power(3)
        assert ggg.a == -3
        assert ggg.b == 8

    def test_chebyshev_recurrence_a_n(self):
        """The half-cycle sums a_n = g^n + g^(-n) satisfy a_n = t·a_(n-1) − a_(n-2),
        with a_0 = 2 and a_1 = t."""
        R = ChebyshevRing()
        g = R.generator()
        g_inv = g.inverse()

        # Compute a_n for n = 0..5
        a = [None] * 6
        for n in range(6):
            term = (g.power(n) + g_inv.power(n))
            a[n] = sp.simplify(term.a)  # g-component should always vanish
            assert sp.simplify(term.b) == 0, f'a_{n} has nonzero g-component'

        # a_0 = 2
        assert a[0] == 2
        # a_1 = t
        assert sp.simplify(a[1] - R.t) == 0
        # Recurrence: a_n = t·a_(n-1) − a_(n-2) for n >= 2
        for n in range(2, 6):
            recurrence = sp.simplify(a[n] - (R.t * a[n - 1] - a[n - 2]))
            assert recurrence == 0, f'recurrence broken at n={n}'


# --- Canonical Traction generators 0^(p/q) -------------------------------

class TestCanonicalTractionGenerators:
    """The four canonical Traction half-power generators 0^(s/2) for s ∈ {0, 1, ω, −1}.

    Each generator is itself symbolic — distinct from any scalar — and
    yields the cardinal δ when squared:
        j  = 0^(0/2)                       j² = 0^0    = +1  (hyperbolic)
        ε  = 0^(1/2)                       ε² = 0^1    =  0  (parabolic)
        η  = 0^(ω/2)                       η² = 0^ω    = −1  (elliptic)
        k  = 0^(−1/2)                      k² = 0^(−1) =  ω  (projective)

    Note: sympy eagerly reduces Rational(0, 2) → 0 and then fires 0^0 → 1,
    which would collapse j to the integer 1 entirely. To preserve j as a
    symbolic half-power generator, we construct its exponent with
    evaluate=False. The other three (ε, η, k) stay symbolic naturally.

    Their natural Chebyshev traces t = g + g⁻¹:
        j: j² = 1 ⇒ j⁻¹ = j ⇒ t = 2j   (projects to t=2 under j → 1)
        ε: ε⁻¹ = k ⇒ t = ε + k = 0^(1/2) + 0^(−1/2)
        η: η⁻¹ = −η ⇒ t = η + (−η) = 0
        k: k⁻¹ = ε ⇒ t = k + ε        (same trace as ε; conjugate roots)

    Three distinct rings: hyperbolic at t=2j (projects to t=2, the classical
    parabolic light cone), elliptic at t=0, and a shared parabolic/projective
    ring at t = ε + k.
    """

    # --- Each generator as a Traction expression -------------------------

    def test_j_form_and_square(self):
        """j = 0^(0/2) is the hyperbolic generator. It is structurally
        distinct from the integer 1 — j ≠ 1 — though its square collapses
        to the cardinal δ_hyp = 1.

        Constructed with evaluate=False to defeat sympy's eager rational
        reduction of 0/2 → 0 (which would then trigger 0^0 → 1)."""
        exp_half = sp.Mul(sp.Integer(0), sp.Rational(1, 2), evaluate=False)
        j = sp.Pow(Zero(), exp_half, evaluate=False)
        assert j != sp.Integer(1)
        assert traction_simplify(j * j) == sp.Integer(1)

    def test_eps_form_and_square(self):
        """0^(1/2) stays symbolic as √0; squared collapses to the Zero atom (= 0 = δ_par)."""
        eps = Zero() ** sp.Rational(1, 2)
        # Symbolic: not yet collapsed to either 0 or to Zero atom
        assert eps != sp.Integer(0)
        # Squared: ε² = 0^1 = Zero atom
        assert traction_simplify(eps * eps) == Zero()

    def test_eta_squared_is_minus_one(self):
        """0^(ω/2) is the elliptic generator η; η² = 0^ω = −1 = δ_ell."""
        eta = Zero() ** (Omega() / 2)
        assert traction_simplify(eta * eta) == sp.Integer(-1)

    def test_k_form_and_square(self):
        """0^(−1/2) stays symbolic as √ω; squared gives ω = δ_proj."""
        k = Zero() ** sp.Rational(-1, 2)
        assert k != Omega()
        assert traction_simplify(k * k) == Omega()

    def test_eps_times_k_is_one(self):
        """ε · k = 0^(1/2) · 0^(−1/2) = 0^0 = 1."""
        eps = Zero() ** sp.Rational(1, 2)
        k = Zero() ** sp.Rational(-1, 2)
        assert traction_simplify(eps * k) == sp.Integer(1)

    # --- Each generator as a ring scalar coefficient ---------------------

    def test_eps_scalar_squared_in_ring_collapses_to_zero_atom(self):
        """Element (ε, 0) squared in any ring: scalar part is ε² = 0 (Zero atom)."""
        R = ChebyshevRing()  # t arbitrary; scalar squaring is t-independent
        eps = Zero() ** sp.Rational(1, 2)
        x = R.element(eps, 0)
        x_sq = x * x
        assert x_sq.a == Zero()
        assert x_sq.b == 0

    def test_k_scalar_squared_in_ring_collapses_to_omega_atom(self):
        """Element (k, 0) squared: scalar part is k² = ω."""
        R = ChebyshevRing()
        k = Zero() ** sp.Rational(-1, 2)
        x = R.element(k, 0)
        x_sq = x * x
        assert x_sq.a == Omega()
        assert x_sq.b == 0

    def test_eta_scalar_squared_in_ring_collapses_to_minus_one(self):
        """Element (η, 0) squared: scalar part is η² = −1."""
        R = ChebyshevRing()
        eta = Zero() ** (Omega() / 2)
        x = R.element(eta, 0)
        x_sq = x * x
        assert x_sq.a == sp.Integer(-1)
        assert x_sq.b == 0

    # --- Each generator's natural Chebyshev trace ------------------------

    def test_hyperbolic_natural_trace(self):
        """j² = 1 ⇒ j is its own multiplicative inverse: j⁻¹ = j.
        So the natural Chebyshev trace is t = j + j⁻¹ = 2j.
        Under j → 1 projection, t → 2 — the parabolic light cone of
        classical Chebyshev. In the ring at t=2: g·g = (−1, 2), which
        equals 1 when g specializes to 1."""
        R = ChebyshevRing(t=sp.Integer(2))
        g = R.generator()
        assert g * g == R.element(-1, 2)

    def test_elliptic_natural_trace(self):
        """η = 0^(ω/2) ⇒ t = η + η⁻¹ = i − i = 0.  In the ring at t=0: g·g = (−1, 0)."""
        R = ChebyshevRing(t=sp.Integer(0))
        g = R.generator()
        assert g * g == R.element(-1, 0)

    def test_parabolic_projective_share_natural_trace(self):
        """ε⁻¹ = k and k⁻¹ = ε, so both ε and k have Chebyshev trace t = ε + k.
        In the ring at this trace, the abstract g satisfies g² = (−1, ε + k)."""
        eps = Zero() ** sp.Rational(1, 2)
        k = Zero() ** sp.Rational(-1, 2)
        t = eps + k
        R = ChebyshevRing(t=t)
        g = R.generator()
        gg = g * g
        # g-component should be exactly t (no expansion needed)
        assert gg.a == sp.Integer(-1)
        assert sp.simplify(gg.b - t) == 0

    # --- Embedding the canonical generators as roots of g² = t·g − 1 -----

    def test_eps_satisfies_chebyshev_relation_at_natural_trace(self):
        """At t = ε + k, substituting g = ε into  t·g − 1  gives ε² = 0 = δ_par.
            t·ε − 1 = (ε + k)·ε − 1 = ε² + k·ε − 1 = 0 + 1 − 1 = 0."""
        eps = Zero() ** sp.Rational(1, 2)
        k = Zero() ** sp.Rational(-1, 2)
        t = eps + k
        relation = sp.expand(t * eps - 1)
        assert traction_simplify(relation) == Zero()

    def test_k_satisfies_chebyshev_relation_at_natural_trace(self):
        """At t = ε + k, substituting g = k into  t·g − 1  gives k² = ω = δ_proj.
            t·k − 1 = (ε + k)·k − 1 = ε·k + k² − 1 = 1 + ω − 1 = ω."""
        eps = Zero() ** sp.Rational(1, 2)
        k = Zero() ** sp.Rational(-1, 2)
        t = eps + k
        relation = sp.expand(t * k - 1)
        assert traction_simplify(relation) == Omega()

    def test_eta_satisfies_chebyshev_relation_at_natural_trace(self):
        """At t = 0, substituting g = η into  t·g − 1  gives η² = −1 = δ_ell.
            0·η − 1 = 0·η − 1 = -1   (since 0·η, although Traction-non-absorbing,
            collapses additively here once embedded in -1 - 0·η)."""
        eta = Zero() ** (Omega() / 2)
        t = sp.Integer(0)
        relation = sp.expand(t * eta - 1)
        assert traction_simplify(relation) == sp.Integer(-1)

    def test_j_satisfies_chebyshev_relation_at_natural_trace(self):
        """At the projected trace t = 2j → 2 (j → 1 projection), substituting
        the projected j = 1 into t·g − 1 yields j² = 1 = δ_hyp.
            2·1 − 1 = 1.
        The hyperbolic case is degenerate (parabolic light cone), so this
        is a numeric consistency check rather than a structural one — the
        unprojected ring at t=2j is symbolic but degenerate."""
        j_projected = sp.Integer(1)  # projection of 0^(0/2)
        t_projected = sp.Integer(2)  # projection of 2j
        relation = sp.expand(t_projected * j_projected - 1)
        assert traction_simplify(relation) == sp.Integer(1)
