"""
traction_portable.py — Standalone Traction algebra evaluator
=============================================================

A self-contained, drop-in Python module for symbolic computation in the
"Traction" algebra used in Constructive Operational Type Theory (COTT).

Only dependency: SymPy (>=1.10).

What this module provides
-------------------------
- Custom SymPy Expr subclasses for the traction atoms:
      Zero    (printed as "0")     Omega    (printed as "w" / ω)
      Null    (printed as "null" / ∅)
- Pre-bound singletons:
      z = Zero()                   w = Omega()                   null = Null()
- Symbolic logarithms that stay unevaluated until they can be resolved:
      Log0, LogW   and convenience callers   log0(x), logw(x)
- An optional graded-algebra element  Z_n(x)  with constructor  Z(n, x).
- A simplifier  traction_simplify(expr)  that applies all the rewrite rules.
- A complex projection  project_complex(expr)  using Chebyshev evaluation.
- Light formatting helpers  format_result / format_complex.

Quick start
-----------
    from traction_portable import z, w, traction_simplify, project_complex

    z**0                       # -> 1
    z**w                       # -> -1   (since 0^ω = -1)
    z**(-1)                    # -> ω
    z * w                      # -> 1
    traction_simplify(2*z*3*w) # -> 6

    # i = 0^(ω/2):
    i_traction = z**(w/2)
    project_complex(i_traction)   # -> exp(i·THETA · π/(2·THETA)) = exp(iπ/2) = i

The algebra in one paragraph
----------------------------
Traction extends ordinary arithmetic so every operation is total (defined
everywhere) and reversible (no information loss). 0 (Zero) and ω (Omega) are
multiplicative reciprocals: 0·ω = 1, 0^(-1) = ω. 0^0 = 1, 0^1 = 0, 0^ω = -1,
and the generalized identities 0^(0^x) = x, 0^(ω^x) = -x hold for any x.
Subtraction a - a does NOT collapse to numeric zero — it produces ∅ (Null),
the erasure element. The complex projection sends 0 → e^(iθ) for a chosen
Chebyshev angle θ, which by default makes 0² = -1 (i.e. 0 plays the role of i).

Configuration
-------------
The Chebyshev angle THETA controls the complex projection:

    >>> import traction_portable as tp
    >>> tp.set_cheb_theta(tp.sp_pi / 2)   # default: θ = π/2 ⇒ 0 ↦ i
    >>> tp.set_cheb_theta(tp.sp_pi / 4)   # θ = π/4 ⇒ 0 ↦ e^(iπ/4)

Whatever angle is chosen, ω in exponents is substituted with π/THETA so
that 0^ω always projects to e^(iπ) = -1.
"""

from sympy import (
    Expr, S, sympify, Pow, Mul, Add, Function,
    Integer, Rational, Symbol, Number, I,
    pi as sp_pi, exp as sp_exp, log as sp_log,
)

__all__ = [
    # Atoms
    'Zero', 'Omega', 'Null', 'Log0', 'LogW', 'GradedElement',
    # Singletons
    'z', 'w', 'null',
    # Constructors
    'zpow', 'wpow', 'Z',
    # Logs / simplification / projection
    'log0', 'logw', 'traction_simplify', 'project_complex',
    'resolve', 'resolve_log',
    # Formatting
    'format_result', 'format_complex',
    # Configuration
    'set_cheb_theta', 'get_cheb_theta', 'sp_pi',
]


# ============================================================
# Configuration: Chebyshev evaluation angle
# ============================================================
# In the Chebyshev projection, 0^z ↦ e^(i·THETA·z).  In *exponents*, ω is
# substituted with π/THETA, so 0^ω always projects to e^(iπ) = -1.
# Default is π/2: 0 ↦ i, and 0² = -1 — the most common choice.
_CHEB_THETA = sp_pi / 2


def set_cheb_theta(theta):
    """Override the global Chebyshev angle used by project_complex()."""
    global _CHEB_THETA
    _CHEB_THETA = sympify(theta)


def get_cheb_theta():
    """Return the current Chebyshev angle (a SymPy expression)."""
    return _CHEB_THETA


def _omega_exp_val():
    """Value substituted for ω when it appears in an exponent: π / THETA."""
    return sp_pi / _CHEB_THETA


# ============================================================
# Core Traction Atoms
# ============================================================

class Zero(Expr):
    """
    Traction zero -- not absorbing under multiplication.
    Has reciprocal omega, satisfying 0 * ω = 1.

    Power rules (handled by _eval_power):
        0^0 = 1     0^1 = 0     0^(-n) = ω^n     0^ω = -1
        0^(0^x) = x         0^(ω^x) = -x
    """
    is_commutative = True
    is_number = False
    is_zero = False
    _op_priority = 15.0

    def __new__(cls):
        return Expr.__new__(cls)

    def _sympystr(self, printer):
        return '0'

    def _latex(self, printer):
        return r'\mathbf{0}'

    def __repr__(self):
        return '0'

    def __str__(self):
        return '0'

    def _eval_power(self, exp):
        if exp == S.Zero or isinstance(exp, Zero):
            return S.One
        if exp == S.One:
            return Zero()
        if isinstance(exp, (Integer, Rational)) and exp.is_negative:
            return Pow(Omega(), -exp)
        if isinstance(exp, Omega):
            return S.NegativeOne
        if isinstance(exp, Pow) and isinstance(exp.base, Zero):
            return exp.exp        # 0^(0^x) = x
        if isinstance(exp, Pow) and isinstance(exp.base, Omega):
            return -exp.exp       # 0^(ω^x) = -x
        return None

    def __mul__(self, other):
        other = sympify(other)
        if isinstance(other, Omega):
            return S.One
        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        other = sympify(other)
        if isinstance(other, Omega):
            return S.One
        return Expr.__rmul__(self, other)

    def __truediv__(self, other):
        other = sympify(other)
        return self * Pow(other, S.NegativeOne)

    def __rtruediv__(self, other):
        other = sympify(other)
        return other * Pow(self, S.NegativeOne)


class Omega(Expr):
    """
    Omega (ω) -- reciprocal of zero. ω = 1/0, and 0 · ω = 1.

    Power rules:
        ω^0 = 1     ω^1 = ω     ω^(-n) = 0^n     ω^ω = -1
        ω^(0^x) = -x        ω^(ω^x) = -1/x
    """
    is_commutative = True
    is_number = False
    is_zero = False
    _op_priority = 15.0

    def __new__(cls):
        return Expr.__new__(cls)

    def _sympystr(self, printer):
        return 'w'

    def _latex(self, printer):
        return r'\omega'

    def __repr__(self):
        return 'w'

    def __str__(self):
        return 'w'

    def _eval_power(self, exp):
        if exp == S.Zero or isinstance(exp, Zero):
            return S.One
        if exp == S.One:
            return Omega()
        if isinstance(exp, (Integer, Rational)) and exp.is_negative:
            return Pow(Zero(), -exp)
        if isinstance(exp, Omega):
            return S.NegativeOne
        if isinstance(exp, Pow) and isinstance(exp.base, Zero):
            return -exp.exp           # ω^(0^x) = -x
        if isinstance(exp, Pow) and isinstance(exp.base, Omega):
            return -S.One / exp.exp   # ω^(ω^x) = -1/x
        return None

    def __mul__(self, other):
        other = sympify(other)
        if isinstance(other, Zero):
            return S.One
        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        other = sympify(other)
        if isinstance(other, Zero):
            return S.One
        return Expr.__rmul__(self, other)

    def __truediv__(self, other):
        other = sympify(other)
        return self * Pow(other, S.NegativeOne)

    def __rtruediv__(self, other):
        other = sympify(other)
        return other * Pow(self, S.NegativeOne)


class Null(Expr):
    """
    Null (∅) -- the erasure element. Result of a - a in traction algebra.
    Distinct from numeric zero (which doesn't exist as an atom in this algebra).
    """
    is_commutative = True
    is_number = False
    is_zero = False
    _op_priority = 15.0

    def __new__(cls):
        return Expr.__new__(cls)

    def _sympystr(self, printer):
        return 'null'

    def _latex(self, printer):
        return r'\emptyset'

    def __repr__(self):
        return 'null'

    def __str__(self):
        return 'null'


class Log0(Function):
    """Unevaluated base-0 logarithm. Stays symbolic until projected."""
    def _sympystr(self, printer):
        return f'log_0({printer.doprint(self.args[0])})'

    def _latex(self, printer):
        return r'\log_{\mathbf{0}}\left(' + printer.doprint(self.args[0]) + r'\right)'


class LogW(Function):
    """Unevaluated base-ω logarithm. Stays symbolic until projected."""
    def _sympystr(self, printer):
        return f'log_w({printer.doprint(self.args[0])})'

    def _latex(self, printer):
        return r'\log_{\omega}\left(' + printer.doprint(self.args[0]) + r'\right)'


# ============================================================
# Optional: Graded Algebra (Z-action)
# ============================================================
# Z_n(x) shifts arithmetic operations between grades.  This block is
# self-contained — delete it if you don't need the graded algebra; nothing
# else in this file depends on GradedElement.

class GradedElement(Expr):
    """
    Z_n(value) -- an element at grade n in the operation hierarchy.

    Same-grade arithmetic shifts the grade by ±1:
        Z_n(a) + Z_n(b) = Z_{n-1}(a·b)        addition  → multiplication
        Z_n(a) · Z_n(b) = Z_{n+1}(a+b)        multiply  → addition
        Z_n(a) - Z_n(b) = Z_{n-1}(a/b)        subtract  → division
        Z_n(a) / Z_n(b) = Z_{n+1}(a-b)        division  → subtraction

    Exponentiation with scalars:
        b ^ Z_n(a) = Z_{n-1}(a·b)
        Z_n(a) ^ b = Z_{n+1}(a·b)

    Fixed points (collapse on construction):
        Z_n(0)  = Z_{n-1}(1)
        Z_n(ω)  = Z_{n-1}(-1)
        Z_n(-1) alternates with Z_{n-1}(ω) and collapses to grade 0 or 1.
        Z_0(x)  = x        (identity grade)
        Z_n(Z_{n±1}(x)) = x
    """
    is_commutative = True
    _op_priority = 16.0

    def __new__(cls, grade, value):
        grade = sympify(grade)
        value = sympify(value)

        if grade == S.Zero:
            return value

        if (isinstance(value, Integer) and value == S.Zero) or isinstance(value, Zero):
            return GradedElement.__new__(cls, grade - 1, S.One)
        if isinstance(value, Omega):
            return GradedElement.__new__(cls, grade - 1, S.NegativeOne)
        if isinstance(value, Integer) and value == S.NegativeOne:
            g = int(grade)
            if g >= 2:
                final_grade = g % 2
                if final_grade == 0:
                    return S.NegativeOne
                return Expr.__new__(cls, S.One, S.NegativeOne)

        if isinstance(value, GradedElement):
            inner_grade = value.args[0]
            inner_value = value.args[1]
            diff = grade - inner_grade
            if diff == S.One or diff == S.NegativeOne:
                return inner_value

        return Expr.__new__(cls, grade, value)

    @property
    def grade(self):
        return self.args[0]

    @property
    def value(self):
        return self.args[1]

    def _sympystr(self, printer):
        return f'Z_{printer.doprint(self.grade)}({printer.doprint(self.value)})'

    def _latex(self, printer):
        return r'Z_{' + printer.doprint(self.grade) + r'}\left(' + printer.doprint(self.value) + r'\right)'

    def __repr__(self):
        return f'Z_{self.grade}({self.value})'

    def __str__(self):
        return f'Z_{self.grade}({self.value})'

    def __add__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade - 1, self.value * other.value)
        return Expr.__add__(self, other)

    def __radd__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade - 1, other.value * self.value)
        return Expr.__radd__(self, other)

    def __sub__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade - 1, self.value / other.value)
        return Expr.__sub__(self, other)

    def __rsub__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade - 1, other.value / self.value)
        return Expr.__rsub__(self, other)

    def __mul__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade + 1, self.value + other.value)
        return Expr.__mul__(self, other)

    def __rmul__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade + 1, other.value + self.value)
        return Expr.__rmul__(self, other)

    def __truediv__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade + 1, self.value - other.value)
        return Expr.__truediv__(self, other)

    def __rtruediv__(self, other):
        other = sympify(other)
        if isinstance(other, GradedElement) and other.grade == self.grade:
            return GradedElement(self.grade + 1, other.value - self.value)
        return Expr.__rtruediv__(self, other)

    def __pow__(self, exp):
        exp = sympify(exp)
        if not isinstance(exp, GradedElement):
            return GradedElement(self.grade + 1, self.value * exp)
        return Expr.__pow__(self, exp)

    def __rpow__(self, base):
        base = sympify(base)
        if not isinstance(base, GradedElement):
            return GradedElement(self.grade - 1, self.value * base)
        return Expr.__rpow__(self, base)

    def __neg__(self):
        return Pow(GradedElement(self.grade + 1, self.value), S.NegativeOne)

    def _eval_power(self, exp):
        if not isinstance(exp, GradedElement):
            return GradedElement(self.grade + 1, self.value * exp)
        return None

    def inverse(self):
        return GradedElement(self.grade + 1, -self.value)


def Z(n, value):
    """Shorthand: Z(n, x) constructs Z_n(x)."""
    return GradedElement(n, value)


# ============================================================
# Singleton aliases
# ============================================================
z = Zero()
w = Omega()
null = Null()


def zpow(n):
    """Shorthand for 0^n."""
    return Pow(Zero(), sympify(n))


def wpow(n):
    """Shorthand for ω^n."""
    return Pow(Omega(), sympify(n))


# ============================================================
# Simplifier
# ============================================================

def traction_simplify(expr):
    """
    Apply Traction-algebra rewrite rules to a SymPy expression.
    Processes bottom-up: simplifies sub-expressions first.
    """
    expr = sympify(expr)

    if isinstance(expr, (Zero, Omega, Null, Number, Symbol)):
        return expr

    if isinstance(expr, Pow):
        base = traction_simplify(expr.base)
        exp = traction_simplify(expr.exp)
        # x^0 = 1 for any x (Zero() exponent may not be caught by SymPy)
        if isinstance(exp, Zero):
            return S.One
        # Universal power-of-power: (v^a)^b -> v^(a·b)
        if isinstance(base, Pow):
            combined_exp = traction_simplify(Mul(base.exp, exp))
            return traction_simplify(Pow(base.base, combined_exp))
        # Exponent distribution: 0^(A+B) -> 0^A · 0^B
        # Only fire when at least one part simplifies beyond a bare zero-power,
        # otherwise _simplify_mul will recombine and we get an infinite loop.
        if isinstance(base, Zero) and isinstance(exp, Add):
            terms = Add.make_args(exp)
            parts = [traction_simplify(Pow(Zero(), t)) for t in terms]
            if any(not (isinstance(p, Pow) and isinstance(p.base, Zero)) for p in parts):
                return traction_simplify(Mul(*parts))
        # Exponent factoring: 0^(c · A) -> (0^A)^c when A contains traction types.
        # Substitute ω -> 0^(-1) first so e.g. 0^(2ω) -> 0^(2·0^(-1))
        # then factor as (0^(0^(-1)))^2 = (-1)^2 = 1.
        if isinstance(base, Zero) and isinstance(exp, Mul):
            exp_sub = exp.subs(Omega(), Pow(Zero(), S.NegativeOne))
            factors = Mul.make_args(exp_sub)
            scalars = []
            traction_parts = []
            for f in factors:
                if isinstance(f, (Integer, Rational)):
                    scalars.append(f)
                else:
                    traction_parts.append(f)
            if scalars and traction_parts:
                scalar = Mul(*scalars)
                inner = Mul(*traction_parts) if len(traction_parts) > 1 else traction_parts[0]
                inner_result = traction_simplify(Pow(Zero(), inner))
                # Skip when inner_result == 1: 1^c = 1 trivially, would discard phase info.
                if (inner_result != S.One
                        and not (isinstance(inner_result, Pow) and isinstance(inner_result.base, Zero))):
                    final = traction_simplify(Pow(inner_result, scalar))
                    # Only apply if final stays in traction (avoid escape to (-1)^(1/7) etc.)
                    if isinstance(final, (Integer, Rational, Zero, Omega, Null)):
                        return final
        return Pow(base, exp)

    if isinstance(expr, Mul):
        return _simplify_mul(expr)

    if isinstance(expr, Add):
        return _simplify_add(expr)

    if hasattr(expr, 'args') and expr.args:
        new_args = [traction_simplify(a) for a in expr.args]
        return expr.func(*new_args)

    return expr


def _simplify_mul(expr):
    """
    Apply the Traction multiplication table.

    Unifies all Zero and Omega factors into a single base-0 power
    using ω^a = 0^(-a), then sums exponents:
        0^a · 0^b  ->  0^(a+b)
        0^a · ω^b  ->  0^(a-b)
        ω^a · ω^b  ->  0^(-a-b)
    """
    raw_args = list(Mul.make_args(expr))
    args = [traction_simplify(a) for a in raw_args]

    flat = []
    for a in args:
        flat.extend(Mul.make_args(a))

    zero_exp = S.Zero
    others = []

    for a in flat:
        if isinstance(a, Zero):
            zero_exp += S.One
        elif isinstance(a, Omega):
            zero_exp += S.NegativeOne
        elif isinstance(a, Pow) and isinstance(a.base, Zero):
            zero_exp += a.exp
        elif isinstance(a, Pow) and isinstance(a.base, Omega):
            zero_exp -= a.exp
        else:
            others.append(a)

    if zero_exp != S.Zero:
        power = Pow(Zero(), traction_simplify(zero_exp))
        others.append(power)

    if not others:
        return S.One
    if len(others) == 1:
        return others[0]
    return Mul(*others)


def _simplify_add(expr):
    """Apply Traction addition rules. Cancellation produces ∅, not numeric zero."""
    args = [traction_simplify(a) for a in expr.args]
    result = Add(*args)

    # a - a = ∅ (SymPy collapses cancellation to S.Zero; reify it as Null)
    if result == S.Zero:
        return Null()

    return result


# ============================================================
# Logarithms
# ============================================================

def log0(expr):
    """
    Base-0 logarithm.

        log_0(1)   = 0         log_0(0)   = 1
        log_0(0^n) = n         log_0(n)   = 0^n     (n > 0)
        log_0(ω^n) = -n        log_0(-n)  = ω^n     (n > 0)

    Returns the unevaluated symbolic form Log0(expr) when not directly resolvable.
    """
    expr = sympify(expr)
    if expr == S.One:
        return S.Zero
    if isinstance(expr, Zero):
        return S.One
    if isinstance(expr, Omega):
        return S.NegativeOne
    if isinstance(expr, Pow) and isinstance(expr.base, Zero):
        return expr.exp
    if isinstance(expr, Pow) and isinstance(expr.base, Omega):
        return -expr.exp
    if isinstance(expr, Integer):
        if expr.is_positive:
            return Pow(Zero(), expr)
        if expr.is_negative:
            return Pow(Omega(), -expr)
    return Log0(expr)


def logw(expr):
    """
    Base-ω logarithm.

        log_w(1)   = 0         log_w(ω)   = 1
        log_w(ω^n) = n         log_w(0^n) = -n
        log_w(-n)  = 0^n       (n > 0)

    Returns the unevaluated symbolic form LogW(expr) when not directly resolvable.
    """
    expr = sympify(expr)
    if expr == S.One:
        return S.Zero
    if isinstance(expr, Omega):
        return S.One
    if isinstance(expr, Zero):
        return S.NegativeOne
    if isinstance(expr, Pow) and isinstance(expr.base, Omega):
        return expr.exp
    if isinstance(expr, Pow) and isinstance(expr.base, Zero):
        return -expr.exp
    if isinstance(expr, Integer) and expr.is_negative:
        return Pow(Zero(), -expr)
    return LogW(expr)


# ============================================================
# Identity-cycle resolution helpers
# ============================================================

def resolve(expr):
    """
    Force simplification by wrapping expr through 0^(0^expr), which is a
    no-op identity (since 0^(0^x) = x).  The intermediate trip through the
    zero-power domain often triggers rules that wouldn't fire on the
    original form.
    """
    expr = traction_simplify(sympify(expr))
    return traction_simplify(Pow(Zero(), Pow(Zero(), expr)))


def resolve_log(expr):
    """Alternative resolution path: log_0(0^expr) = expr."""
    expr = traction_simplify(sympify(expr))
    wrapped = Pow(Zero(), expr)
    return log0(wrapped)


# ============================================================
# Complex Projection (Chebyshev evaluation)
# ============================================================

def project_complex(expr):
    """
    Project a traction expression to the complex numbers.

    Core mapping (Chebyshev evaluation):
        0^z  ↦  e^(i · THETA · z)
    In exponents, ω is substituted with π/THETA, so 0^ω always lands at
    e^(iπ) = -1 regardless of θ.  By default THETA = π/2, which makes
    0² = -1 (so 0 plays the role of i) and 0^(ω/2) = e^(iπ/2)·... = i.

    Use set_cheb_theta() to change the angle.

    Returns a SymPy complex expression; call .evalf() / complex() to get a
    Python complex number.
    """
    expr = traction_simplify(expr)
    return _project(expr)


def _project(expr):
    """Recursively project a traction expression to ℂ."""
    if isinstance(expr, (Integer, Rational)):
        return expr
    if isinstance(expr, Zero):
        return sp_exp(I * _CHEB_THETA)
    if isinstance(expr, Omega):
        return sp_exp(-I * _CHEB_THETA)
    if isinstance(expr, Null):
        return S.Zero
    if isinstance(expr, Symbol):
        return expr

    if isinstance(expr, Log0):
        # 0^z = e^(i·THETA·z) ⇒ z = ln(y) / (i·THETA)
        arg = expr.args[0]
        if arg.has(Omega):
            arg = arg.subs(Omega(), _omega_exp_val())
        proj_arg = _project(arg)
        return sp_log(proj_arg) / (I * _CHEB_THETA)

    if isinstance(expr, LogW):
        arg = expr.args[0]
        if arg.has(Omega):
            arg = arg.subs(Omega(), _omega_exp_val())
        proj_arg = _project(arg)
        return -sp_log(proj_arg) / (I * _CHEB_THETA)

    if isinstance(expr, Pow):
        return _project_pow(expr)

    if isinstance(expr, Mul):
        result = S.One
        for arg in expr.args:
            result = result * _project(arg)
        return result

    if isinstance(expr, Add):
        result = S.Zero
        for arg in expr.args:
            result = result + _project(arg)
        return result

    return expr


def _project_pow(expr):
    """Project a Pow expression."""
    base = expr.base
    exponent = expr.exp

    if not isinstance(base, (Zero, Omega)):
        # Distribute over Mul base if it contains traction atoms.
        if isinstance(base, Mul) and (base.has(Zero) or base.has(Omega)):
            result = S.One
            for factor in Mul.make_args(base):
                result = result * _project(Pow(factor, exponent))
            return result
        pb = _project(base)
        exp_projected = (exponent.subs(Omega(), _omega_exp_val())
                         if exponent.has(Omega) else exponent)
        pe = _project(exp_projected)
        # Negative real bases: (-a)^n = a^n · e^(iπn)
        if isinstance(pb, Number) and pb.is_negative:
            return Pow(-pb, pe) * sp_exp(I * sp_pi * pe)
        return Pow(pb, pe)

    if isinstance(base, Omega):
        exponent = -exponent  # ω^a = 0^(-a)

    exponent = traction_simplify(exponent)

    if exponent == S.Zero or isinstance(exponent, Zero):
        return S.One
    if isinstance(exponent, Omega):
        return S.NegativeOne

    exp_sub = exponent.subs(Omega(), _omega_exp_val()) if exponent.has(Omega) else exponent
    proj_exp = _project(exp_sub)
    return sp_exp(I * _CHEB_THETA * proj_exp)


# ============================================================
# Lightweight formatters (optional)
# ============================================================

def format_result(expr):
    """Pretty-print a traction expression as a compact ASCII/Unicode string."""
    if expr is None:
        return ''
    if isinstance(expr, Zero):
        return '0'
    if isinstance(expr, Omega):
        return 'ω'  # ω
    if isinstance(expr, Null):
        return '∅'  # ∅
    if isinstance(expr, GradedElement):
        return f'Z_{format_result(expr.grade)}({format_result(expr.value)})'
    if isinstance(expr, Symbol):
        return str(expr)
    if isinstance(expr, Log0):
        return f'log₀({format_result(expr.args[0])})'
    if isinstance(expr, LogW):
        return f'logω({format_result(expr.args[0])})'
    if expr == S.NegativeOne:
        return '-1'
    if isinstance(expr, Integer):
        return str(expr)
    if isinstance(expr, Rational):
        return f'{expr.p}/{expr.q}'
    if isinstance(expr, Pow):
        base = format_result(expr.base)
        exponent = format_result(expr.exp)
        if _needs_parens(expr.base):
            base = f'({base})'
        if _needs_parens_exp(expr.exp):
            exponent = f'({exponent})'
        return f'{base}^{exponent}'
    if isinstance(expr, Mul):
        args = list(expr.args)
        if args[0] == S.NegativeOne and len(args) >= 2:
            rest = Mul(*args[1:]) if len(args) > 2 else args[1]
            rest_str = format_result(rest)
            if _needs_parens(rest):
                rest_str = f'({rest_str})'
            return f'-{rest_str}'
        parts = []
        for arg in args:
            s = format_result(arg)
            if isinstance(arg, Add):
                s = f'({s})'
            parts.append(s)
        return '·'.join(parts)  # ·
    if isinstance(expr, Add):
        parts = []
        for i, arg in enumerate(expr.args):
            s = format_result(arg)
            if i > 0 and not s.startswith('-'):
                s = '+' + s
            parts.append(s)
        return ''.join(parts)
    return str(expr)


def _needs_parens(expr):
    return isinstance(expr, (Add, Mul)) or (isinstance(expr, Integer) and expr < 0)


def _needs_parens_exp(expr):
    if isinstance(expr, Integer):
        return False
    return isinstance(expr, (Add, Mul, Rational))


def format_complex(expr):
    """
    Project expr through project_complex() and return a compact string.
    Returns '' if the projection is identical to the input (nothing new to show).
    """
    if expr is None:
        return ''
    try:
        proj = project_complex(expr)
    except Exception:
        return ''

    proj_str = _format_sympy_complex(proj)
    orig_str = format_result(expr)
    if proj_str == orig_str:
        return ''
    return proj_str


def _format_sympy_complex(expr):
    """Compact formatting of a SymPy complex expression."""
    from sympy import im, re

    if expr.has(Zero) or expr.has(Omega) or expr.has(Null):
        return format_result(expr)

    try:
        simplified = expr.simplify()
        if len(str(simplified)) <= len(str(expr)):
            expr = simplified
    except Exception:
        pass

    if expr.free_symbols and len(str(expr)) < 80:
        return (str(expr).replace('I', 'i').replace('**', '^').replace('*', '·'))

    if expr.is_real:
        return _format_real(expr)

    if expr.is_imaginary:
        coeff = expr / I
        if coeff == S.One:
            return 'i'
        if coeff == S.NegativeOne:
            return '-i'
        return f'{_format_real(coeff)}i'

    try:
        r = re(expr).simplify()
        i_part = im(expr).simplify()
        if r != S.Zero and i_part != S.Zero:
            r_str = _format_real(r)
            if i_part == S.One:
                i_str = '+i'
            elif i_part == S.NegativeOne:
                i_str = '-i'
            elif i_part.is_negative:
                i_str = f'{_format_real(i_part)}i'
            else:
                i_str = f'+{_format_real(i_part)}i'
            return f'{r_str}{i_str}'
        if r != S.Zero:
            return _format_real(r)
        if i_part != S.Zero:
            return f'{_format_real(i_part)}i'
    except Exception:
        pass

    return str(expr)


def _format_real(expr):
    if isinstance(expr, Integer):
        return str(int(expr))
    if isinstance(expr, Rational):
        return f'{expr.p}/{expr.q}'
    try:
        val = float(expr)
        if val == int(val):
            return str(int(val))
        return f'{val:.6g}'
    except (TypeError, ValueError):
        return str(expr)


# ============================================================
# Self-check / demo
# ============================================================

if __name__ == '__main__':
    # A tiny smoke test exercising the headline identities.
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    checks = [
        ('0^0',         z**0,                 S.One),
        ('0^1',         z**1,                 z),
        ('0^(-1)',      z**(-1),              w),
        ('0^w',         z**w,                 S.NegativeOne),
        ('w^w',         w**w,                 S.NegativeOne),
        ('0*w',         z * w,                S.One),
        ('w*0',         w * z,                S.One),
        ('0/0',         z / z,                S.One),
        ('0/w',         z / w,                z**2),
        ('1/0',         S.One / z,            w),
        ('2*0*3*w',     traction_simplify(2*z*3*w), Integer(6)),
        ('0^(0^3)',     z**(z**3),            Integer(3)),
        ('0^(w^3)',     z**(w**3),            Integer(-3)),
        ('log_0(0)',    log0(z),              S.One),
        ('log_0(w)',    log0(w),              S.NegativeOne),
        ('log_0(-1)',   log0(S.NegativeOne),  w),
        ('log_w(-3)',   logw(Integer(-3)),    z**3),
    ]

    failures = 0
    for label, got, expected in checks:
        ok = got == expected
        mark = 'OK' if ok else 'FAIL'
        print(f'  [{mark}] {label:18s} = {format_result(got):12s}  (expected {format_result(expected)})')
        if not ok:
            failures += 1

    # Complex projection check at default θ = π/2
    print()
    print('Complex projection (θ = π/2, so 0 ↦ i):')
    for label, expr in [
        ('C(0)',       z),
        ('C(0^2)',     z**2),
        ('C(w)',       w),
        ('C(0^w)',     z**w),
        ('C(0^(w/2))', z**(w/2)),
    ]:
        proj = project_complex(expr).simplify()
        print(f'  {label:14s} = {format_complex(expr) or proj}')

    print()
    if failures:
        print(f'{failures} check(s) failed')
    else:
        print('All identity checks passed.')
