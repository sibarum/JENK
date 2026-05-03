"""
Custom algebra POC built on sympy.

Generators
----------
Idempotents:        a, b, c            with  x^k = x  for all integers k >= 1
Imaginary root:     i                  with  i^2 = -1
Nilpotent root:     o                  with  o^2 = 0
Sign root:          j                  with  j^2 = 1

Graded carrier
--------------
A family K_n(v) indexed by an integer level n and an inner value v, subject to:

    K_n(0)              = K_{n-1}(1)
    K_n(1)              = K_{n-1}(1/0)
    K_n(0**0)           = K_{n+1}(0)
    K_n((1/0)**(1/0))   = K_{n+1}(1/0)

The four "special" inner values (0, 1, 1/0, 0**0, (1/0)**(1/0)) each collapse onto
the canonical form K_m(1) for some integer m, with the rules above determining m.

Worked out:
    K_n(0)            = K_{n-1}(1)        [rule 1]
    K_n(1)            = K_n(1)            [canonical]
    K_n(1/0)          = K_{n+1}(1)        [rule 2 reversed]
    K_n(0**0)         = K_{n+1}(0) = K_n(1)              [rules 3, then 1]
    K_n((1/0)**(1/0)) = K_{n+1}(1/0) = K_{n+2}(1)        [rules 4, then 2]
"""

from sympy import Function, Integer, S, Symbol


# --------------------------------------------------------------------------- #
# Generators                                                                  #
# --------------------------------------------------------------------------- #

class Idempotent(Symbol):
    """Generator with x^k = x for every positive integer k."""
    is_commutative = True

    def __new__(cls, name):
        return Symbol.__new__(cls, name)

    def _eval_power(self, exp):
        if exp.is_Integer and exp.is_positive:
            return self
        return None


class ImaginaryRoot(Symbol):
    """Generator with x^2 = -1.  Powers cycle 1, x, -1, -x."""
    is_commutative = True

    def __new__(cls, name):
        return Symbol.__new__(cls, name)

    def _eval_power(self, exp):
        if exp.is_Integer:
            n = int(exp) % 4
            return [S.One, self, S.NegativeOne, -self][n]
        return None


class NilpotentRoot(Symbol):
    """Generator with x^2 = 0.  Hence x^k = 0 for k >= 2."""
    is_commutative = True

    def __new__(cls, name):
        return Symbol.__new__(cls, name)

    def _eval_power(self, exp):
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return S.One
            if n == 1:
                return self
            if n >= 2:
                return S.Zero
        return None


class SignRoot(Symbol):
    """Generator with x^2 = 1.  Hence x^k = 1 if k even, x if k odd."""
    is_commutative = True

    def __new__(cls, name):
        return Symbol.__new__(cls, name)

    def _eval_power(self, exp):
        if exp.is_Integer:
            return self if int(exp) % 2 else S.One
        return None


a = Idempotent('a')
b = Idempotent('b')
c = Idempotent('c')

i = ImaginaryRoot('i')
o = NilpotentRoot('o')
j = SignRoot('j')


# --------------------------------------------------------------------------- #
# Graded carrier K_n(value)                                                   #
# --------------------------------------------------------------------------- #

# Symbolic placeholders for inner values that sympy would otherwise refuse to
# represent without folding (1/0 -> zoo, 0**0 -> nan).
INV0 = Symbol('INV0')      # stands for the formal value 1/0
ZPZ = Symbol('ZPZ')        # stands for the formal value 0**0
INVINV = Symbol('INVINV')  # stands for the formal value (1/0)**(1/0)


class K(Function):
    """Graded carrier K_n(v).

    Canonicalises the five special inner values onto K_m(1) using the four
    grading rules.  Other inner values are left untouched, so the carrier is
    usable as an opaque sympy function on arbitrary expressions too.
    """

    @classmethod
    def eval(cls, n, v):
        if not n.is_Integer:
            return None
        m = int(n)

        if v == S.Zero:
            return cls(Integer(m - 1), S.One)
        if v == INV0:
            return cls(Integer(m + 1), S.One)
        if v == ZPZ:
            return cls(Integer(m), S.One)
        if v == INVINV:
            return cls(Integer(m + 2), S.One)
        # v == 1, or any non-special value: already canonical / opaque.
        return None


# --------------------------------------------------------------------------- #
# Demo                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Idempotents:")
    print(f"  a**1 = {a**1},   a**2 = {a**2},   a**5 = {a**5}")
    print(f"  b*b  = {b*b},   c**3 = {c**3}")

    print("\nRoots:")
    print(f"  i**2 = {i**2},   i**3 = {i**3},   i**4 = {i**4},   i**5 = {i**5}")
    print(f"  o**1 = {o**1},   o**2 = {o**2},   o**7 = {o**7}")
    print(f"  j**2 = {j**2},   j**3 = {j**3},   j**100 = {j**100}")

    print("\nMixed expression:")
    expr = (a + i) * (b + o) + j**2 * c
    print(f"  (a + i)(b + o) + j**2 * c = {expr.expand()}")

    print("\nGraded carrier K_n(v):")
    for n in (-1, 0, 1, 2):
        print(f"  K_{n}(0)        = {K(n, 0)}")
        print(f"  K_{n}(1)        = {K(n, 1)}")
        print(f"  K_{n}(INV0)     = {K(n, INV0)}")
        print(f"  K_{n}(ZPZ)      = {K(n, ZPZ)}")
        print(f"  K_{n}(INVINV)   = {K(n, INVINV)}")
        print()

    print("Relation checks (each should be True):")
    print(f"  K_n(0)         == K_{{n-1}}(1):       {K(2, 0)        == K(1, 1)}")
    print(f"  K_n(1)         == K_{{n-1}}(1/0):     {K(2, 1)        == K(1, INV0)}")
    print(f"  K_n(0**0)      == K_{{n+1}}(0):       {K(2, ZPZ)      == K(3, 0)}")
    print(f"  K_n((1/0)^(1/0)) == K_{{n+1}}(1/0):   {K(2, INVINV)   == K(3, INV0)}")
