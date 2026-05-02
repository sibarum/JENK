"""jenk — Computational tools for Jenk / Traction theory.

Submodules:
    traction         Framework atoms (Zero, Omega, Null), built-in
                     identities (0·ω = 1, 0^ω = −1, etc.), traction_simplify,
                     and the Chebyshev complex projection.
    chebyshev_ring   The Chebyshev ring  Q[t][g] / (g² − t·g + 1)  with
                     Traction-aware symbolic algebra. Trace `t` may be a
                     free symbol, a numeric constant, or a Traction atom.
    chebyshev_neuron 2-in / 2-out layer that applies ring multiplication.
                     A layer is parameterized by (t, c, d), realizing the
                     ring weight c + d·g at trace t.
    chebyshev_chain  Left-to-right composition of ChebyshevNeurons.

Naming convention (post-audit):
    t   = Chebyshev trace parameter   (g + g⁻¹ = t,  g² = t·g − 1)
    s   = four-mode framework param   (g(s) = 0^(s/2);  cardinals at
                                       s ∈ {0, 1, ω, −1})
The two are different rings — see tmp/diagnostics/audit_s_vs_t.md.
"""
