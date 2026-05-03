"""Visualize op targets vs architecture's polynomial-degree-2 reach.

Three rows × four columns:
  Row 1: target function  f(a, b) = a OP b      over (a, b) ∈ [-5, 5]²
  Row 2: best degree-2 polynomial fit            (architecture expressivity ceiling)
  Row 3: |residual| = |target − best_fit|        (where the architecture fails)

Cols: +, −, *, /

The polynomial fit is the closed-form L²-best degree-2 polynomial in (a, b)
on the grid — equivalent to what (input · input) · bias can express at one
readout slot.  Division is fit on the band |b| > 0.5 (excluding the strip
near the singularity).  The residual visualises where the polynomial
function class can't reach the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_grid(extent: float = 5.0, n: int = 200):
    a = np.linspace(-extent, extent, n)
    b = np.linspace(-extent, extent, n)
    A, B = np.meshgrid(a, b)
    return A, B


def fit_degree2(A: np.ndarray, B: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None):
    """Fit  c0 + c1·a + c2·b + c3·a² + c4·a·b + c5·b²  via least squares."""
    A_flat = A.flatten()
    B_flat = B.flatten()
    target_flat = target.flatten()
    if mask is not None:
        m = mask.flatten()
        A_flat, B_flat, target_flat = A_flat[m], B_flat[m], target_flat[m]
    X = np.column_stack([
        np.ones_like(A_flat), A_flat, B_flat,
        A_flat ** 2, A_flat * B_flat, B_flat ** 2,
    ])
    coefs, *_ = np.linalg.lstsq(X, target_flat, rcond=None)

    def evaluate(a, b):
        return (coefs[0] + coefs[1]*a + coefs[2]*b
                + coefs[3]*a**2 + coefs[4]*a*b + coefs[5]*b**2)

    return evaluate, coefs


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    A, B = make_grid(extent=5.0, n=200)

    # Op definitions: (name, target_func, fit_mask)
    eps_band = 0.5
    ops = [
        ('a + b', A + B, None),
        ('a − b', A - B, None),
        ('a · b', A * B, None),
        ('a / b',
         np.divide(A, B, out=np.full_like(A, np.nan), where=(np.abs(B) > 1e-9)),
         np.abs(B) > eps_band),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    row_titles = [
        'Target function  f(a, b)',
        'Best degree-2 polynomial fit (architecture ceiling)',
        '|residual| = |target − fit|',
    ]

    for col, (name, target, mask) in enumerate(ops):
        if mask is not None:
            target_for_fit = target.copy()
        else:
            target_for_fit = target
        fit, coefs = fit_degree2(A, B, target_for_fit, mask)
        prediction = fit(A, B)
        residual = np.abs(target - prediction)

        # Row 1: target
        target_clip = np.clip(target, -50, 50)
        vmax_t = np.nanmax(np.abs(target_clip))
        im = axes[0, col].imshow(
            target_clip, extent=[-5, 5, -5, 5], origin='lower',
            cmap='RdBu_r', vmin=-vmax_t, vmax=vmax_t, aspect='auto',
        )
        axes[0, col].set_title(name, fontsize=14)
        axes[0, col].set_xlabel('a'); axes[0, col].set_ylabel('b')
        plt.colorbar(im, ax=axes[0, col], shrink=0.8)

        # Row 2: polynomial fit
        pred_clip = np.clip(prediction, -50, 50)
        vmax_p = max(np.nanmax(np.abs(pred_clip)), 1e-6)
        im = axes[1, col].imshow(
            pred_clip, extent=[-5, 5, -5, 5], origin='lower',
            cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='auto',
        )
        coef_str = (f"const={coefs[0]:+.2f} a={coefs[1]:+.2f} b={coefs[2]:+.2f}\n"
                    f"a²={coefs[3]:+.2f} ab={coefs[4]:+.2f} b²={coefs[5]:+.2f}")
        axes[1, col].set_title(coef_str, fontsize=9)
        axes[1, col].set_xlabel('a'); axes[1, col].set_ylabel('b')
        plt.colorbar(im, ax=axes[1, col], shrink=0.8)

        # Row 3: |residual|
        residual_clip = np.clip(residual, 0, 20)
        max_resid = np.nanmax(residual)
        im = axes[2, col].imshow(
            residual_clip, extent=[-5, 5, -5, 5], origin='lower',
            cmap='hot', vmin=0, aspect='auto',
        )
        axes[2, col].set_title(f'max |resid| ≈ {max_resid:.2f}', fontsize=10)
        axes[2, col].set_xlabel('a'); axes[2, col].set_ylabel('b')
        plt.colorbar(im, ax=axes[2, col], shrink=0.8)

    for r, title in enumerate(row_titles):
        axes[r, 0].text(
            -0.4, 0.5, title, transform=axes[r, 0].transAxes,
            rotation=90, va='center', ha='center', fontsize=11, fontweight='bold',
        )

    fig.suptitle(
        'Op function classes vs degree-2 polynomial reach\n'
        '(architecture: (input · input) · bias  ≡  degree-2 polynomial in (a, b))',
        fontsize=14,
    )
    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.96])

    out_path = Path(__file__).parent / 'op_function_classes.png'
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
