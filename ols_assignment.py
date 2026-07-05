"""
Assignment Sub-task 1: OLS by hand, verified in code
Data: (1,2), (2,3), (3,5), (4,6)
"""

import numpy as np
from sklearn.linear_model import LinearRegression

print("=" * 55)
print("OLS BY HAND — Two Routes")
print("=" * 55)
print(f"\nData points: (1,2), (2,3), (3,5), (4,6)")

# ── Data ────────────────────────────────────────────────────
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 3, 5, 6], dtype=float)
n = len(x)

# ── ROUTE A: slope = Sxy / Sxx ──────────────────────────────
print("\n" + "-" * 55)
print("ROUTE A — Covariance over Variance (Sums Table)")
print("-" * 55)

x_bar = x.mean()
y_bar = y.mean()
print(f"\n  x̄ = {x_bar}   ȳ = {y_bar}")

print(f"\n  {'i':>3} {'x':>5} {'y':>5} {'x-x̄':>7} {'y-ȳ':>7} "
      f"{'(x-x̄)(y-ȳ)':>12} {'(x-x̄)²':>9}")
print("  " + "-" * 55)

Sxy = 0
Sxx = 0
for i in range(n):
    dx = x[i] - x_bar
    dy = y[i] - y_bar
    prod = dx * dy
    sq   = dx ** 2
    Sxy += prod
    Sxx += sq
    print(f"  {i+1:>3} {x[i]:>5.0f} {y[i]:>5.0f} "
          f"{dx:>7.2f} {dy:>7.2f} {prod:>12.4f} {sq:>9.4f}")

print("  " + "-" * 55)
print(f"  {'Σ':>3} {'':>5} {'':>5} {'':>7} {'':>7} "
      f"{Sxy:>12.4f} {Sxx:>9.4f}")

beta1_A = Sxy / Sxx
beta0_A = y_bar - beta1_A * x_bar
print(f"\n  β̂₁ = Sxy/Sxx = {Sxy}/{Sxx} = {beta1_A:.4f}")
print(f"  β̂₀ = ȳ - β̂₁×x̄ = {y_bar} - {beta1_A:.4f}×{x_bar} = {beta0_A:.4f}")
print(f"\n  Route A → ŷ = {beta0_A:.4f} + {beta1_A:.4f}×x")

# ── ROUTE B: Normal Equation ─────────────────────────────────
print("\n" + "-" * 55)
print("ROUTE B — Normal Equation  β̂ = (XᵀX)⁻¹ Xᵀy")
print("-" * 55)

# Design matrix
X = np.column_stack([np.ones(n), x])
print(f"\n  Design matrix X:\n{X}")

XtX = X.T @ X
Xty = X.T @ y
print(f"\n  XᵀX =\n{XtX}")
print(f"\n  Xᵀy = {Xty}")

XtX_inv = np.linalg.inv(XtX)
print(f"\n  det(XᵀX) = {np.linalg.det(XtX):.4f}")
print(f"\n  (XᵀX)⁻¹ =\n{XtX_inv}")

beta_B = XtX_inv @ Xty
beta0_B = beta_B[0]
beta1_B = beta_B[1]
print(f"\n  β̂ = (XᵀX)⁻¹ Xᵀy = {beta_B}")
print(f"\n  Route B → ŷ = {beta0_B:.4f} + {beta1_B:.4f}×x")

# ── Check both routes match ──────────────────────────────────
print("\n" + "-" * 55)
print("COMPARISON")
print("-" * 55)
print(f"\n  {'':10} {'β̂₀':>10} {'β̂₁':>10}")
print(f"  {'Route A':10} {beta0_A:>10.4f} {beta1_A:>10.4f}")
print(f"  {'Route B':10} {beta0_B:>10.4f} {beta1_B:>10.4f}")
match = np.allclose([beta0_A, beta1_A], [beta0_B, beta1_B], atol=1e-6)
print(f"\n  Both routes match: {match} ✓" if match else "\n  Mismatch!")

# ── Code verification ────────────────────────────────────────
print("\n" + "-" * 55)
print("CODE VERIFICATION")
print("-" * 55)

# numpy.polyfit
coeffs_np = np.polyfit(x, y, 1)
print(f"\n  numpy.polyfit:")
print(f"    β̂₁ = {coeffs_np[0]:.4f}   β̂₀ = {coeffs_np[1]:.4f}")

# sklearn
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
print(f"\n  sklearn LinearRegression:")
print(f"    β̂₁ = {model.coef_[0]:.4f}   β̂₀ = {model.intercept_:.4f}")

# ── Predictions ──────────────────────────────────────────────
print("\n" + "-" * 55)
print("PREDICTIONS vs ACTUALS")
print("-" * 55)
print(f"\n  {'x':>5} {'y actual':>10} {'ŷ predicted':>12} {'residual':>10}")
print("  " + "-" * 40)
for i in range(n):
    y_hat = beta0_B + beta1_B * x[i]
    res   = y[i] - y_hat
    print(f"  {x[i]:>5.0f} {y[i]:>10.0f} {y_hat:>12.4f} {res:>10.4f}")

# ── Interpretation ───────────────────────────────────────────
print("\n" + "-" * 55)
print("SLOPE INTERPRETATION")
print("-" * 55)
print(f"""
  β̂₁ = {beta1_B:.4f}

  One extra unit of x is associated with an expected
  increase of {beta1_B:.4f} units in y, holding all else equal.
""")

print("=" * 55)
print("Done!")
