# ============================================================
# Assignment Sub-task 5: Loss Families Under an Outlier
# Course: Advanced Data Technologies (ADT) — Session 2
# Works in: Google Colab, Jupyter Notebook, local Python
# ============================================================

# numpy: used for dataset generation and array operations
import numpy as np

# matplotlib.pyplot: used to plot the three fitted lines over the data
import matplotlib.pyplot as plt

# LinearRegression: OLS fit — minimises MSE = (1/n)*Σ(y - ŷ)²
from sklearn.linear_model import LinearRegression

# QuantileRegressor: MAE fit — minimises Σ|y - ŷ|, gives median regression
# quantile=0.5 means median, alpha=0 means no extra regularisation penalty
from sklearn.linear_model import QuantileRegressor

# HuberRegressor: Huber fit — quadratic inside ε, linear outside ε
# robust to outliers because the outlier tail only gets a linear penalty
from sklearn.linear_model import HuberRegressor

# ── Step 1: Build a clean dataset on a line ──────────────────

# fix random seed so results are reproducible every run
np.random.seed(3)

# n_clean: number of well-behaved points near the line
n_clean = 15

# x_clean: evenly spread x values from 0 to 8
x_clean = np.linspace(0, 8, n_clean)

# true_slope: the real slope of the underlying line
true_slope = 1.5

# true_intercept: the real intercept of the underlying line
true_intercept = 1.0

# noise: small Gaussian noise so points sit near but not exactly on the line
noise = np.random.randn(n_clean) * 0.4

# y_clean: target values = true line plus small noise
y_clean = true_intercept + true_slope * x_clean + noise

# ── Step 2: Add one clear outlier ────────────────────────────

# outlier_x: placed at the far right end of the x range
# this maximises the lever effect — a point far from the centre
# has more power to rotate the slope up or down
outlier_x = 8.5

# outlier_y: far BELOW the true line
# the true line at x=8.5 gives y = 1 + 1.5*8.5 = 13.75
# setting y=1.0 creates a residual of about 12.75 — very large
outlier_y = 1.0

# combine clean points and the outlier into one dataset
x_all = np.append(x_clean, outlier_x)
y_all = np.append(y_clean, outlier_y)

# total number of points including the outlier
n_all = len(x_all)

# reshape x for sklearn: needs a 2D column vector (n_samples, 1)
X_all = x_all.reshape(-1, 1)

# ── Step 3: Fit three models ──────────────────────────────────

# ── Model 1: MSE (OLS) ───────────────────────────────────────
# Formula: minimise (1/n) * Σ(y - ŷ)²
# Squaring the residual means a large residual dominates the loss
# The outlier at x=8.5, y=1 has residual ≈ 12.75
# Squared: 12.75² ≈ 162 — enormous compared to clean points (≈ 0.16 each)
# The optimiser rotates the line toward the outlier to reduce this one huge term

# create and fit the OLS model
mse_model = LinearRegression()
mse_model.fit(X_all, y_all)

# extract fitted slope and intercept
mse_slope     = mse_model.coef_[0]
mse_intercept = mse_model.intercept_

# ── Model 2: MAE (Median Regression) ─────────────────────────
# Formula: minimise Σ|y - ŷ|
# Absolute error — every residual contributes |r|, not r²
# The outlier contributes |12.75| — same proportional weight as any other point
# The minimiser of summed absolute error is the MEDIAN
# quantile=0.5: asks for the median regression line
# alpha=0: no extra regularisation penalty added on top of MAE
# solver="highs": required in newer sklearn versions to avoid warnings

# create and fit the MAE model
mae_model = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
mae_model.fit(X_all, y_all)

# extract fitted slope and intercept
mae_slope     = mae_model.coef_[0]
mae_intercept = mae_model.intercept_

# ── Model 3: Huber ────────────────────────────────────────────
# Formula:
#   if |r| ≤ ε:  loss = (1/2) * r²         (quadratic — like MSE)
#   if |r| > ε:  loss = ε * (|r| - ε/2)    (linear — like MAE)
# epsilon=1.5: boundary between the two regimes
# Clean points (small residuals) get squared loss — smooth gradients
# The outlier (large residual) gets linear loss — its power is capped
# This is why Huber exists: MAE has a kink at zero; Huber is smooth everywhere

# create and fit the Huber model
huber_model = HuberRegressor(epsilon=1.5)
huber_model.fit(X_all, y_all)

# extract fitted slope and intercept
huber_slope     = huber_model.coef_[0]
huber_intercept = huber_model.intercept_

# print all three fitted lines and their deviations from the true slope
print("=" * 55)
print("FITTED LINES")
print("=" * 55)
print(f"\n  True line:   ŷ = {true_intercept} + {true_slope:.3f}×x")
print(f"\n  MSE (OLS):   ŷ = {mse_intercept:.3f} + {mse_slope:.3f}×x")
print(f"  MAE:         ŷ = {mae_intercept:.3f} + {mae_slope:.3f}×x")
print(f"  Huber:       ŷ = {huber_intercept:.3f} + {huber_slope:.3f}×x")
print(f"\n  Slope deviation from true ({true_slope}):")
print(f"    MSE:   {abs(mse_slope   - true_slope):.3f}  ← pulled most!")
print(f"    Huber: {abs(huber_slope - true_slope):.3f}")
print(f"    MAE:   {abs(mae_slope   - true_slope):.3f}  ← pulled least")

# ── Step 4: Plot ──────────────────────────────────────────────

# x_line: dense range for drawing smooth fitted lines
x_line = np.linspace(x_all.min() - 0.5, x_all.max() + 0.5, 200)

# compute fitted y values for all three models on the dense range
y_mse   = mse_intercept   + mse_slope   * x_line
y_mae   = mae_intercept   + mae_slope   * x_line
y_huber = huber_intercept + huber_slope * x_line

# compute the true line on the dense range for reference
y_true  = true_intercept  + true_slope  * x_line

# create a single figure
fig, ax = plt.subplots(figsize=(10, 6))

# plot clean data points in dark blue
ax.scatter(x_clean, y_clean, c='steelblue', s=70, zorder=5,
           edgecolors='navy', lw=1.2, label='Clean data points')

# plot the outlier as a large red star so it stands out immediately
ax.scatter(outlier_x, outlier_y, c='red', s=350, marker='*',
           zorder=6, edgecolors='darkred', lw=1.5,
           label=f'Outlier — ({outlier_x}, {outlier_y})')

# annotate the outlier with an arrow and label
ax.annotate(f'Outlier ({outlier_x}, {outlier_y})\ntrue y ≈ 13.75',
            xy=(outlier_x, outlier_y),
            xytext=(outlier_x - 3.5, outlier_y + 1.5),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

# plot the true line as a thin black dashed line for reference
ax.plot(x_line, y_true, 'k--', lw=1.5, alpha=0.4,
        label=f'True line  slope={true_slope:.3f}')

# plot the MSE (OLS) line in red — clearly pulled downward by the outlier
ax.plot(x_line, y_mse, 'r-', lw=2.5,
        label=f'MSE / OLS    slope={mse_slope:.3f}  ← pulled down!')

# plot the MAE line in green — stays close to the true line
ax.plot(x_line, y_mae, 'g-', lw=2.5,
        label=f'MAE (median) slope={mae_slope:.3f}  ← robust')

# plot the Huber line in orange — between MSE and MAE
ax.plot(x_line, y_huber, color='orange', lw=2.5,
        label=f'Huber        slope={huber_slope:.3f}  ← compromise')

# add a text box explaining the mechanism
ax.text(0.02, 0.98,
        'Mechanism:\n'
        'MSE squares residuals — the outlier residual\n'
        f'is ≈ 12.75, squared = {12.75**2:.0f}.\n'
        'That single term dominates the total loss\n'
        'and rotates the line toward the outlier.\n\n'
        'MAE uses |r| — the outlier is just one\n'
        'point among many with equal weight.\n'
        'Huber caps the outlier with linear loss.',
        transform=ax.transAxes, va='top', ha='left',
        fontsize=8.5, color='#222',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  alpha=0.92, edgecolor='gray'))

# axis labels
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)

# figure title
ax.set_title('Sub-task 5: Loss Families Under an Outlier\n'
             'MSE rotated by the outlier — MAE and Huber resist it',
             fontweight='bold', fontsize=12, color='#1F4E79')

# legend — moved to avoid overlap with the outlier annotation
ax.legend(fontsize=9, loc='upper left')

# light grid
ax.grid(alpha=0.25)

# adjust layout
plt.tight_layout()

# save figure — relative path works in Colab and locally
plt.savefig('loss_families.png', dpi=160, bbox_inches='tight')

# display inline — required for Colab
plt.show()

# ── Step 5: Written answers ───────────────────────────────────

print("\n" + "=" * 55)
print("WRITTEN ANSWERS")
print("=" * 55)
print(f"""
  Which line was pulled toward the outlier:
  MSE (OLS) was pulled most — slope dropped from the
  true {true_slope} to {mse_slope:.3f}, a deviation of
  {abs(mse_slope-true_slope):.3f}. The line rotated downward
  toward the outlier at ({outlier_x}, {outlier_y}).

  The mechanism:
  MSE squares every residual. The outlier at x={outlier_x}
  sits far from the line (residual ≈ 12.75).
  Squaring gives ≈ 162 — roughly {int(12.75**2 / 0.4**2)}× larger than
  a typical clean-point residual squared (≈ 0.16).
  That single term dominates the total loss and the
  optimiser rotates the line toward the outlier to
  reduce it, at the cost of fitting the clean points worse.

  MAE weights every residual as |r|. The outlier
  contributes |12.75| — the same proportional weight
  as any other point. The line barely moves.

  Huber uses squared loss inside ε=1.5 and linear
  loss beyond it. The outlier residual (12.75 >> 1.5)
  gets only a linear penalty — its power to pull the
  line is capped. The slope lands between MSE and MAE.
""")
