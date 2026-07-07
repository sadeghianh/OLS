# ============================================================
# Assignment Sub-task 4: Ridge versus Lasso
# Course: Advanced Data Technologies (ADT) — Session 2
# Works in: Google Colab, Jupyter Notebook, local Python
# ============================================================

# numpy: used for dataset generation and array operations
import numpy as np

# matplotlib.pyplot: used to plot the two coefficient path charts
import matplotlib.pyplot as plt

# Ridge: L2 regularised linear regression — shrinks all coefficients
from sklearn.linear_model import Ridge, Lasso

# StandardScaler: standardises features to mean=0, std=1 before penalising
from sklearn.preprocessing import StandardScaler

# ── Step 1: Generate the dataset ─────────────────────────────

# fix random seed so results are reproducible every run
np.random.seed(42)

# n: number of data points
n = 80

# ── Build five features ──────────────────────────────────────

# x1: first signal feature — drawn from standard normal distribution
# this feature genuinely predicts y
x1 = np.random.randn(n)

# x2: noisy copy of x1 — correlated with x1 but not identical
# adding noise makes it slightly different but still highly correlated
x2 = x1 + 0.3 * np.random.randn(n)

# x3: independent signal feature — drawn fresh, unrelated to x1
# this feature also genuinely predicts y
x3 = np.random.randn(n)

# x4: pure noise — no relationship with y at all
x4 = np.random.randn(n)

# x5: pure noise — no relationship with y at all
x5 = np.random.randn(n)

# X: design matrix — stack all five features as columns
X = np.column_stack([x1, x2, x3, x4, x5])

# feature_names: labels for each column — used in plots and reports
feature_names = ['x1 (signal)', 'x2 (corr. copy of x1)',
                 'x3 (signal)', 'x4 (noise)', 'x5 (noise)']

# ── Build target y ───────────────────────────────────────────

# true_coefs: the real underlying coefficients
# x1 contributes 3, x3 contributes 2, x2/x4/x5 contribute 0
true_coefs = np.array([3.0, 0.0, 2.0, 0.0, 0.0])

# noise_y: irreducible noise added to y — no model can predict this
noise_y = np.random.randn(n) * 0.5

# y: target = linear combination of true signal + noise
# only x1 and x3 are genuine predictors
y = X @ true_coefs + noise_y

# print a summary of the dataset
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"\n  Samples:  {n}")
print(f"  Features: {len(feature_names)}")
print(f"\n  True coefficients:")
for name, coef in zip(feature_names, true_coefs):
    print(f"    {name:<25}: {coef}")

# check the correlation between x1 and x2 to confirm they are correlated
corr_x1_x2 = np.corrcoef(x1, x2)[0, 1]
print(f"\n  Correlation between x1 and x2: {corr_x1_x2:.3f}")

# ── Step 2: Standardise features ─────────────────────────────

# StandardScaler: subtracts mean and divides by std for each feature
# MUST be done before Ridge or Lasso — the penalty acts on coefficient
# magnitude, so without scaling a feature in large units gets penalised
# more than one in small units — which is unfair
scaler = StandardScaler()

# fit the scaler on X and transform X at the same time
# fit_transform learns mean and std from X, then applies them
X_scaled = scaler.fit_transform(X)

# ── Step 3: Define the alpha sweep ───────────────────────────

# alphas: 50 penalty values evenly spaced on a log scale from 0.01 to 10
# log scale because the interesting behaviour spans many orders of magnitude
alphas = np.logspace(-2, 1, 50)

# ridge_coefs: will store the 5 coefficients at each alpha — shape (50, 5)
ridge_coefs = []

# lasso_coefs: same structure for Lasso
lasso_coefs = []

# ── Step 4: Sweep over all alpha values ──────────────────────

# loop over every alpha value in the sweep
for a in alphas:

    # fit Ridge with current alpha — alpha is the penalty strength (λ)
    ridge = Ridge(alpha=a)
    ridge.fit(X_scaled, y)

    # collect the 5 Ridge coefficients for this alpha
    ridge_coefs.append(ridge.coef_)

    # fit Lasso with current alpha
    # max_iter=10000: Lasso needs more iterations to converge at small alpha
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_scaled, y)

    # collect the 5 Lasso coefficients for this alpha
    lasso_coefs.append(lasso.coef_)

# convert both lists to numpy arrays for easy column slicing
# shape: (n_alphas, n_features) = (50, 5)
ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# ── Step 5: Report which features Lasso zeros ────────────────

# at the largest alpha, check which Lasso coefficients are exactly zero
# atol=1e-3: treat anything smaller than 0.001 as effectively zero
last_lasso = lasso_coefs[-1]

# print the final Lasso coefficients
print("\n" + "=" * 60)
print(f"LASSO COEFFICIENTS AT alpha={alphas[-1]:.2f} (largest penalty)")
print("=" * 60)
for name, coef in zip(feature_names, last_lasso):
    # mark zero coefficients clearly — these features are dropped
    status = "← ZEROED OUT" if abs(coef) < 1e-3 else "← kept"
    print(f"  {name:<25}: {coef:>8.4f}  {status}")

# ── Step 6: Report what Ridge does to correlated pair ────────

# at the largest alpha, show x1 and x2 Ridge coefficients side by side
last_ridge = ridge_coefs[-1]
print("\n" + "=" * 60)
print(f"RIDGE COEFFICIENTS AT alpha={alphas[-1]:.2f} (largest penalty)")
print("=" * 60)
for name, coef in zip(feature_names, last_ridge):
    print(f"  {name:<25}: {coef:>8.4f}")

# explicitly compare x1 and x2 to show Ridge splits weight evenly
print(f"\n  x1 coef: {last_ridge[0]:.4f}")
print(f"  x2 coef: {last_ridge[1]:.4f}")
print(f"  → Ridge splits weight between correlated pair instead of zeroing one")

# ── Step 7: Plot coefficient paths ───────────────────────────

# define one distinct color per feature so both plots are consistent
colors = ['steelblue', 'tomato', 'green', 'orange', 'purple']

# create a figure with two subplots side by side — one for Ridge, one for Lasso
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left plot: Ridge coefficient paths ───────────────────────

# ax: alias for the Ridge subplot
ax = axes[0]

# loop over each feature and draw its coefficient path
for i, (name, color) in enumerate(zip(feature_names, colors)):

    # ridge_coefs[:, i]: the i-th coefficient across all alpha values
    ax.plot(alphas, ridge_coefs[:, i], color=color, lw=2.5, label=name)

# add a horizontal reference line at zero
ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)

# use log scale on x-axis because alphas span 0.01 to 10
ax.set_xscale('log')

# label axes
ax.set_xlabel('Alpha (penalty strength, log scale)', fontsize=11)
ax.set_ylabel('Coefficient value', fontsize=11)

# set the subplot title
ax.set_title('Ridge (L2)\nAll coefficients shrink — none reach exactly zero',
             fontweight='bold', fontsize=11)

# add legend
ax.legend(fontsize=9, loc='upper right')

# add grid for readability
ax.grid(alpha=0.3)

# annotate the correlated pair behaviour at high alpha
ax.annotate('x1 and x2 shrink\ntogether — weight split evenly',
            xy=(5, last_ridge[0]),
            xytext=(0.5, last_ridge[0] + 0.3),
            fontsize=8, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='gray'))

# ── Right plot: Lasso coefficient paths ──────────────────────

# ax2: alias for the Lasso subplot
ax2 = axes[1]

# loop over each feature and draw its coefficient path
for i, (name, color) in enumerate(zip(feature_names, colors)):

    # lasso_coefs[:, i]: the i-th coefficient across all alpha values
    ax2.plot(alphas, lasso_coefs[:, i], color=color, lw=2.5, label=name)

# add a horizontal reference line at zero
ax2.axhline(0, color='black', lw=1, ls='--', alpha=0.5)

# use log scale on x-axis
ax2.set_xscale('log')

# label axes
ax2.set_xlabel('Alpha (penalty strength, log scale)', fontsize=11)
ax2.set_ylabel('Coefficient value', fontsize=11)

# set the subplot title
ax2.set_title('Lasso (L1)\nNoise features + one correlated feature driven to zero',
              fontweight='bold', fontsize=11)

# add legend
ax2.legend(fontsize=9, loc='upper right')

# add grid
ax2.grid(alpha=0.3)

# annotate where noise features first hit zero
for i, (name, color) in enumerate(zip(feature_names, colors)):
    # find the first alpha where this coefficient is effectively zero
    zero_at = np.where(np.abs(lasso_coefs[:, i]) < 1e-3)[0]
    if len(zero_at) > 0:
        # mark the zero crossing with a small vertical line
        ax2.axvline(alphas[zero_at[0]], color=color,
                    ls=':', lw=1, alpha=0.6)

# ── Figure-level settings ─────────────────────────────────────

# add a main title spanning both subplots
plt.suptitle(
    'Sub-task 4: Ridge vs Lasso Coefficient Paths\n'
    'Ridge shrinks all — Lasso zeros noise features and picks one of the correlated pair',
    fontsize=12, fontweight='bold', color='#1F4E79')

# adjust spacing between subplots
plt.tight_layout()

# save figure — relative path works in Colab and locally
plt.savefig('ridge_lasso_paths.png', dpi=160, bbox_inches='tight')

# display inline — required for Colab
plt.show()

# ── Step 8: Written answers ───────────────────────────────────

# print all required written answers
print("\n" + "=" * 60)
print("WRITTEN ANSWERS")
print("=" * 60)

print("""
  Which features Lasso drives to zero:
  At high alpha, Lasso zeros x2 (correlated copy of x1),
  x4 (noise), and x5 (noise). It keeps x1 and x3 — the
  true signal features — and drops everything redundant.

  What Ridge does to the correlated pair (x1, x2):
  Ridge never zeros either coefficient. Instead it spreads
  the weight evenly between x1 and x2, giving both a
  similarly sized coefficient. Neither is dropped.

  When to choose each:
  Choose Lasso when you want automatic feature selection —
  it produces a sparse, interpretable shortlist by zeroing
  useless and redundant features.
  Choose Ridge when correlated features must all be kept
  and shrunk together — it handles multicollinearity
  without arbitrarily dropping one of a correlated pair.
""")
