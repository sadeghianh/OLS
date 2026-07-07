# ============================================================
# Assignment Sub-task 2: Gradient Descent on the MSE Bowl
# Course: Advanced Data Technologies (ADT) — Session 2
# Data: (1,2), (2,3), (3,5), (4,6) — same as sub-task 1
# Works in: Google Colab, Jupyter Notebook, local Python
# ============================================================

# numpy: numerical computing — used for arrays and gradient math
import numpy as np

# matplotlib.pyplot: plotting library — used for all three subplots
import matplotlib.pyplot as plt

# ── Data ─────────────────────────────────────────────────────

# x: predictor variable — four x values from the assignment
x = np.array([1, 2, 3, 4], dtype=float)

# y: target variable — four y values we want to predict
y = np.array([2, 3, 5, 6], dtype=float)

# n: number of data points
n = len(x)

# ── Helper: MSE ──────────────────────────────────────────────

def mse(b0, b1, x, y):
    # y_hat: predicted values at current b0 and b1
    y_hat = b0 + b1 * x
    # return mean of squared residuals
    return np.mean((y - y_hat) ** 2)

# ── Gradient Descent on full regression ──────────────────────

def gradient_descent(x, y, lr, n_iter, b0_init=0.0, b1_init=0.0):
    # initialise intercept at b0_init
    b0 = b0_init
    # initialise slope at b1_init
    b1 = b1_init
    # n: number of data points
    n = len(x)
    # losses: list to store MSE at every iteration
    losses = []

    # repeat the update rule n_iter times
    for _ in range(n_iter):
        # compute current predictions
        y_hat = b0 + b1 * x
        # compute residuals: predicted minus actual
        residuals = y_hat - y
        # gradient of MSE w.r.t. intercept: (2/n)*Σ(ŷ-y)
        grad_b0 = (2 / n) * np.sum(residuals)
        # gradient of MSE w.r.t. slope: (2/n)*Σ(ŷ-y)*x
        grad_b1 = (2 / n) * np.sum(residuals * x)
        # update intercept: step downhill
        b0 = b0 - lr * grad_b0
        # update slope: step downhill
        b1 = b1 - lr * grad_b1
        # compute MSE at updated parameters
        current_mse = mse(b0, b1, x, y)
        # stop early if diverged (NaN or explosion)
        if np.isnan(current_mse) or current_mse > 1e8:
            losses.append(float('nan'))
            break
        # log the current MSE
        losses.append(current_mse)

    # return final coefficients and full loss history
    return b0, b1, losses

# ── Toy GD: f(w) = (w-3)² ────────────────────────────────────
# This is a simplified 1D function used ONLY to demonstrate
# the three regimes: converge / oscillate / diverge.
# It is NOT the regression formula — it is a teaching tool.
# The true minimum is at w=3 where f(3)=0.
# Gradient = df/dw = 2*(w-3)
# Update:   w_new = w_old - lr * 2*(w-3) = (1-2*lr)*w + 6*lr
# Per-step multiplier = |1-2*lr| — this decides the regime:
#   |1-2*lr| < 1  →  converges
#   |1-2*lr| = 1  →  oscillates (lr=1.0: w bounces 0,6,0,6)
#   |1-2*lr| > 1  →  diverges

def toy_gd(w_init, lr, n_iter):
    # initialise weight at w_init
    w = w_init
    # w_history: store the w VALUE at each step (not f(w))
    # showing w makes the oscillation between 0 and 6 clearly visible
    w_history = []

    # loop for n_iter steps
    for _ in range(n_iter):
        # gradient of (w-3)² = 2*(w-3)
        grad = 2 * (w - 3)
        # update: w_new = w_old - lr * gradient
        w = w - lr * grad
        # stop if w explodes (diverged)
        if abs(w) > 1e4:
            w_history.append(float('nan'))
            break
        # log the current w value
        w_history.append(w)

    # return final w and trajectory
    return w, w_history

# ── Run regression experiments ────────────────────────────────

# experiment A: very slow rate — shows creeping behaviour
lr_slow  = 0.001
b0_slow,  b1_slow,  losses_slow  = gradient_descent(x, y, lr_slow,  2000)

# experiment B: good rate — converges cleanly
lr_good  = 0.02
b0_good,  b1_good,  losses_good  = gradient_descent(x, y, lr_good,  2000)

# experiment C: too-large rate — oscillates or diverges
lr_bad   = 0.11
b0_bad,   b1_bad,   losses_bad   = gradient_descent(x, y, lr_bad,   300)

# print results for the good rate
print("=" * 55)
print(f"Good rate (lr={lr_good})")
print("=" * 55)
print(f"  β̂₀ = {b0_good:.4f}  (OLS: 0.5000)")
print(f"  β̂₁ = {b1_good:.4f}  (OLS: 1.4000)")
print(f"  MSE = {losses_good[-1]:.6f}")

# ── Run toy experiments ───────────────────────────────────────

# toy_iters: 30 steps — enough to clearly see each regime
toy_iters = 30

# converges slowly: |1-2*0.3| = 0.4
w_conv, wh_conv = toy_gd(0.0, lr=0.3,  n_iter=toy_iters)

# converges fast: |1-2*0.49| ≈ 0.02
w_fast, wh_fast = toy_gd(0.0, lr=0.49, n_iter=toy_iters)

# bounded oscillation: |1-2*1.0| = 1.0 — w bounces 0, 6, 0, 6 exactly
w_osc,  wh_osc  = toy_gd(0.0, lr=1.0,  n_iter=toy_iters)

# diverges: |1-2*1.2| = 1.4 > 1 — w grows without bound
w_div,  wh_div  = toy_gd(0.0, lr=1.2,  n_iter=toy_iters)

# ── Plotting ──────────────────────────────────────────────────

# create three subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Plot 1: three learning rates on regression ────────────────
ax = axes[0]

# plot slow rate in gray — stays high for a long time (creeping)
ax.plot(range(len(losses_slow)), losses_slow, color='gray', lw=2,
        label=f'lr={lr_slow} (creeping — too slow)', alpha=0.8)

# plot good rate in blue — drops steeply then flattens
ax.plot(range(len(losses_good)), losses_good, 'b-', lw=2.5,
        label=f'lr={lr_good} (good — converges)')

# collect only valid non-NaN values from bad rate
valid_bad = [(i, v) for i, v in enumerate(losses_bad) if not np.isnan(v)]
if valid_bad:
    idx_b, val_b = zip(*valid_bad)
    # plot bad rate in red — shows spike then instability
    ax.plot(idx_b, val_b, 'r-', lw=2,
            label=f'lr={lr_bad} (too large — oscillates)')

# mark the converged MSE value with a horizontal dashed line
ax.axhline(losses_good[-1], color='green', ls='--', lw=1.5,
           label=f'OLS MSE = {losses_good[-1]:.4f}')

# axis labels and title
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('Three Learning Rates\nSlow / Good / Too-large',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# annotate final coefficients
ax.text(0.98, 0.95,
        f'β̂₀={b0_good:.4f}\nβ̂₁={b1_good:.4f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color='blue',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# zoom in on first 500 iterations so the differences are visible
ax.set_xlim(0, 500)

# ── Plot 2: zoom on convergence tail ─────────────────────────
ax2 = axes[1]

# plot slow rate — still falling slowly at iteration 500-2000
ax2.plot(range(len(losses_slow)), losses_slow, color='gray', lw=2,
         label=f'lr={lr_slow} (still creeping)', alpha=0.8)

# plot good rate — already flat (converged)
ax2.plot(range(len(losses_good)), losses_good, 'b-', lw=2.5,
         label=f'lr={lr_good} (converged)')

# horizontal reference line at OLS MSE
ax2.axhline(losses_good[-1], color='green', ls='--', lw=1.5,
            label=f'OLS minimum = {losses_good[-1]:.4f}')

# axis labels and title
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('MSE', fontsize=11)
ax2.set_title('Zoom: Slow vs Good rate\nSlow is still falling at iteration 2000!',
              fontweight='bold', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# set y limit to show the gap between slow and good clearly
ax2.set_ylim(0, 1.0)

# ── Plot 3: toy f(w)=(w-3)² — showing w trajectory ───────────
ax3 = axes[2]

# add horizontal reference line at w=3 (the true minimum)
ax3.axhline(3, color='black', lw=1.5, ls='--', label='True minimum w=3')

# plot converging-slow trajectory in blue
ax3.plot(range(len(wh_conv)), wh_conv, 'b-o', ms=5, lw=2,
         label='lr=0.3 (converges slowly toward 3)')

# plot converging-fast trajectory in green
ax3.plot(range(len(wh_fast)), wh_fast, 'g-o', ms=5, lw=2,
         label='lr=0.49 (reaches 3 in ~2 steps)')

# plot oscillating trajectory in orange — bounces between 0 and 6
ax3.plot(range(len(wh_osc)), wh_osc, color='orange', marker='o',
         ms=5, lw=2, label='lr=1.0 (oscillates: 0 → 6 → 0 → 6)')

# collect valid values from diverging trajectory
valid_div = [(i, v) for i, v in enumerate(wh_div) if not np.isnan(v)]
if valid_div:
    idx_d, val_d = zip(*valid_div)
    # plot diverging trajectory in red — shoots away from 3
    ax3.plot(idx_d, val_d, 'r-o', ms=5, lw=2,
             label='lr=1.2 (diverges — w flies away)')

# axis labels and title
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('w (weight value)', fontsize=11)
ax3.set_title('Toy f(w)=(w-3)²: w trajectory\nOscillation now clearly visible!',
              fontweight='bold', fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# fix y range so oscillation between 0 and 6 is clearly visible
ax3.set_ylim(-2, 10)

# add annotation explaining the oscillation
ax3.annotate('bounces between\n0 and 6 forever!',
             xy=(5, 0), xytext=(12, 2),
             fontsize=9, color='darkorange', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='darkorange'))

# ── Figure-level settings ─────────────────────────────────────

# add overall title
plt.suptitle(
    'Sub-task 2: Gradient Descent on the MSE Bowl\n'
    'The learning rate controls the step size — too small: creeps, '
    'too large: oscillates or diverges.',
    fontsize=11, fontweight='bold', color='#1F4E79')

# adjust spacing
plt.tight_layout()

# save the figure — path is relative so it works in Colab too
plt.savefig('gradient_descent.png', dpi=160, bbox_inches='tight')

# display inline — required for Colab
plt.show()

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"\n  {'':20} {'β̂₀':>10} {'β̂₁':>10}")
print(f"  {'Sub-task 1 (OLS)':20} {'0.5000':>10} {'1.4000':>10}")
print(f"  {'GD lr=0.02':20} {b0_good:>10.4f} {b1_good:>10.4f}")
print(f"\n  Both methods reach the same coefficients ✓")
print(f"""
  Learning rate interpretation:
  The learning rate controls how large a step the parameters
  take at each iteration — too small and the model creeps,
  too large and it oscillates or diverges.
""")
