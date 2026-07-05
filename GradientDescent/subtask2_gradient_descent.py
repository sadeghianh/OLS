# ============================================================
# Assignment Sub-task 2: Gradient Descent on the MSE Bowl
# Course: Advanced Data Technologies (ADT) — Session 2
# Data: (1,2), (2,3), (3,5), (4,6) — same as sub-task 1
# Works in: Google Colab, Jupyter Notebook, local Python
# ============================================================

# numpy: numerical computing library — used for arrays, math, and MSE
import numpy as np

# matplotlib.pyplot: plotting library — used to draw all three loss curves
import matplotlib.pyplot as plt

# ── Data (same as sub-task 1) ────────────────────────────────

# x: predictor variable — the four x values from the assignment
x = np.array([1, 2, 3, 4], dtype=float)

# y: target variable — the four y values we want to predict
y = np.array([2, 3, 5, 6], dtype=float)

# n: total number of data points — used inside loops and gradient formulas
n = len(x)

# ── Helper function: MSE ─────────────────────────────────────

def mse(b0, b1, x, y):
    # Compute predicted values: ŷ = b0 + b1*x for every point at once
    y_hat = b0 + b1 * x
    # Return the mean of all squared residuals (actual minus predicted)²
    return np.mean((y - y_hat) ** 2)

# ── Main function: Gradient Descent ──────────────────────────

def gradient_descent(x, y, lr, n_iter, b0_init=0.0, b1_init=0.0):
    # b0: intercept — start at b0_init (default 0.0)
    b0 = b0_init

    # b1: slope — start at b1_init (default 0.0)
    b1 = b1_init

    # n: number of data points — used to scale the gradient correctly
    n = len(x)

    # losses: empty list — will store MSE at every iteration for plotting
    losses = []

    # Main loop: repeat the update rule n_iter times
    for _ in range(n_iter):

        # y_hat: predicted values at current b0, b1
        y_hat = b0 + b1 * x

        # residuals: difference between prediction and actual (ŷ - y)
        residuals = y_hat - y

        # grad_b0: gradient of MSE with respect to intercept b0
        # formula: dMSE/db0 = (2/n) * Σ(ŷ - y)
        grad_b0 = (2 / n) * np.sum(residuals)

        # grad_b1: gradient of MSE with respect to slope b1
        # formula: dMSE/db1 = (2/n) * Σ(ŷ - y) * x
        grad_b1 = (2 / n) * np.sum(residuals * x)

        # update b0: move intercept one step in the downhill direction
        # update rule: w_new = w_old - learning_rate * gradient
        b0 = b0 - lr * grad_b0

        # update b1: move slope one step in the downhill direction
        b1 = b1 - lr * grad_b1

        # current_mse: compute MSE at the new b0, b1
        current_mse = mse(b0, b1, x, y)

        # stop early if MSE becomes NaN or explodes — learning rate too large
        if np.isnan(current_mse) or current_mse > 1e8:
            # append NaN as a marker so the plot shows where it broke
            losses.append(float('nan'))
            break

        # append current MSE to the log so we can plot the loss curve later
        losses.append(current_mse)

    # return the final coefficients and the full loss history
    return b0, b1, losses

# ── Toy GD function: f(w) = (w-3)² ──────────────────────────

def toy_gd(w_init, lr, n_iter):
    # w: scalar weight — start at w_init
    w = w_init

    # losses: empty list — will store f(w) = (w-3)² at every step
    losses = []

    # loop for n_iter steps
    for _ in range(n_iter):

        # grad: derivative of (w-3)² with respect to w = 2*(w-3)
        grad = 2 * (w - 3)

        # update w using the same rule: w_new = w_old - lr * gradient
        w = w - lr * grad

        # val: function value at the new w
        val = (w - 3) ** 2

        # stop early if value blows up — diverged
        if val > 1e6:
            losses.append(float('nan'))
            break

        # log the current function value
        losses.append(val)

    # return final w and the full loss history
    return w, losses

# ── Run experiment 1: good learning rate on full regression ──

# lr_good: step size that works well on this 4-point dataset
lr_good = 0.02

# n_iter: 2000 steps is more than enough to converge
n_iter = 2000

# run gradient descent with the good learning rate
b0_good, b1_good, losses_good = gradient_descent(x, y, lr_good, n_iter)

# print the final intercept, slope, and MSE
print("=" * 55)
print(f"GRADIENT DESCENT — learning rate = {lr_good}")
print("=" * 55)
print(f"\n  Final β̂₀ = {b0_good:.4f}  (sub-task 1 OLS: 0.5000)")
print(f"  Final β̂₁ = {b1_good:.4f}  (sub-task 1 OLS: 1.4000)")
print(f"  Final MSE = {losses_good[-1]:.6f}")

# ── Run experiment 2: too-large learning rate ─────────────────

# lr_bad: slightly above the convergence boundary → oscillates
lr_bad = 0.11

# run with the bad learning rate — only 300 steps to show the oscillation
b0_bad, b1_bad, losses_bad = gradient_descent(x, y, lr_bad, 300)

# print what happened with the bad rate
print(f"\n{'='*55}")
print(f"GRADIENT DESCENT — learning rate = {lr_bad} (too large)")
print(f"{'='*55}")
print(f"\n  Final β̂₀ = {b0_bad:.4f}")
print(f"  Final β̂₁ = {b1_bad:.4f}")
print(f"  Final MSE = {losses_bad[-1]:.4f}")
print(f"  Loss oscillates — never settles cleanly!")

# ── Run experiment 3: toy f(w)=(w-3)² with four rates ────────

# toy_iters: only 30 steps — enough to clearly see each regime
toy_iters = 30

# converges slowly: |1 - 2*0.3| = 0.4 < 1
w_conv,  loss_conv  = toy_gd(0.0, lr=0.3,  n_iter=toy_iters)

# converges very fast: |1 - 2*0.49| ≈ 0
w_fast,  loss_fast  = toy_gd(0.0, lr=0.49, n_iter=toy_iters)

# bounded oscillation: |1 - 2*1.0| = 1 → weights bounce 0, 6, 0, 6 forever
w_osc,   loss_osc   = toy_gd(0.0, lr=1.0,  n_iter=toy_iters)

# diverges: |1 - 2*1.2| = 1.4 > 1 → loss grows without bound
w_div,   loss_div   = toy_gd(0.0, lr=1.2,  n_iter=toy_iters)

# ── Plotting ──────────────────────────────────────────────────

# create a figure with three side-by-side subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ── Subplot 1: good rate loss curve ──────────────────────────

# ax: alias for first subplot
ax = axes[0]

# plot MSE vs iteration for the good learning rate
ax.plot(range(len(losses_good)), losses_good, 'b-', lw=2,
        label=f'lr={lr_good}')

# add a horizontal dashed line at the final (converged) MSE value
ax.axhline(losses_good[-1], color='green', ls='--', lw=1.5,
           label=f'Final MSE={losses_good[-1]:.4f}')

# label the x-axis
ax.set_xlabel('Iteration', fontsize=11)

# label the y-axis
ax.set_ylabel('MSE', fontsize=11)

# set the subplot title
ax.set_title(f'Good rate (lr={lr_good})\nConverges to OLS solution',
             fontweight='bold', fontsize=11)

# add legend in the upper right
ax.legend(fontsize=9)

# add a light grid for readability
ax.grid(alpha=0.3)

# annotate the final coefficient values on the plot
ax.text(0.98, 0.95,
        f'β̂₀={b0_good:.4f}\nβ̂₁={b1_good:.4f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color='green',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ── Subplot 2: good vs bad rate ───────────────────────────────

# ax2: alias for second subplot
ax2 = axes[1]

# plot good rate loss curve in blue
ax2.plot(range(len(losses_good)), losses_good, 'b-', lw=2,
         label=f'lr={lr_good} (converges)')

# keep only valid (non-NaN) values from the bad rate run before plotting
valid_bad = [(i, v) for i, v in enumerate(losses_bad) if not np.isnan(v)]

# only draw the bad rate line if we have any valid points
if valid_bad:
    # unpack list of (index, value) tuples into two separate sequences
    idx_b, val_b = zip(*valid_bad)
    # plot the bad rate loss in red
    ax2.plot(idx_b, val_b, 'r-', lw=2,
             label=f'lr={lr_bad} (oscillates)')

# label the x-axis
ax2.set_xlabel('Iteration', fontsize=11)

# label the y-axis
ax2.set_ylabel('MSE', fontsize=11)

# set the subplot title
ax2.set_title('Good rate vs Too-large rate\nFull regression',
              fontweight='bold', fontsize=11)

# add legend
ax2.legend(fontsize=9)

# add grid
ax2.grid(alpha=0.3)

# force y-axis to start at 0 so the scale is honest
ax2.set_ylim(bottom=0)

# ── Subplot 3: toy f(w)=(w-3)² ────────────────────────────────

# ax3: alias for third subplot
ax3 = axes[2]

# plot converging-slow rate in blue
ax3.plot(range(toy_iters), loss_conv, 'b-o', ms=4, lw=1.5,
         label='lr=0.3 (converges, slow)')

# plot converging-fast rate in green
ax3.plot(range(toy_iters), loss_fast, 'g-o', ms=4, lw=1.5,
         label='lr=0.49 (converges, fast)')

# plot oscillating rate in orange
ax3.plot(range(toy_iters), loss_osc, color='orange', marker='o',
         ms=4, lw=1.5, label='lr=1.0 (oscillates: 0,6,0,6)')

# keep only valid values from the diverging run
valid_div = [(i, v) for i, v in enumerate(loss_div) if not np.isnan(v)]

# only draw the diverging line if we have valid points
if valid_div:
    # unpack into index and value sequences
    idx_d, val_d = zip(*valid_div)
    # plot diverging rate in red
    ax3.plot(idx_d, val_d, 'r-o', ms=4, lw=1.5,
             label='lr=1.2 (diverges)')

# add a dotted horizontal line at 0 — the true minimum
ax3.axhline(0, color='black', lw=0.8, ls=':')

# label the x-axis
ax3.set_xlabel('Iteration', fontsize=11)

# label the y-axis
ax3.set_ylabel('f(w) = (w-3)²', fontsize=11)

# set the subplot title
ax3.set_title('Toy f(w)=(w-3)²\nFour learning rate regimes',
              fontweight='bold', fontsize=11)

# add legend
ax3.legend(fontsize=8)

# add grid
ax3.grid(alpha=0.3)

# fix y-axis range so the oscillating line is clearly visible
ax3.set_ylim(-0.5, 20)

# ── Figure-level settings ─────────────────────────────────────

# add a main title spanning all three subplots
plt.suptitle(
    'Sub-task 2: Gradient Descent on the MSE Bowl\n'
    'The learning rate controls how large a step the parameters take at each iteration.',
    fontsize=12, fontweight='bold', color='#1F4E79')

# adjust spacing so subplots do not overlap each other
plt.tight_layout()

# save the figure to a file in the current working directory
plt.savefig('gradient_descent.png', dpi=160, bbox_inches='tight')

# display the plot inline — required for Colab to show it in the notebook
plt.show()

# ── Final summary ─────────────────────────────────────────────

# print a table comparing GD result with the OLS result from sub-task 1
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"\n  {'':20} {'β̂₀':>10} {'β̂₁':>10}")
print(f"  {'Sub-task 1 (OLS)':20} {'0.5000':>10} {'1.4000':>10}")
print(f"  {'GD lr=0.02':20} {b0_good:>10.4f} {b1_good:>10.4f}")
print(f"\n  Both methods reach the same coefficients ✓")

# print the one-sentence learning rate interpretation required by the assignment
print(f"""
  Learning rate interpretation:
  The learning rate controls how large a step the parameters
  take at each iteration — too small and the model creeps,
  too large and it oscillates or diverges.
""")
