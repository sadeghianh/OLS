# ============================================================
# Assignment Sub-task 3: Polynomial Sweep — Underfit to Overfit
# Course: Advanced Data Technologies (ADT) — Session 2
# Works in: Google Colab, Jupyter Notebook, local Python
# ============================================================

# numpy: used for data generation, polynomial fitting and evaluation
import numpy as np

# matplotlib.pyplot: used to plot the U curve and the three fits
import matplotlib.pyplot as plt

# train_test_split: used to split data into training and validation sets
from sklearn.model_selection import train_test_split

# ── Step 1: Generate a fresh noisy dataset ───────────────────

# fix the random seed so results are reproducible every run
np.random.seed(7)

# n_points: total number of data points to generate
n_points = 60

# x: evenly spaced values between 0 and 10
x = np.linspace(0, 10, n_points)

# true_signal: the underlying relationship we want the model to learn
# a smooth sine wave with a linear trend — not too simple, not too complex
true_signal = np.sin(x) + 0.4 * x

# noise: random Gaussian noise added on top of the true signal
# this represents measurement error or irreducible randomness
noise = np.random.randn(n_points) * 0.8

# y: observed target = true signal plus noise
y = true_signal + noise

# ── Step 2: Train / Validation split ─────────────────────────

# split x and y into training (2/3) and validation (1/3)
# shuffle=True: randomise the split so it is not just the first 40 points
# random_state=42: makes the split reproducible
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.33, shuffle=True, random_state=42)

# print split sizes to confirm the ratio is approximately 2:1
print(f"Training points:   {len(x_train)}")
print(f"Validation points: {len(x_val)}")

# ── Step 3: Fit polynomials of degree 1 through 15 ───────────

# degrees: list of polynomial degrees to try
degrees = list(range(1, 16))

# train_mse: will store training MSE for each degree
train_mse = []

# val_mse: will store validation MSE for each degree
val_mse = []

# loop over every degree and fit a polynomial
for deg in degrees:

    # np.polyfit: fits a polynomial of given degree to training data only
    # returns coefficients from highest to lowest power
    coeffs = np.polyfit(x_train, y_train, deg)

    # np.polyval: evaluates the polynomial at given x values
    # compute predictions on the training set
    y_train_pred = np.polyval(coeffs, x_train)

    # compute predictions on the validation set using same coefficients
    y_val_pred = np.polyval(coeffs, x_val)

    # training MSE: mean squared error on the training set
    mse_train = np.mean((y_train - y_train_pred) ** 2)

    # validation MSE: mean squared error on the held-out validation set
    mse_val = np.mean((y_val - y_val_pred) ** 2)

    # append both errors to their lists
    train_mse.append(mse_train)
    val_mse.append(mse_val)

    # print a summary row for this degree
    print(f"  degree={deg:2d}   train MSE={mse_train:.4f}   val MSE={mse_val:.4f}")

# ── Step 4: Find the best degree ─────────────────────────────

# best_idx: index of the degree with the lowest validation MSE
best_idx = int(np.argmin(val_mse))

# best_degree: the actual degree number (index + 1 because degrees start at 1)
best_degree = degrees[best_idx]

# deploy_degree: degree we would actually deploy
# choose at or just left of the U minimum to avoid overfitting
# "just left" means we prefer a slightly simpler model over the exact minimum
deploy_degree = max(1, best_degree - 1)

# print the result
print(f"\n  Validation minimum at degree {best_degree}")
print(f"  Deploy degree chosen: {deploy_degree} (at or just left of minimum)")

# ── Step 5: Fit three example curves for visual comparison ───

# fit degree 1 (underfit: straight line)
coeffs_1  = np.polyfit(x_train, y_train, 1)

# fit the deploy degree (good fit)
coeffs_d  = np.polyfit(x_train, y_train, deploy_degree)

# fit degree 15 (overfit: wiggly)
coeffs_15 = np.polyfit(x_train, y_train, 15)

# x_dense: fine grid for smooth curve plotting
x_dense = np.linspace(x.min(), x.max(), 300)

# evaluate all three fits on the dense grid
y_fit_1  = np.polyval(coeffs_1,  x_dense)
y_fit_d  = np.polyval(coeffs_d,  x_dense)
y_fit_15 = np.polyval(coeffs_15, x_dense)

# also evaluate the true signal on the dense grid for reference
y_true_dense = np.sin(x_dense) + 0.4 * x_dense

# ── Step 6: Plot ──────────────────────────────────────────────

# create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left plot: the U curve ────────────────────────────────────

# ax: alias for the left subplot
ax = axes[0]

# plot training MSE in blue — always falls as degree increases
ax.semilogy(degrees, train_mse, 'b-o', ms=6, lw=2,
            label='Train MSE')

# plot validation MSE in red — forms the U shape
ax.semilogy(degrees, val_mse, 'r-o', ms=6, lw=2,
            label='Validation MSE')

# mark the validation minimum with a vertical dashed line
ax.axvline(best_degree, color='orange', ls='--', lw=2,
           label=f'Val minimum (degree={best_degree})')

# mark the deploy degree with a green vertical line
ax.axvline(deploy_degree, color='green', ls='-', lw=2.5,
           label=f'Deploy here (degree={deploy_degree})')

# shade the underfit region (degrees below deploy)
ax.axvspan(0.5, deploy_degree, alpha=0.08, color='orange',
           label='Underfit region (high bias)')

# shade the overfit region (degrees above best)
ax.axvspan(best_degree, 15.5, alpha=0.08, color='red',
           label='Overfit region (high variance)')

# label the x-axis
ax.set_xlabel('Polynomial degree', fontsize=12)

# label the y-axis — log scale so the U stays readable when high degrees explode
ax.set_ylabel('MSE (log scale)', fontsize=12)

# set subplot title
ax.set_title('U Curve: Train vs Validation MSE\n'
             'Log scale keeps the U visible when high-degree errors explode',
             fontweight='bold', fontsize=11)

# add legend
ax.legend(fontsize=8, loc='upper right')

# add grid for readability
ax.grid(alpha=0.3, which='both')

# set x-axis ticks to show every degree
ax.set_xticks(degrees)

# annotate bias and variance regions
ax.text(1.5, max(val_mse)*0.5, 'HIGH\nBIAS',
        ha='center', fontsize=10, color='darkorange', fontweight='bold')
ax.text(12, max(val_mse)*0.5, 'HIGH\nVARIANCE',
        ha='center', fontsize=10, color='darkred', fontweight='bold')

# ── Right plot: three fits on the same scatter ────────────────

# ax2: alias for the right subplot
ax2 = axes[1]

# plot training points in blue
ax2.scatter(x_train, y_train, c='steelblue', s=30, alpha=0.7,
            zorder=5, label='Train points')

# plot validation points in red
ax2.scatter(x_val, y_val, c='tomato', s=30, alpha=0.7,
            zorder=5, label='Validation points')

# plot the true signal as a black dashed line for reference
ax2.plot(x_dense, y_true_dense, 'k--', lw=1.5, alpha=0.4,
         label='True signal (hidden from model)')

# plot degree 1 fit in orange — flat line, misses all curvature (high bias)
ax2.plot(x_dense, y_fit_1, color='orange', lw=2.5,
         label='Degree 1 — underfit (high bias)')

# plot deploy degree fit in green — tracks the true shape well
ax2.plot(x_dense, y_fit_d, color='green', lw=2.5,
         label=f'Degree {deploy_degree} — good fit (deploy this)')

# clip degree 15 predictions to keep the y-axis readable
y_fit_15_clipped = np.clip(y_fit_15, y.min() - 3, y.max() + 3)

# plot degree 15 fit in red — wiggly, memorises training noise (high variance)
ax2.plot(x_dense, y_fit_15_clipped, color='red', lw=2,
         label='Degree 15 — overfit (high variance)', alpha=0.85)

# label the x-axis
ax2.set_xlabel('x', fontsize=12)

# label the y-axis
ax2.set_ylabel('y', fontsize=12)

# set subplot title
ax2.set_title(f'Three Fits on the Same Data\n'
              f'Orange=underfit, Green=good (degree {deploy_degree}), Red=overfit',
              fontweight='bold', fontsize=11)

# add legend
ax2.legend(fontsize=8, loc='upper left')

# add grid
ax2.grid(alpha=0.2)

# ── Figure-level title ────────────────────────────────────────

# add a main title spanning both subplots
plt.suptitle(
    'Sub-task 3: Polynomial Sweep — Underfit to Overfit\n'
    f'Deploy degree = {deploy_degree}  |  '
    'More training data → gap shrinks → U minimum shifts right',
    fontsize=12, fontweight='bold', color='#1F4E79')

# adjust spacing so subplots do not overlap
plt.tight_layout()

# save the figure — relative path works in Colab and locally
plt.savefig('polynomial_sweep.png', dpi=160, bbox_inches='tight')

# display the plot inline — required for Colab
plt.show()

# ── Step 7: Written answers ───────────────────────────────────

# print the key written answers required by the assignment
print("\n" + "=" * 60)
print("WRITTEN ANSWERS")
print("=" * 60)

# state the deploy degree
print(f"""
  Deploy degree: {deploy_degree}
  (at or just left of the validation minimum at degree {best_degree})
""")

# state how the gap moves with more training data
print("""
  How the gap moves with more training data:
  The train-validation gap SHRINKS with more training data.
  More points force the model to explain a broader spread of
  the data, so it can no longer swing freely to chase noise.
  Variance is reduced, and the U minimum shifts slightly right
  (more data licenses a little more complexity).
""")

# one sentence connecting wiggly fit to high-variance arm
print("""
  Connecting the wiggly fit to the U curve:
  The wiggly degree-15 fit IS the high-variance arm of the U —
  it memorises the training noise so completely that it lurches
  to a different shape on any fresh sample, which is exactly
  why its validation MSE climbs back up on the right arm.
""")
