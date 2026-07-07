# Loss Families Under an Outlier

Assignment Sub-task 5 from Advanced Data Technologies (ADT), Session 2.

## What this does

Builds a small clean dataset on a line, adds one high-leverage outlier,
then fits three regression lines using different loss functions.
Shows which line the outlier pulls toward itself and explains the mechanism.

## Files

| File | Description |
|------|-------------|
| `subtask5_loss_families.py` | Main script — data generation, three model fits, plot, written answers |

## Dataset

| Points | Description |
|--------|-------------|
| 15 clean points | Sit near the true line y = 1 + 1.5x with small noise |
| 1 outlier at (8.5, 1.0) | True y at x=8.5 would be ≈13.75 — residual ≈ 12.75 |

The outlier is placed at the far right end of the x range — a
high-leverage position — so it has maximum power to rotate the slope.

## Results

| Model | Slope | Deviation from true (1.5) |
|-------|-------|--------------------------|
| True line | 1.500 | 0.000 |
| MSE (OLS) | 1.027 | 0.473 ← pulled most! |
| Huber | 1.486 | 0.014 |
| MAE (median) | 1.512 | 0.012 ← pulled least |

## The mechanism

**MSE** squares every residual. The outlier residual is ≈12.75.
Squared: 12.75² ≈ 162 — roughly 1000× larger than a typical clean-point
residual squared (≈0.16). That single term dominates the total loss
and the optimiser rotates the line toward the outlier to reduce it.

**MAE** uses |r|. The outlier contributes |12.75| — the same proportional
weight as any other point. The line barely moves.

**Huber** uses squared loss inside ε=1.5 and linear loss beyond it.
The outlier residual (12.75 >> 1.5) gets only a linear penalty —
its power to pull the line is capped. The slope lands between MSE and MAE.

## Written answers

**Which line was pulled:** MSE (OLS) — slope rotated from 1.5 to 1.027.

**The mechanism:** MSE squares residuals. One large residual becomes
quadratically larger, dominates the total loss, and forces the line toward it.

**When to choose each:**
- MSE: when there are no outliers, or when outliers represent real signal
- MAE: when you want robustness and the outliers are measurement errors
- Huber: the compromise — robust to outliers while smooth everywhere

## How to run

### Google Colab
1. Upload `subtask5_loss_families.py`
2. Run:
```python
exec(open('subtask5_loss_families.py').read())
```

### Local Python
```bash
python subtask5_loss_families.py
```

## Course
ADT — Session 2: Linear Regression and Regularisation
Instructor: Tim Garbe
SRH University Leipzig
