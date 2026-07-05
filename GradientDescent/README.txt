# Gradient Descent — MSE Bowl

Assignment Sub-task 2 from Advanced Data Technologies (ADT), Session 2.

## What this does

Implements gradient descent from scratch for univariate linear regression.
Fits the same four data points as Sub-task 1 using the update rule from lecture:

```
w_new = w_old - learning_rate × gradient
```

Then verifies that gradient descent converges to the same coefficients as OLS.

## Files

| File | Description |
|------|-------------|
| `subtask2_gradient_descent.py` | Main script — GD implementation, loss curve plot, learning rate comparison |

## Results

| Method | β̂₀ | β̂₁ |
|--------|-----|-----|
| OLS (Sub-task 1) | 0.5000 | 1.4000 |
| Gradient Descent (lr=0.02) | 0.5000 | 1.4000 |

Both methods land on the same coefficients ✓

## Learning rate regimes

| Learning rate | Behaviour |
|---------------|-----------|
| lr = 0.02 | Converges — loss falls smoothly to minimum |
| lr = 0.11 | Oscillates — loss bounces, never settles |
| lr = 1.2 (toy) | Diverges — loss grows without bound |

## How to run

### Google Colab
1. Upload `subtask2_gradient_descent.py`
2. Run:
```python
exec(open('subtask2_gradient_descent.py').read())
```

### Local Python
```bash
python subtask2_gradient_descent.py
```

## Key concept

> The learning rate controls how large a step the parameters take at each iteration —
> too small and the model creeps, too large and it oscillates or diverges.

## Course
ADT — Session 2: Linear Regression and Regularisation
Instructor: Tim Garbe
SRH University Leipzig
