# Polynomial Sweep — Underfit to Overfit

Assignment Sub-task 3 from Advanced Data Technologies (ADT), Session 2.

## What this does

Generates a fresh noisy dataset and fits polynomials of degree 1 through 15.
Splits the data into training (2/3) and validation (1/3) sets first,
fits on training only, and measures MSE on both sets at every degree.
Plots the U curve and three example fits to show the full bias-variance story.

## Files

| File | Description |
|------|-------------|
| `subtask3_polynomial_sweep.py` | Main script — data generation, sweep, U curve plot, written answers |

## Results

| Degree | Train MSE | Val MSE |
|--------|-----------|---------|
| 1 | 1.3473 | 1.2535 |
| 3 | 1.2177 | 0.8466 |
| 4 | 0.8954 | 0.6367 ← minimum |
| 15 | 0.5742 | 24.4269 ← exploded |

- Validation minimum: **degree 4**
- Deploy degree: **degree 3** (just left of minimum)
- Log scale used on y-axis so the U stays readable when high-degree errors explode

## Key findings

**Underfit (degree 1):**
Train MSE and validation MSE are both high and close together.
The straight line cannot bend to follow the curved data — high bias.

**Good fit (degree 3):**
Validation MSE is near its minimum.
The curve tracks the true shape without memorising noise.

**Overfit (degree 15):**
Train MSE keeps falling but validation MSE explodes to 24.4.
The wiggly curve memorises every training point including its noise —
it is the high-variance arm of the U.

## Written answers

**Deploy degree:** 3 — at or just left of the validation minimum.

**Gap with more training data:**
The train-validation gap shrinks with more training data.
More points force the model to explain a broader spread,
so it can no longer swing freely to chase noise.
The U minimum shifts slightly right — more data licenses
a little more complexity.

**Connecting wiggly fit to the U:**
The wiggly degree-15 fit IS the high-variance arm of the U —
it memorises training noise so completely that it lurches to
a different shape on any fresh sample, which is exactly why
its validation MSE climbs back up on the right arm.

## How to run

### Google Colab
1. Upload `subtask3_polynomial_sweep.py`
2. Run:
```python
exec(open('subtask3_polynomial_sweep.py').read())
```

### Local Python
```bash
python subtask3_polynomial_sweep.py
```

## Course
ADT — Session 2: Linear Regression and Regularisation
Instructor: Tim Garbe
SRH University Leipzig
