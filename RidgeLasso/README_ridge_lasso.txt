# Ridge versus Lasso

Assignment Sub-task 4 from Advanced Data Technologies (ADT), Session 2.

## What are Ridge and Lasso?

Both are modified versions of linear regression with one difference:
they add a penalty on the size of the coefficients.

Without regularisation, a model can overfit by making coefficients
very large to chase every training point. Ridge and Lasso say:
large coefficients have a cost — keep them small.

**Ridge** shrinks all coefficients toward zero but never reaches zero.
When two features are correlated, Ridge splits the weight between them.

**Lasso** drives some coefficients to exactly zero, removing those
features from the model entirely. It automatically selects the
features that actually matter and drops the rest.

## Dataset

Five features were generated to demonstrate the difference clearly:

| Feature | Role | True coefficient |
|---------|------|-----------------|
| x1 | Signal — genuinely predicts y | 3.0 |
| x2 | Noisy copy of x1 — correlated (r=0.96) | 0.0 |
| x3 | Independent signal | 2.0 |
| x4 | Pure noise | 0.0 |
| x5 | Pure noise | 0.0 |

## Files

| File | Description |
|------|-------------|
| `subtask4_ridge_lasso.py` | Main script — data generation, alpha sweep, coefficient path plots, written answers |

## Results

**Lasso at high penalty:**

| Feature | Coefficient | Status |
|---------|-------------|--------|
| x1 (signal) | > 0 | kept |
| x2 (correlated copy) | 0.000 | zeroed out |
| x3 (signal) | > 0 | kept |
| x4 (noise) | 0.000 | zeroed out |
| x5 (noise) | 0.000 | zeroed out |

Lasso correctly identified the two true signal features and dropped everything else.

**Ridge at high penalty:**

| Feature | Coefficient |
|---------|-------------|
| x1 (signal) | 1.634 |
| x2 (correlated copy) | 0.918 |
| x3 (signal) | 1.804 |
| x4 (noise) | -0.074 |
| x5 (noise) | 0.104 |

Ridge kept all features but shrank them. It split the weight
between x1 and x2 instead of dropping one.

## When to choose each

**Choose Lasso when:**
- You have many features and want to know which ones actually matter
- You want a sparse, interpretable model
- Features are not strongly correlated with each other

**Choose Ridge when:**
- You want to keep all features and shrink them together
- You have correlated features that should both contribute
- Interpretability is less important than prediction stability

## How to run

### Google Colab
1. Upload `subtask4_ridge_lasso.py`
2. Run:
```python
exec(open('subtask4_ridge_lasso.py').read())
```

### Local Python
```bash
python subtask4_ridge_lasso.py
```

## Course
ADT — Session 2: Linear Regression and Regularisation
Instructor: Tim Garbe
SRH University Leipzig
