# Sub-Plan: Random Forest & Ensemble Methods

## Status: Not Started

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Averages/votes of many decision trees (each on bootstrapped data + random feature subsets) |
| **Loss function** | Same as individual trees (Gini/Entropy/Variance), but applied per-tree |
| **Optimization** | Train each tree independently (parallelizable), then aggregate |
| **Bias-variance management** | Bagging reduces **variance** (bias stays ~same as single tree). Random features de-correlate trees -> further variance reduction: $\text{Var} = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$ |
| **Key trick** | Average many high-variance, low-bias models to get a low-variance, low-bias ensemble |

**Historical era**: Era 3 — Ensemble Methods (Breiman, 2001). The insight that many weak models > one strong model.

**VC dimension**: Effectively much lower than a single deep tree. Averaging reduces the effective complexity, even though each tree has infinite $d_{VC}$. This is why RF is robust to overfitting — more trees almost never hurts.

**What problem from 03 does this solve?** Single decision trees have very high variance (unstable). Random Forest fixes this by averaging many de-correlated trees.

**What problem does it create?** Less interpretable than a single tree. Cannot extrapolate beyond training data range. Still weaker than boosting on most tabular tasks. -> Motivates Gradient Boosting (09) which reduces bias sequentially.

**New math introduced**: Bootstrap theory, ensemble variance formula, OOB estimation as free cross-validation.

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

#### Ensemble Theory
- **Bias-Variance decomposition for ensembles**:
  - Single model: $\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Noise}$
  - Average of $M$ models with pairwise correlation $\rho$:
    $\text{Var}_{ensemble} = \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2$
  - Key insight: reducing $\rho$ (correlation between models) reduces ensemble variance
- **Condorcet's Jury Theorem** (intuition): If each voter is >50% accurate and independent, majority vote improves with more voters

#### Bagging (Bootstrap Aggregating)
- **Bootstrap sampling**: Draw $n$ samples with replacement from dataset of size $n$
  - Each bootstrap sample contains ~63.2% unique training points ($1 - (1-1/n)^n \to 1 - 1/e$)
  - Remaining ~36.8% = Out-of-Bag (OOB) samples -> free validation set
- **Aggregation**: Average (regression) or majority vote (classification)
- **Variance reduction**: reduces variance while keeping bias roughly the same

#### Random Forest
- **Random Forest = Bagging + Random Feature Subsets**
  - At each split, consider only $m$ random features out of $p$ total
  - Typical: $m = \sqrt{p}$ (classification), $m = p/3$ (regression)
  - This de-correlates trees -> reduces $\rho$ -> reduces variance further
- **Feature Importance**:
  - **Mean Decrease in Impurity (MDI)**: Sum of impurity decrease at each split weighted by samples reaching node
  - **Permutation Importance**: Shuffle feature $j$, measure accuracy drop
  - MDI has bias toward high-cardinality features; permutation is more reliable
- **OOB Error**: Predict each sample using only trees that didn't include it in bootstrap -> no need for separate validation set

#### Other Ensemble Methods (overview)
- **Voting**: Hard voting (majority) vs Soft voting (average probabilities)
- **Stacking**: Train a meta-learner on base model predictions
- **Boosting**: Covered in depth in 09_Gradient_Boosting

### 1.2 Statistical Foundation

- **Law of Large Numbers**: As $M \to \infty$, ensemble converges
- **Bootstrap theory**: Efron (1979) — non-parametric estimation of sampling distribution
- **OOB estimate**: Unbiased estimate of generalization error (Breiman, 1996)
- **Variable importance testing**: Conditional vs marginal importance, corrected importance for correlated features

### 1.3 Historical Context

- Bagging: Breiman (1996)
- Random Forest: Breiman (2001) — one of the most influential ML papers
- Dominated Kaggle competitions before gradient boosting took over
- Still widely used due to simplicity and robustness

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Much lower variance than single trees | Less interpretable than single tree |
| Robust to overfitting (more trees rarely hurts) | Slower training/inference than single tree |
| Built-in OOB validation | Cannot extrapolate beyond training range |
| Handles high-dimensional data well | Memory heavy (stores all trees) |
| Feature importance built-in | MDI importance biased toward continuous features |
| Parallelizable (trees are independent) | Not as accurate as gradient boosting on tabular data |

### 1.5 Connections

- **To Quant**: Feature importance for alpha signal selection, robust prediction without extensive tuning
- **Caution for Quant**: Standard bagging violates time ordering — must use time-series aware cross-validation (e.g., purged k-fold from Marcos Lopez de Prado's "Advances in Financial Machine Learning")
- **To later topics**: Gradient Boosting (09) is the sequential counterpart; Stacking combines heterogeneous models

---

## 2. Planned Examples

### Example 1: `examples/classification_with_feature_importance/`
**Goal**: Random Forest on a classification dataset with feature importance analysis
- Train RF on sklearn breast cancer or wine dataset
- Implement bagging from scratch (bootstrap + aggregate decision trees)
- Compare: single tree vs bagged trees vs random forest
- Plot feature importance (MDI vs Permutation)
- Show OOB error convergence as number of trees increases
- **Files**: `main.py`, `random_forest_from_scratch.py`

### Example 2: `examples/alpha_signal_selection/`
**Goal**: Use RF feature importance to rank and select alpha signals
- Generate synthetic features: momentum, mean-reversion, volatility, volume, sentiment scores, noise features
- Target: future return sign or quintile
- Train Random Forest, extract feature importance
- Compare MDI vs permutation importance — show how noise features are correctly ranked low
- Discuss: time-series cross-validation (purged k-fold) to avoid lookahead bias
- **Files**: `main.py`, `signal_ranker.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Understand bootstrap, bagging, and why random feature subsets help
   - Work through the variance reduction formula
   - Understand OOB estimation
2. **Implement from scratch**: Bagging + Random Forest using Decision Tree from (03)
   - Bootstrap sampling
   - Random feature subset at each split
   - OOB error calculation
3. **Example 1**: Classification with feature importance
4. **Example 2**: Alpha signal selection (quant application)
5. **Reflect**: When does RF outperform gradient boosting? When does it fail? (Extrapolation, time-series)

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $\text{Var}_{ens} = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$ | Ensemble variance |
| $P(\text{not in bootstrap}) = (1-1/n)^n \to 1/e \approx 0.368$ | OOB fraction |
| $m = \sqrt{p}$ (clf), $m = p/3$ (reg) | Feature subset heuristic |

---

## 5. Prerequisites

- Decision Trees (03) — Random Forest is an ensemble of decision trees
- Basic probability: bootstrap sampling, independence
