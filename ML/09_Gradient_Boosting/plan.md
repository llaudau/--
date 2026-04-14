# Sub-Plan: Gradient Boosting (XGBoost / LightGBM)

## Status: Not Started

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

#### General Gradient Boosting Framework
- **Idea**: Build an additive model sequentially — each new model corrects the errors of the ensemble so far
  $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$
  where $\eta$ is learning rate and $h_m$ is the $m$-th weak learner
- **Functional gradient descent**: Optimize in function space
  1. Compute pseudo-residuals: $r_i^{(m)} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$
  2. Fit $h_m$ to pseudo-residuals $\{(x_i, r_i^{(m)})\}$
  3. Update: $F_m = F_{m-1} + \eta \cdot h_m$
- **For MSE loss**: pseudo-residuals $= y_i - F_{m-1}(x_i)$ (literal residuals)
- **For log-loss**: pseudo-residuals $= y_i - \sigma(F_{m-1}(x_i))$ (probability residuals)
- **Learning rate** $\eta$ (shrinkage): Smaller $\eta$ requires more trees but generalizes better (regularization)

#### AdaBoost (precursor)
- Special case: exponential loss, weighted resampling
- Weight misclassified samples higher -> next learner focuses on hard examples
- $\alpha_m = \frac{1}{2} \ln \frac{1 - \text{err}_m}{\text{err}_m}$: learner weight inversely proportional to error

#### XGBoost Specifics
- **Regularized objective**:
  $\mathcal{L} = \sum_i L(y_i, \hat{y}_i) + \sum_m [\gamma T_m + \frac{1}{2}\lambda \|w_m\|^2]$
  - $T_m$ = number of leaves in tree $m$, $w_m$ = leaf weights
  - $\gamma$ penalizes tree complexity, $\lambda$ penalizes leaf weight magnitude
- **Second-order approximation** (Newton's method):
  $\mathcal{L}^{(m)} \approx \sum_i [g_i h_m(x_i) + \frac{1}{2} h_i h_m(x_i)^2] + \Omega(h_m)$
  - $g_i = \partial L / \partial \hat{y}_i$ (gradient), $h_i = \partial^2 L / \partial \hat{y}_i^2$ (Hessian)
- **Optimal leaf weight**: $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$
- **Split gain**: $\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$
  - Only split if Gain > 0

#### LightGBM Innovations
- **Gradient-based One-Side Sampling (GOSS)**: Keep samples with large gradients, downsample small-gradient samples
- **Exclusive Feature Bundling (EFB)**: Bundle mutually exclusive sparse features
- **Leaf-wise tree growth** (vs XGBoost's level-wise): Grow the leaf with max delta loss -> deeper, more asymmetric trees
- **Histogram-based splitting**: Bin continuous features into discrete bins -> $O(n)$ instead of $O(n \log n)$ per split
- **Categorical feature support**: Native handling without one-hot encoding

#### CatBoost (brief)
- **Ordered boosting**: Avoids target leakage in gradient estimation
- **Symmetric trees**: Balanced trees for faster inference
- **Best native categorical support**: Target encoding with permutation

### 1.2 Statistical Foundation

- **Bias-Variance**: Boosting primarily reduces **bias** (unlike bagging which reduces variance)
  - Each tree corrects systematic errors of the ensemble
  - But can overfit with too many trees (variance increases)
  - Learning rate $\eta$ controls this tradeoff
- **Regularization mechanisms**:
  - Learning rate (shrinkage): $\eta < 1$
  - Subsampling rows and columns (like random forest)
  - Tree constraints: max_depth, min_child_weight
  - L1/L2 on leaf weights (XGBoost)
  - Early stopping: monitor validation loss, stop when it increases
- **Feature importance**: Gain-based (total gain from splits on feature), split count, or SHAP values
- **SHAP (SHapley Additive exPlanations)**: Game-theoretic approach to feature importance
  - Consistent, locally accurate, additive
  - Tree-based SHAP is fast: $O(TLD^2)$ per prediction

### 1.3 Historical Context

- Freund & Schapire (1997): AdaBoost
- Friedman (2001): Gradient Boosting Machines
- Chen & Guestrin (2016): XGBoost — dominated Kaggle
- Ke et al. (2017): LightGBM — faster training
- Prokhorenkova et al. (2018): CatBoost — better categorical handling
- **Current status**: Gradient boosting is the default for tabular data in industry and competitions

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| State-of-the-art for tabular data | Many hyperparameters to tune |
| Handles mixed feature types | Slower training than random forest |
| Built-in regularization | Sequential (less parallelizable than RF) |
| Feature importance + SHAP | Can overfit without early stopping |
| Handles missing values natively (XGB/LGBM) | Cannot extrapolate beyond training range |
| Fast inference (tree traversal) | Sensitive to noisy labels |

### 1.5 Connections

- **To Quant**: **The dominant model** for tabular quant signals — used in nearly every top Kaggle competition and many production quant systems; feature importance for signal selection; SHAP for model interpretability (regulatory compliance); time-series specific: must use purged cross-validation, walk-forward optimization
- **Caution for Quant**:
  - Lookahead bias: standard CV leaks future info in time-series
  - Non-stationarity: market regimes change; model trained on one regime may fail in another
  - Transaction costs: model must predict returns larger than costs
  - Marcos Lopez de Prado's purged k-fold cross-validation is essential
- **To later topics**: Gradient boosting is conceptually similar to residual learning in ResNet (15) — both learn corrections to previous predictions

---

## 2. Planned Examples

### Example 1: `examples/xgboost_vs_lightgbm/`
**Goal**: Compare XGBoost and LightGBM on a classification task
- Train both on a tabular dataset (e.g., sklearn classification or Kaggle-style)
- Implement simple gradient boosting from scratch (MSE loss, decision stumps)
- Compare: accuracy, training time, tree structure
- Hyperparameter tuning with Optuna or grid search
- SHAP value analysis for feature importance
- **Files**: `main.py`, `gradient_boosting_from_scratch.py`, `comparison.py`

### Example 2: `examples/quant_signal_prediction/`
**Goal**: Predict stock returns using gradient boosting with proper quant methodology
- Features: momentum (1m, 3m, 12m), volatility, value ratios, volume, technical indicators
- Target: forward 1-month return quintile or sign
- **Critical**: Implement purged walk-forward cross-validation
  - No lookahead: train on past, predict future
  - Purge gap between train and test to avoid info leakage from overlapping labels
- Compare XGBoost vs LightGBM vs Random Forest
- SHAP analysis: which signals matter most?
- Discuss: how to go from model prediction to portfolio construction
- **Files**: `main.py`, `quant_gbm.py`, `purged_cv.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Understand gradient boosting as functional gradient descent
   - Derive pseudo-residuals for MSE and log-loss
   - Understand XGBoost's second-order approximation and regularization
   - Compare AdaBoost, GBM, XGBoost, LightGBM
2. **Implement from scratch**: Simple gradient boosting with decision stumps
   - Fit trees to pseudo-residuals
   - Learning rate and early stopping
3. **Example 1**: XGBoost vs LightGBM comparison
4. **Example 2**: Quant signal prediction with proper methodology
5. **Deep-dive**: SHAP values — understand and implement
6. **Reflect**: Why is gradient boosting king for tabular data? When do neural networks win? (Unstructured data, very large datasets)

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $F_m = F_{m-1} + \eta \cdot h_m$ | Boosting update |
| $r_i^{(m)} = -\partial L / \partial F_{m-1}(x_i)$ | Pseudo-residuals |
| $w_j^* = -G_j / (H_j + \lambda)$ | XGBoost optimal leaf weight |
| Gain $= \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}] - \gamma$ | XGBoost split gain |

---

## 5. Prerequisites

- Decision Trees (03): Gradient boosting uses trees as base learners
- Random Forest (04): Understand bagging for comparison
- Calculus: Gradients and Hessians (second-order Taylor expansion)
- For quant example: basic understanding of financial features (momentum, volatility)

---

## 6. Recommended Reading

- Original XGBoost paper: Chen & Guestrin (2016)
- LightGBM paper: Ke et al. (2017)
- "Advances in Financial Machine Learning" — Marcos Lopez de Prado (chapters on cross-validation and feature importance)
- SHAP paper: Lundberg & Lee (2017)
