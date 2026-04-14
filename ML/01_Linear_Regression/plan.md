# Sub-Plan: Linear Regression

## Status: In Progress (README + visual_guide example done)

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Linear functions: $f(x) = w^Tx + b$ |
| **Loss function** | MSE: $\frac{1}{n}\|y - X\beta\|^2$ |
| **Optimization** | Closed-form (normal equation) or gradient descent |
| **Bias-variance management** | Ridge (L2) / Lasso (L1) regularization shrinks coefficients |
| **Key trick** | Convex loss -> unique global minimum; closed-form solution exists |

**Historical era**: Era 1 — Statistical Origins (Gauss, 1809). The very first ML algorithm.

**VC dimension**: $d_{VC} = p + 1$ (number of features + 1). Generalization bound: need $n \gg p$.

**What problem does this solve?** The most basic regression: fit a linear function to data. All later regression methods are responses to its limitations (linearity assumption, sensitivity to outliers, overfitting with many features).

**What problem does it create?** Assumes linearity. Cannot capture non-linear patterns. With many features, overfits. -> Motivates regularization (Ridge/Lasso), then non-linear models (trees, neural nets).

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

- **Model definition**: $y = X\beta + \epsilon$, where $X \in \mathbb{R}^{n \times p}$, $\beta \in \mathbb{R}^p$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **Closed-form solution (Normal Equation)**:
  - $\hat{\beta} = (X^T X)^{-1} X^T y$
  - Derivation via setting $\nabla_\beta \mathcal{L} = 0$
  - Computational complexity: $O(p^3)$ for matrix inversion
- **Loss function**: Mean Squared Error (MSE)
  - $\mathcal{L}(\beta) = \frac{1}{n} \|y - X\beta\|_2^2$
- **Gradient Descent approach**:
  - Gradient: $\nabla_\beta \mathcal{L} = -\frac{2}{n} X^T(y - X\beta)$
  - Update rule: $\beta \leftarrow \beta - \alpha \nabla_\beta \mathcal{L}$
  - Variants: Batch GD, Stochastic GD, Mini-batch GD
- **Regularized variants**:
  - Ridge (L2): $\mathcal{L} + \lambda \|\beta\|_2^2 \Rightarrow \hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$
  - Lasso (L1): $\mathcal{L} + \lambda \|\beta\|_1$ (no closed form, use coordinate descent)
  - Elastic Net: $\mathcal{L} + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$

### 1.2 Statistical Foundation

- **Gauss-Markov Theorem**: OLS is BLUE (Best Linear Unbiased Estimator) under assumptions
- **Assumptions**:
  1. Linearity: $E[y|X] = X\beta$
  2. Exogeneity: $E[\epsilon|X] = 0$
  3. No perfect multicollinearity: $\text{rank}(X) = p$
  4. Homoscedasticity: $\text{Var}(\epsilon|X) = \sigma^2 I$
  5. Normality (for inference): $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **Hypothesis testing**:
  - t-test for individual coefficients: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$
  - F-test for overall significance
  - Confidence intervals for $\beta$
- **Model evaluation**:
  - $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$, Adjusted $R^2$
  - AIC, BIC for model selection
  - Residual analysis: Q-Q plots, heteroscedasticity tests
- **Bias-Variance tradeoff**:
  - OLS: zero bias, potentially high variance
  - Ridge: introduces bias, reduces variance
  - Connection to regularization strength $\lambda$

### 1.3 Historical Context

- Legendre (1805) and Gauss (1809) — method of least squares
- Originally for astronomical orbit fitting
- Foundation of all regression analysis and econometrics

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Closed-form solution, fast | Assumes linearity |
| Highly interpretable coefficients | Sensitive to outliers |
| Well-understood statistical properties | Poor with multicollinearity |
| Foundation for more complex models | Cannot capture non-linear relationships |
| Regularization variants handle overfitting | Feature engineering required for non-linearity |

### 1.5 Connections

- **To Quant**: Fama-French factor models ($r_i = \alpha + \beta_1 MKT + \beta_2 SMB + \beta_3 HML + \epsilon$), portfolio regression, risk attribution
- **To later topics**: Logistic regression (02) extends this to classification; Neural networks (10) generalize via non-linear activation; Ridge regression connects to L2 regularization in deep learning (12)

---

## 2. Planned Examples

### Example 1: `examples/housing_price_prediction/`
**Goal**: Classic regression on California housing dataset
- Load and explore data (EDA with matplotlib)
- Implement OLS from scratch (normal equation + gradient descent)
- Compare with `sklearn.linear_model.LinearRegression`
- Residual analysis and diagnostic plots
- **Files**: `main.py`, `linear_regression_from_scratch.py`

### Example 2: `examples/factor_model_quant/`
**Goal**: Build a simple Fama-French-style factor model
- Fetch stock return data (use synthetic or CSV)
- Regress individual stock returns on market factors
- Interpret $\beta$ coefficients as factor exposures
- Ridge regression for multi-factor model with multicollinearity
- Evaluate in-sample vs out-of-sample $R^2$
- **Files**: `main.py`, `factor_model.py`, `data/`

---

## 3. Learning Steps

1. **Theory first**: Read and understand the math in README.md
   - Derive normal equation by hand
   - Understand the geometric interpretation (projection onto column space of X)
2. **Implement from scratch**: Code OLS with numpy only
   - Normal equation
   - Gradient descent (batch + stochastic)
   - Ridge regression
3. **Example 1**: Housing price prediction (standard ML workflow)
4. **Example 2**: Factor model (quant application)
5. **Reflect**: What happens when assumptions are violated? How does regularization help?

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $\hat{\beta} = (X^T X)^{-1} X^T y$ | Normal equation |
| $\hat{\beta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$ | Ridge solution |
| $R^2 = 1 - SS_{res}/SS_{tot}$ | Coefficient of determination |
| $\nabla_\beta \mathcal{L} = -\frac{2}{n} X^T(y - X\beta)$ | MSE gradient |

---

## 5. Prerequisites

- Linear algebra: matrix multiplication, inverse, eigenvalues
- Calculus: partial derivatives, gradient
- Probability: normal distribution, expectation, variance
