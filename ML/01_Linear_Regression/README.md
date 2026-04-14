# Linear Regression

## 1. What Is It?

Linear regression models the relationship between a target variable $y$ and input features $X$ as a linear function:

$$y = X\beta + \epsilon$$

where:
- $X \in \mathbb{R}^{n \times p}$: feature matrix ($n$ samples, $p$ features)
- $\beta \in \mathbb{R}^p$: weight vector (coefficients)
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$: noise term
- $y \in \mathbb{R}^n$: target vector

The goal is to find $\hat{\beta}$ that best fits the data.

---

## 2. Mathematical Foundation

### 2.1 Loss Function — Mean Squared Error (MSE)

$$\mathcal{L}(\beta) = \frac{1}{n} \|y - X\beta\|_2^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - x_i^T \beta)^2$$

Why MSE? Under Gaussian noise, minimizing MSE is equivalent to Maximum Likelihood Estimation (MLE).

### 2.2 Closed-Form Solution (Normal Equation)

Set the gradient to zero:

$$\nabla_\beta \mathcal{L} = -\frac{2}{n} X^T(y - X\beta) = 0$$

Solving:

$$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$

**Geometric interpretation**: $\hat{y} = X\hat{\beta}$ is the orthogonal projection of $y$ onto the column space of $X$. The residual $y - \hat{y}$ is perpendicular to every column of $X$.

**Complexity**: $O(np^2 + p^3)$ — dominated by $p^3$ for matrix inversion. Fine when $p$ is small (like in a physics experiment), problematic when $p$ is large.

### 2.3 Gradient Descent

When $p$ is large or $X^TX$ is ill-conditioned, use iterative optimization:

$$\beta \leftarrow \beta - \alpha \nabla_\beta \mathcal{L} = \beta + \frac{2\alpha}{n} X^T(y - X\beta)$$

where $\alpha$ is the **learning rate**.

| Variant | Update uses | Pros | Cons |
|---------|-------------|------|------|
| Batch GD | All $n$ samples | Stable convergence | Slow per step |
| Stochastic GD (SGD) | 1 random sample | Fast per step | Noisy updates |
| Mini-batch GD | $B$ random samples | Balance of both | Need to choose $B$ |

In physics, you'd never use gradient descent for a linear fit — the normal equation is fast enough. In ML with millions of samples and thousands of features, gradient descent (especially SGD) is essential.

### 2.4 Regularization

When you have many features, OLS can **overfit** — fitting noise rather than signal. Regularization adds a penalty:

#### Ridge Regression (L2)

$$\mathcal{L}_{ridge} = \frac{1}{n}\|y - X\beta\|^2 + \lambda \|\beta\|_2^2$$

$$\hat{\beta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$$

- Shrinks all coefficients toward zero (but never exactly zero)
- Makes $(X^TX + \lambda I)$ always invertible — fixes multicollinearity
- $\lambda \to 0$: OLS; $\lambda \to \infty$: all coefficients → 0

#### Lasso Regression (L1)

$$\mathcal{L}_{lasso} = \frac{1}{n}\|y - X\beta\|^2 + \lambda \|\beta\|_1$$

- Drives some coefficients to **exactly zero** → automatic feature selection
- No closed-form solution — solved by coordinate descent
- Useful when you suspect many features are irrelevant

#### Elastic Net

$$\mathcal{L}_{elastic} = \frac{1}{n}\|y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$$

Combines L1 (sparsity) and L2 (stability). Best of both worlds.

---

## 3. Statistical Foundation

### 3.1 Gauss-Markov Theorem

Under assumptions (linearity, exogeneity, no multicollinearity, homoscedasticity), OLS is **BLUE**: Best Linear Unbiased Estimator. No other linear estimator has lower variance.

Note: Ridge is **biased** but can have lower MSE through variance reduction.

### 3.2 Assumptions of OLS

| # | Assumption | What it means | What breaks if violated |
|---|-----------|---------------|------------------------|
| 1 | Linearity | $E[y|X] = X\beta$ | Model is fundamentally wrong |
| 2 | Exogeneity | $E[\epsilon|X] = 0$ | Coefficients are biased |
| 3 | No perfect multicollinearity | $\text{rank}(X) = p$ | $(X^TX)^{-1}$ doesn't exist |
| 4 | Homoscedasticity | $\text{Var}(\epsilon|X) = \sigma^2 I$ | Standard errors are wrong |
| 5 | Normality | $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ | Hypothesis tests invalid |

In physics, assumption 1 is usually guaranteed by theory. In ML, it's the first thing to question.

### 3.3 Model Evaluation

- **$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$**: Fraction of variance explained
  - $R^2 = 1$: perfect fit; $R^2 = 0$: no better than predicting the mean
  - In physics, $R^2 > 0.99$ is typical. In ML/finance, $R^2 = 0.05$ can be profitable.
- **Adjusted $R^2$**: Penalizes adding useless features: $R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$
- **AIC / BIC**: Model selection criteria that penalize complexity
- **Train/Test Split**: The ML way — fit on training data, evaluate on held-out test data

### 3.4 Bias-Variance Tradeoff

$$\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Model | Bias | Variance |
|-------|------|----------|
| OLS (few features) | May be high (if true relationship is non-linear) | Low |
| OLS (many features) | Low | High (overfitting) |
| Ridge ($\lambda$ large) | High (over-shrunk) | Low |
| Ridge ($\lambda$ optimal) | Moderate | Moderate |

---

## 4. Historical Context

- **Legendre (1805)** and **Gauss (1809)**: Independently developed the method of least squares for fitting astronomical orbits — literally born from physics.
- **Galton (1886)**: Coined "regression" (regression to the mean) studying hereditary stature.
- **Today**: Foundation of econometrics, factor models in finance, and the simplest ML baseline.

---

## 5. Pros and Cons

| Pros | Cons |
|------|------|
| Closed-form solution, very fast | Assumes linearity |
| Highly interpretable coefficients | Sensitive to outliers |
| Well-understood statistical properties | Poor with multicollinearity (without regularization) |
| Foundation for complex models | Cannot capture non-linear relationships |
| Regularization handles overfitting | Feature engineering needed for non-linearity |

---

## 6. Connections

### To Quant Finance
- **Fama-French factor model**: $r_i - r_f = \alpha_i + \beta_1 MKT + \beta_2 SMB + \beta_3 HML + \epsilon$
  - Coefficients = factor exposures; $\alpha$ = excess return
- **Risk attribution**: Decompose portfolio return into factor contributions
- **Signal combination**: Linear combination of alpha signals as a first pass

### To Later Topics in This Repo
- **Logistic Regression (02)**: Same framework with sigmoid, for classification
- **MLP (10)**: Neural network = stacked non-linear regressions
- **Regularization (12)**: Ridge/Lasso concepts extend to deep learning (weight decay = Ridge)
- **Everything**: Gradient descent from linear regression is the same algorithm that trains billion-parameter models

---

## 7. Key Equations Summary

| Equation | Name |
|----------|------|
| $\hat{\beta} = (X^T X)^{-1} X^T y$ | Normal equation |
| $\hat{\beta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$ | Ridge solution |
| $R^2 = 1 - SS_{res}/SS_{tot}$ | Coefficient of determination |
| $\nabla_\beta \mathcal{L} = -\frac{2}{n} X^T(y - X\beta)$ | MSE gradient |
| $\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Noise}$ | Bias-variance decomposition |
