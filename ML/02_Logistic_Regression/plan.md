# Sub-Plan: Logistic Regression

## Status: Not Started

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Linear decision boundary with sigmoid: $f(x) = \sigma(w^Tx + b)$ |
| **Loss function** | Binary cross-entropy (= negative log-likelihood): $-\frac{1}{n}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ |
| **Optimization** | Gradient descent (no closed-form), Newton-Raphson / IRLS |
| **Bias-variance management** | L2/L1 regularization on weights; C parameter in sklearn = $1/\lambda$ |
| **Key trick** | Sigmoid maps linear output to probability; cross-entropy is convex -> global optimum guaranteed |

**Historical era**: Era 1 — Statistical Origins (Berkson, 1944). Extends linear regression to classification.

**VC dimension**: Same as linear classifier: $d_{VC} = p + 1$. Linear decision boundary in feature space.

**What problem from 01 does this solve?** Linear regression outputs unbounded real values. For classification (yes/no, up/down), we need probabilities in $[0,1]$. Sigmoid wraps the linear output into a probability.

**What problem does it create?** Still assumes a linear decision boundary. Cannot model XOR-like patterns. -> Motivates kernel methods (SVM, 05), tree-based methods (03), and neural networks (10).

**New math introduced**: Maximum Likelihood Estimation (MLE), cross-entropy loss, information theory connection. MLE unifies probability and optimization — this idea powers all of deep learning.

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

- **Model definition**: Binary classification — predict $P(y=1|x)$
  - $P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$
  - Decision boundary: $w^T x + b = 0$ (linear hyperplane)
- **Sigmoid function** $\sigma(z) = \frac{1}{1+e^{-z}}$:
  - Properties: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, maps $\mathbb{R} \to (0,1)$
  - Log-odds (logit): $\log \frac{p}{1-p} = w^T x + b$
- **Loss function**: Binary Cross-Entropy (negative log-likelihood)
  - $\mathcal{L}(w) = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$
  - Why not MSE? Non-convex for sigmoid output; cross-entropy is convex
- **Optimization**: No closed-form solution — must use iterative methods
  - Gradient: $\nabla_w \mathcal{L} = \frac{1}{n} X^T (\hat{p} - y)$ (same form as linear regression!)
  - Newton-Raphson / IRLS (Iteratively Reweighted Least Squares)
  - Gradient descent variants (SGD, Adam)
- **Multi-class extensions**:
  - One-vs-Rest (OvR): K binary classifiers
  - Softmax regression: $P(y=k|x) = \frac{e^{w_k^T x}}{\sum_j e^{w_j^T x}}$
  - Cross-entropy loss for multi-class
- **Regularization**:
  - L2 (Ridge): prevents large weights, standard in practice
  - L1 (Lasso): sparse feature selection
  - `C` parameter in sklearn: $C = 1/\lambda$ (inverse regularization)

### 1.2 Statistical Foundation

- **Maximum Likelihood Estimation (MLE)**:
  - Likelihood: $L(w) = \prod_{i=1}^n \hat{p}_i^{y_i} (1-\hat{p}_i)^{1-y_i}$
  - Log-likelihood leads to cross-entropy loss
  - Asymptotic normality of MLE: $\hat{w} \sim \mathcal{N}(w^*, I^{-1}(w^*))$
- **Connection to exponential family**: Bernoulli distribution with canonical link = logit
- **Generalized Linear Model (GLM)** framework:
  - Link function: logit
  - Linear predictor: $\eta = w^T x$
  - Response distribution: Bernoulli
- **Evaluation metrics** (critical for classification):
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - ROC curve and AUC
  - Calibration: predicted probability vs actual frequency
- **Hypothesis testing**:
  - Wald test for individual coefficients
  - Likelihood ratio test for model comparison
  - Deviance and pseudo-$R^2$

### 1.3 Historical Context

- Joseph Berkson (1944) coined "logit" and proposed the model
- Cox (1958) formalized the regression framework
- Foundation of credit scoring, medical diagnosis, and classification

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Outputs calibrated probabilities | Assumes linear decision boundary |
| Highly interpretable (log-odds) | Cannot capture complex non-linear patterns |
| Convex loss, global optimum guaranteed | Requires feature engineering for non-linearity |
| Fast training, scales well | Sensitive to multicollinearity |
| Strong theoretical foundation (MLE/GLM) | Performance ceiling on complex tasks |

### 1.5 Connections

- **From Linear Regression (01)**: Same gradient form, but with sigmoid transformation
- **To Quant**: Predicting up/down market direction, credit default probability, signal classification (buy/sell/hold)
- **To later topics**: Softmax in neural networks (10) is multi-class logistic regression; Cross-entropy loss used everywhere in deep learning

---

## 2. Planned Examples

### Example 1: `examples/binary_classification/`
**Goal**: Classic binary classification from scratch
- Implement logistic regression with numpy (sigmoid, cross-entropy, gradient descent)
- Train on a 2D dataset (e.g., sklearn `make_moons` or `make_classification`)
- Visualize decision boundary
- Compare with `sklearn.linear_model.LogisticRegression`
- Plot ROC curve, compute AUC
- **Files**: `main.py`, `logistic_regression_from_scratch.py`

### Example 2: `examples/market_direction_prediction/`
**Goal**: Predict next-day market up/down using simple features
- Features: lagged returns, volatility, volume change, moving average crossover
- Target: 1 if next-day return > 0, else 0
- Train logistic regression with L2 regularization
- Evaluate with precision, recall, AUC — discuss why accuracy is misleading for imbalanced data
- Interpret coefficients as feature importance
- **Files**: `main.py`, `market_classifier.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Derive cross-entropy loss from MLE
   - Understand why sigmoid + cross-entropy is convex
   - Compare gradient with linear regression gradient
2. **Implement from scratch**: Logistic regression with numpy
   - Binary case with gradient descent
   - Multi-class with softmax
3. **Example 1**: Binary classification with visualization
4. **Example 2**: Market direction prediction (quant application)
5. **Evaluation deep-dive**: ROC, AUC, calibration curves
6. **Reflect**: When is logistic regression sufficient vs when do we need non-linear models?

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid function |
| $\mathcal{L} = -\frac{1}{n}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | Binary cross-entropy |
| $\nabla_w \mathcal{L} = \frac{1}{n} X^T(\hat{p} - y)$ | Gradient |
| $\text{logit}(p) = \log\frac{p}{1-p} = w^T x + b$ | Log-odds |

---

## 5. Prerequisites

- Linear Regression (01) — understand OLS and gradient descent first
- Probability: Bernoulli distribution, likelihood, MLE
- Information theory basics: entropy, cross-entropy (optional but helpful)
