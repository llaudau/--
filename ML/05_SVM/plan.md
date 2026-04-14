# Sub-Plan: Support Vector Machines (SVM)

## Status: Not Started

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Maximum-margin hyperplanes (linear); via kernel trick: non-linear boundaries in RKHS |
| **Loss function** | Hinge loss: $\max(0, 1 - y \cdot f(x))$ + margin regularization $\frac{1}{2}\|w\|^2$ |
| **Optimization** | Quadratic programming (dual problem); depends on data only through inner products $x_i^Tx_j$ |
| **Bias-variance management** | Margin maximization directly minimizes VC-dimension bound. $C$ trades off margin width vs errors. Kernel choice controls complexity. |
| **Key trick** | Kernel trick: replace inner products with $K(x_i, x_j)$ to operate in infinite-dimensional space without computing the mapping |

**Historical era**: Era 2 — Non-parametric & Geometric (Vapnik, 1992). The algorithm that was *built from* generalization theory.

**VC dimension**: For margin $\gamma$: $d_{VC} \leq \min(R^2/\gamma^2, p) + 1$. Larger margin -> smaller effective VC dimension -> better generalization. **SVM is the only classical algorithm designed directly from VC/SRM theory** — Vapnik invented both the theory and the algorithm.

**What problem from 01-02 does this solve?** Linear/logistic regression finds *a* separating hyperplane. SVM finds the *best* one (maximum margin), with rigorous generalization guarantees. Kernel trick handles non-linearity without explicit feature engineering.

**What problem does it create?** $O(n^2)$ to $O(n^3)$ training — doesn't scale to large datasets. Kernel selection requires expertise. No native probability output. -> Deep learning replaced SVM for large-scale problems.

**New math introduced**: Lagrangian duality, KKT conditions, kernel trick, Mercer's theorem, VC dimension and Structural Risk Minimization (the rigorous generalization theory we discussed).

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

#### Hard-Margin SVM (linearly separable)
- **Goal**: Find the hyperplane $w^T x + b = 0$ that maximizes the margin between classes
- **Margin**: Distance between the hyperplane and the nearest data point = $\frac{2}{\|w\|}$
- **Optimization problem**:
  - $\min_{w,b} \frac{1}{2}\|w\|^2$ subject to $y_i(w^T x_i + b) \geq 1, \forall i$
  - Quadratic programming (QP) with linear constraints — convex!
- **Support vectors**: Points on the margin boundary ($y_i(w^T x_i + b) = 1$)
  - Only support vectors determine the decision boundary
  - Sparse solution: most $\alpha_i = 0$

#### Soft-Margin SVM (non-separable)
- **Slack variables** $\xi_i \geq 0$: allow misclassification
  - $\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$
  - subject to $y_i(w^T x_i + b) \geq 1 - \xi_i$
  - $C$: trade-off between margin width and misclassification penalty
- **Hinge loss** interpretation: $\mathcal{L} = \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i f(x_i))$

#### Lagrangian Dual Problem
- **KKT conditions** and Lagrange multipliers $\alpha_i$
- **Dual formulation**:
  $\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$
  subject to $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$
- Key insight: dual depends on data only through inner products $x_i^T x_j$

#### The Kernel Trick
- **Replace** $x_i^T x_j$ with $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$
- Compute inner product in high-dimensional space without explicitly mapping there
- **Common kernels**:
  - Linear: $K(x,z) = x^T z$
  - Polynomial: $K(x,z) = (x^T z + c)^d$
  - RBF (Gaussian): $K(x,z) = \exp(-\gamma \|x-z\|^2)$ — infinite-dimensional feature space
  - Sigmoid: $K(x,z) = \tanh(\kappa x^T z + c)$
- **Mercer's theorem**: $K$ is valid iff kernel matrix is positive semi-definite
- **RBF parameter $\gamma$**: large $\gamma$ = small radius = complex boundary (overfit); small $\gamma$ = smooth boundary (underfit)

#### SVM for Regression (SVR)
- $\epsilon$-insensitive loss: ignore errors smaller than $\epsilon$
- $\min \frac{1}{2}\|w\|^2 + C \sum_i (\xi_i + \xi_i^*)$
  subject to $|y_i - (w^T x_i + b)| \leq \epsilon + \xi_i$

### 1.2 Statistical Foundation

- **VC dimension theory**: SVM maximizes margin = minimizes VC dimension bound on generalization error
- **Structural Risk Minimization (SRM)**: Balance empirical risk + complexity
- **Reproducing Kernel Hilbert Spaces (RKHS)**: Theoretical framework for kernels
- **Comparison with logistic regression**:
  - SVM: hinge loss (non-differentiable at 1), sparse solution
  - Logistic: log loss (smooth), no sparsity, outputs probabilities
- **Multi-class SVM**: One-vs-One ($K(K-1)/2$ classifiers) or One-vs-Rest ($K$ classifiers)

### 1.3 Historical Context

- Vapnik & Chervonenkis (1963) — VC theory
- Boser, Guyon, Vapnik (1992) — kernel SVM
- Dominated ML before deep learning era (2000s)
- Won many competitions, especially with RBF kernel on small-medium datasets

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Effective in high-dimensional spaces | Slow for large datasets: $O(n^2)$ to $O(n^3)$ training |
| Memory efficient (only support vectors) | Sensitive to feature scaling (must normalize!) |
| Kernel trick handles non-linearity elegantly | No native probability output (need Platt scaling) |
| Strong theoretical foundation (VC, SRM) | Kernel and hyperparameter selection is critical |
| Robust to overfitting in high dimensions | Hard to interpret (especially with RBF kernel) |

### 1.5 Connections

- **To Quant**: Classification of market regimes, anomaly detection with one-class SVM, pairs trading signal classification
- **To LQCD**: Kernel methods share mathematical framework with Gaussian processes used in lattice analysis
- **To later topics**: Kernel methods connect to neural networks (infinite-width NNs correspond to kernels — Neural Tangent Kernel); Hinge loss used in some NN training

---

## 2. Planned Examples

### Example 1: `examples/kernel_classification/`
**Goal**: Demonstrate SVM with different kernels on non-linear data
- Generate non-linearly separable 2D data (concentric circles, moons)
- Train SVM with linear, polynomial, and RBF kernels
- Visualize decision boundaries for each kernel
- Show effect of $C$ and $\gamma$ on boundary complexity
- Plot support vectors
- **Files**: `main.py`, `svm_visualization.py`

### Example 2: `examples/market_regime_classification/`
**Goal**: Classify market regimes (bull/bear/sideways) using SVM
- Features: rolling volatility, trend strength, volume patterns, VIX proxy
- Use RBF-SVM for non-linear regime boundaries
- Grid search for optimal $C$ and $\gamma$
- Evaluate with time-series cross-validation
- Compare SVM vs Logistic Regression performance
- **Files**: `main.py`, `regime_classifier.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Derive the dual problem from the primal
   - Understand why the kernel trick works (inner product replacement)
   - Geometric intuition: margin maximization
2. **Implement**: SVM with gradient descent on hinge loss (simplified, no dual)
   - Understand sklearn's SVC internals (libsvm)
3. **Example 1**: Kernel classification with visualization
4. **Example 2**: Market regime classification (quant application)
5. **Reflect**: When is SVM better than logistic regression? When do we prefer tree-based methods? Why did deep learning largely replace SVM?

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $\min \frac{1}{2}\|w\|^2 + C\sum \xi_i$ | Soft-margin primal |
| $K(x,z) = \exp(-\gamma\|x-z\|^2)$ | RBF kernel |
| $\text{Hinge loss} = \max(0, 1 - y \cdot f(x))$ | Hinge loss |
| Margin $= 2 / \|w\|$ | Margin width |

---

## 5. Prerequisites

- Linear Regression (01): optimization basics
- Logistic Regression (02): classification, decision boundaries
- Linear algebra: inner products, norms, eigenvalues (for kernel theory)
- Lagrangian optimization (dual problem) — will be introduced in README
