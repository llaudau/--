# Sub-Plan: K-Nearest Neighbors (KNN)

## Status: Not Started

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | All possible functions (non-parametric, no explicit model) — "lazy learning" |
| **Loss function** | Implicit: majority vote (classification) or mean (regression) of $k$ neighbors |
| **Optimization** | None! No training phase. All computation at prediction time. |
| **Bias-variance management** | $k$ controls the tradeoff: small $k$ = low bias, high variance; large $k$ = high bias, low variance |
| **Key trick** | Don't model anything. Just remember all data, then predict like the closest neighbors. |

**Historical era**: Era 2 — Non-parametric & Geometric (Fix & Hodges, 1951). The laziest possible algorithm.

**VC dimension / generalization**: Cover-Hart theorem (1967): As $n \to \infty$, 1-NN error $\leq 2 \times$ Bayes optimal error. This is a remarkable result — the simplest possible algorithm is at most 2x worse than the best possible classifier.

**What problem from 01-02 does this solve?** No assumptions about data distribution or decision boundary shape. Can model arbitrarily complex boundaries.

**What problem does it create?** Curse of dimensionality: in high dimensions, all points become equidistant, making "nearest" meaningless. $O(nd)$ prediction cost. Must store all training data. -> Motivates dimensionality reduction (PCA, 08) and learned representations (neural nets).

**New math introduced**: Distance metrics (Euclidean, Mahalanobis, cosine), curse of dimensionality, non-parametric density estimation.

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

- **Algorithm**: No training phase — "lazy learning"
  1. Store all training data
  2. For a query point $x_q$, find $k$ nearest neighbors in training set
  3. Classification: majority vote among $k$ neighbors
  4. Regression: average (or weighted average) of $k$ neighbors' values
- **Distance metrics**:
  - Euclidean: $d(x,z) = \sqrt{\sum_j (x_j - z_j)^2}$ — most common
  - Manhattan: $d(x,z) = \sum_j |x_j - z_j|$ — robust to outliers
  - Minkowski: $d(x,z) = (\sum_j |x_j - z_j|^p)^{1/p}$ — generalizes both
  - Cosine similarity: $\text{sim}(x,z) = \frac{x \cdot z}{\|x\|\|z\|}$ — for high-dimensional sparse data
  - Mahalanobis: $d(x,z) = \sqrt{(x-z)^T \Sigma^{-1} (x-z)}$ — accounts for feature correlations
- **Weighted KNN**: Closer neighbors get higher weight
  - Weight by inverse distance: $w_i = 1/d(x_q, x_i)$
  - Weight by Gaussian kernel: $w_i = \exp(-d^2 / 2\sigma^2)$
- **Choosing $k$**:
  - Small $k$: low bias, high variance (sensitive to noise)
  - Large $k$: high bias, low variance (over-smoothing)
  - Typically: $k = \sqrt{n}$ as starting point, tune via cross-validation
  - Use odd $k$ for binary classification to avoid ties
- **Efficient search** (critical for large datasets):
  - Brute force: $O(nd)$ per query
  - KD-Tree: $O(d \log n)$ average, degrades in high dimensions
  - Ball Tree: Better than KD-Tree in moderate dimensions
  - Approximate NN: LSH (Locality Sensitive Hashing), Annoy, FAISS

### 1.2 Statistical Foundation

- **Curse of dimensionality**: As $d$ increases, distances become less meaningful
  - In high-$d$, all points are roughly equidistant
  - Volume of hypersphere shrinks relative to hypercube
  - Need exponentially more data as $d$ grows
  - Rule of thumb: KNN works well for $d < 20$ without dim reduction
- **Asymptotic properties** (Cover & Hart, 1967):
  - As $n \to \infty$, 1-NN error rate $\leq 2 \cdot \text{Bayes error}$
  - As $n \to \infty, k \to \infty, k/n \to 0$: KNN converges to Bayes optimal
- **Connection to kernel density estimation**: KNN is a form of non-parametric density estimation
- **Feature scaling is critical**: Features with large range dominate distance calculation
  - Must standardize or normalize features before KNN

### 1.3 Historical Context

- Fix & Hodges (1951) — earliest formulation
- Cover & Hart (1967) — theoretical analysis
- One of the simplest and oldest ML algorithms
- Still used as baseline and in recommendation systems

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| No training phase, simple to implement | Slow prediction: $O(nd)$ per query |
| Non-parametric, no assumptions | Curse of dimensionality |
| Naturally handles multi-class | Must store entire training set (memory) |
| Adapts to any decision boundary shape | Sensitive to irrelevant features |
| Easy to understand and explain | Requires feature scaling |
| Good baseline model | Poor with imbalanced classes (vote bias) |

### 1.5 Connections

- **To Quant**: Pattern matching in price history, regime detection, anomaly detection for outlier trades, recommendation-style approaches for similar assets
- **To later topics**: KNN connects to kernel methods (05_SVM), and the distance-based intuition reappears in attention mechanisms (22) and metric learning

---

## 2. Planned Examples

### Example 1: `examples/classification_visualization/`
**Goal**: KNN classification with decision boundary visualization
- Implement KNN from scratch with numpy
- Train on 2D synthetic data (sklearn make_classification)
- Visualize decision boundary for different $k$ values (k=1, 5, 20, 50)
- Show effect of different distance metrics
- Compare with sklearn.neighbors.KNeighborsClassifier
- **Files**: `main.py`, `knn_from_scratch.py`

### Example 2: `examples/anomaly_detection/`
**Goal**: Use KNN distance for detecting anomalous market behavior
- Features: return, volume, bid-ask spread, volatility
- Normal data: typical market days; Anomalies: flash crashes, circuit breakers
- KNN anomaly score = average distance to $k$ nearest neighbors
- Points far from their neighbors = anomalies
- Visualize anomaly scores over time, compare with known market events
- **Files**: `main.py`, `knn_anomaly.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Understand distance metrics and curse of dimensionality
   - Calculate distances by hand in 2D
   - Visualize how high-dimensional distances behave
2. **Implement from scratch**: KNN with numpy
   - Brute-force nearest neighbor search
   - Weighted voting
   - k selection via cross-validation
3. **Example 1**: Classification with boundary visualization
4. **Example 2**: Anomaly detection (quant application)
5. **Reflect**: When is KNN appropriate? How does dimensionality affect it? Why is it rarely used for large-scale production?

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $d(x,z) = \sqrt{\sum_j (x_j - z_j)^2}$ | Euclidean distance |
| $d(x,z) = \sqrt{(x-z)^T \Sigma^{-1} (x-z)}$ | Mahalanobis distance |
| 1-NN error $\leq 2 \cdot$ Bayes error (as $n \to \infty$) | Cover-Hart bound |

---

## 5. Prerequisites

- No strict ML prerequisites
- Linear algebra: norms, distances
- Understanding of overfitting / bias-variance tradeoff
