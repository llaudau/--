# Sub-Plan: Decision Trees

## Status: Not Started

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Recursive axis-aligned partitions of feature space (piecewise constant) |
| **Loss function** | Gini impurity or Entropy (classification); Variance reduction (regression) |
| **Optimization** | Greedy top-down splitting (finding global optimum is NP-hard) |
| **Bias-variance management** | Pruning (max_depth, min_samples) = reducing variance; deep tree = low bias, high variance |
| **Key trick** | Non-parametric: no assumption about data distribution. Recursively split on the most informative feature. |

**Historical era**: Era 2 — Non-parametric & Geometric (CART, Breiman 1984). First algorithm to drop distributional assumptions.

**VC dimension**: $d_{VC}$ grows exponentially with depth. Unlimited-depth tree: $d_{VC} = \infty$ (can shatter any finite dataset). This is why trees overfit so badly — and why pruning / ensembles are critical.

**What problem from 01-02 does this solve?** Linear/logistic regression assumes linearity. Decision trees capture arbitrary non-linear boundaries, handle mixed feature types, and require no scaling.

**What problem does it create?** Extremely high variance — small data changes produce completely different trees. Greedy splitting is suboptimal. -> This directly motivates ensembles: Random Forest (04) and Gradient Boosting (09).

**New math introduced**: Information entropy $H = -\sum p\log p$ (Pillar 4 from Introduction). Greedy algorithms. Cost-complexity pruning as a form of regularization ($R_\alpha(T) = R(T) + \alpha|T|$ mirrors Ridge's $\mathcal{L} + \lambda\|\beta\|^2$).

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

- **Tree structure**: Recursive binary partitioning of feature space
  - Each internal node: a split on feature $x_j \leq t$
  - Each leaf: a prediction (class label or mean value)
- **Splitting criteria for classification**:
  - **Gini Impurity**: $G(S) = 1 - \sum_{k=1}^K p_k^2$
    - $p_k$ = fraction of class $k$ in node $S$
    - Gini = 0 means pure node
  - **Entropy / Information Gain**: $H(S) = -\sum_{k=1}^K p_k \log_2 p_k$
    - Information Gain: $IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$
  - **Comparison**: Gini slightly faster (no log), both give similar results in practice
- **Splitting criteria for regression**:
  - **Variance reduction**: $\text{Var}(S) - \sum_v \frac{|S_v|}{|S|} \text{Var}(S_v)$
  - Leaf prediction = mean of samples in leaf
- **Tree building algorithm (CART)**:
  - For each feature $j$ and threshold $t$, compute impurity reduction
  - Choose $(j, t)$ that maximizes impurity reduction
  - Recursively split left and right subsets
  - Greedy algorithm (not globally optimal) — NP-hard to find optimal tree
- **Pruning** (controlling overfitting):
  - **Pre-pruning**: max_depth, min_samples_split, min_samples_leaf
  - **Post-pruning (Cost-Complexity Pruning)**: minimize $R_\alpha(T) = R(T) + \alpha |T|$
    - $R(T)$ = training error, $|T|$ = number of leaves, $\alpha$ = complexity penalty
    - Find optimal $\alpha$ via cross-validation

### 1.2 Statistical Foundation

- **Non-parametric model**: No assumption on data distribution
- **Bias-Variance**:
  - Deep trees: low bias, high variance (overfit)
  - Shallow trees: high bias, low variance (underfit)
  - This motivates ensemble methods (04)
- **Feature importance**: Total impurity reduction contributed by each feature across all splits
- **Handling categorical features**: Can split on subsets (CART: binary splits only)
- **Missing values**: Surrogate splits (use correlated features when primary is missing)

### 1.3 Historical Context

- CART (Breiman et al., 1984) — Classification and Regression Trees
- ID3 (Quinlan, 1986) -> C4.5 (1993) -> C5.0
- Foundation for Random Forests and Gradient Boosting

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Highly interpretable (visualize the tree) | Prone to overfitting without pruning |
| No feature scaling required | Unstable: small data changes -> different tree |
| Handles both numerical and categorical | Greedy splits, not globally optimal |
| Captures non-linear relationships | Axis-aligned splits, poor on diagonal boundaries |
| Built-in feature selection | High variance (motivates ensembles) |
| Fast inference | Can create biased trees with imbalanced data |

### 1.5 Connections

- **To Quant**: Rule-based trading strategies ("if RSI > 70 and volume > avg, then sell"), interpretable risk models required by regulators
- **To later topics**: Random Forest (04) = bagging of trees; Gradient Boosting (09) = sequential trees; Both reduce the high variance of single trees

---

## 2. Planned Examples

### Example 1: `examples/classification_iris/`
**Goal**: Train a decision tree on Iris dataset, visualize the tree
- Build tree from scratch (recursive splitting with Gini/Entropy)
- Visualize with `sklearn.tree.plot_tree` and `graphviz`
- Show how max_depth affects overfitting
- Compare Gini vs Entropy
- **Files**: `main.py`, `decision_tree_from_scratch.py`

### Example 2: `examples/trading_rules/`
**Goal**: Decision tree as an interpretable trading rule extractor
- Features: RSI, MACD, Bollinger Band position, volume ratio, moving average slopes
- Target: profitable trade (1) or not (0) over next N bars
- Train decision tree with limited depth (interpretability)
- Extract and print human-readable rules from the tree
- Discuss why overfitting is dangerous: tree memorizes past patterns that won't repeat
- **Files**: `main.py`, `rule_extractor.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Understand Gini, Entropy, Information Gain
   - Compute splitting criteria by hand on a small example
   - Understand why greedy splitting is used (global optimum is NP-hard)
2. **Implement from scratch**: CART algorithm with numpy
   - Recursive tree building
   - Prediction traversal
   - Pruning with max_depth
3. **Example 1**: Iris classification with visualization
4. **Example 2**: Trading rule extraction (quant application)
5. **Reflect**: Why do single trees overfit? -> motivation for ensembles (04)

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $G(S) = 1 - \sum_k p_k^2$ | Gini Impurity |
| $H(S) = -\sum_k p_k \log_2 p_k$ | Entropy |
| $IG = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$ | Information Gain |
| $R_\alpha(T) = R(T) + \alpha|T|$ | Cost-Complexity Pruning |

---

## 5. Prerequisites

- No strict ML prerequisites — can be learned independently
- Basic probability: conditional probability
- Understanding of overfitting concept
