# Decision Trees

## 0. Algorithm Identity Card

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Recursive axis-aligned partitions of feature space (piecewise constant functions) |
| **Loss function** | Gini impurity or Entropy (classification); Variance reduction (regression) |
| **Optimization** | Greedy top-down splitting (finding global optimum is NP-hard) |
| **Bias-variance management** | Pruning (max_depth, min_samples) reduces variance; unlimited depth = low bias, high variance |
| **Key trick** | Non-parametric: no distributional assumptions. Recursively split on the most informative feature. |

**Historical era**: Era 2 — Non-parametric & Geometric (CART, Breiman 1984)

**VC dimension**: Grows exponentially with depth. Unlimited-depth tree has $d_{VC} = \infty$, which is why trees overfit so badly.

---

## 1. What Problem Does This Solve?

Linear and logistic regression draw **straight lines** through data. But what if the decision boundary is not linear?

Look at the moons example from logistic regression — accuracy was only 87% because a line can't separate two crescents. Decision trees solve this: they partition the space into **rectangles** by asking a sequence of yes/no questions, like a game of 20 questions:

```
Is mean radius > 14.5?
├── Yes: Is worst concavity > 0.22?
│   ├── Yes: MALIGNANT (95% confident)
│   └── No:  Is mean texture > 18.5?
│       ├── Yes: MALIGNANT (70%)
│       └── No:  BENIGN (80%)
└── No:  Is worst perimeter > 105?
    ├── Yes: MALIGNANT (60%)
    └── No:  BENIGN (97% confident)
```

This is a decision tree. It's:
- **Non-parametric**: No assumption about data distribution (unlike linear/logistic which assume linearity)
- **Interpretable**: You can read the rules and explain them to a non-technical person
- **Flexible**: Can model arbitrary non-linear boundaries

---

## 2. Mathematical Foundation

### 2.1 How to Split: Measuring "Impurity"

At each node, we want to find the split that best separates the classes. We need a metric for how "mixed" a node is.

#### Gini Impurity

$$G(S) = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ is the fraction of class $k$ in node $S$.

- Pure node (all one class): $G = 1 - 1^2 = 0$
- Maximally mixed (2 classes, 50/50): $G = 1 - 0.5^2 - 0.5^2 = 0.5$

Intuition: if you randomly label a sample from this node using the class distribution, $G$ is the probability of labeling it **wrong**.

#### Entropy (Information Gain)

$$H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

- Pure node: $H = 0$ (no uncertainty)
- Maximally mixed (50/50): $H = 1$ bit

This is the **Shannon entropy** from information theory — the same quantity used in statistical mechanics ($S = -k_B \sum p_i \ln p_i$). It measures uncertainty/disorder.

**Information Gain** from a split:

$$IG(S, \text{feature}_j, \text{threshold}_t) = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)$$

The best split maximizes information gain = maximizes the reduction in uncertainty.

#### Gini vs Entropy

In practice, they give very similar results. Gini is slightly faster (no logarithm). Entropy has a cleaner information-theoretic interpretation.

#### Regression: Variance Reduction

For regression trees, replace impurity with **variance**:

$$\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$

The leaf prediction is simply the mean $\bar{y}_S$ of all samples in that leaf.

### 2.2 The CART Algorithm

**CART** (Classification and Regression Trees) builds the tree greedily:

```
function BuildTree(data S, depth):
    if stopping condition met (pure node, max_depth, min_samples):
        return Leaf(prediction = majority class or mean)
    
    for each feature j:
        for each threshold t:
            split S into S_L = {x: x_j <= t} and S_R = {x: x_j > t}
            compute impurity_reduction(S, S_L, S_R)
    
    pick (j*, t*) with maximum impurity reduction
    
    return Node(
        feature = j*, threshold = t*,
        left  = BuildTree(S_L, depth + 1),
        right = BuildTree(S_R, depth + 1)
    )
```

**Complexity**: At each node, try all features ($p$) and all possible thresholds (up to $n$ per feature). Total: $O(n \cdot p)$ per split. For a balanced tree of depth $D$: $O(n \cdot p \cdot D)$ total.

**Key fact**: This greedy algorithm does NOT find the globally optimal tree. Finding the optimal tree is **NP-hard** (Hyafil & Rivest, 1976). But the greedy approach works well in practice.

### 2.3 Prediction

For a new data point, traverse the tree from root to leaf:
- At each internal node: go left if $x_j \leq t$, else go right
- At the leaf: output the stored prediction

Prediction is $O(\text{depth})$ — very fast.

### 2.4 Overfitting and Pruning

An unlimited-depth tree can **shatter any dataset** — create one leaf per sample, achieving 0 training error. But it memorizes noise and fails on new data.

This is the most extreme bias-variance problem in classical ML:
- **Deep tree**: zero bias, very high variance (memorization)
- **Shallow tree**: high bias, low variance (underfitting)

#### Pre-pruning (stop early)

Set constraints before building:
- `max_depth`: limit tree depth
- `min_samples_split`: minimum samples required to split a node
- `min_samples_leaf`: minimum samples in each leaf

#### Post-pruning (Cost-Complexity Pruning)

Build the full tree, then prune from the bottom up. Minimize:

$$R_\alpha(T) = \underbrace{R(T)}_{\text{training error}} + \underbrace{\alpha \cdot |T|}_{\text{complexity penalty}}$$

where $|T|$ is the number of leaves and $\alpha$ is a hyperparameter (chosen by cross-validation).

This is the **same idea as regularization** in linear regression:
- Linear: $\text{MSE} + \lambda\|\beta\|^2$ (penalize large coefficients)
- Tree: $\text{Error} + \alpha\cdot\text{num\_leaves}$ (penalize large trees)

Both are instances of **Structural Risk Minimization** — minimize empirical error + complexity.

### 2.5 Feature Importance

A tree naturally ranks features by how much they contribute:

$$\text{Importance}(j) = \sum_{\text{nodes splitting on } j} \frac{n_{\text{node}}}{n_{\text{total}}} \cdot \Delta\text{Impurity}$$

Features that appear near the root (splitting the most data) are most important. Features that never appear in the tree are useless.

---

## 3. Why Trees Matter (Even If You Use Something Else)

Decision trees alone are rarely the best model. So why study them?

Because they are the **building block** for the two most powerful classical ML algorithms:

| Algorithm | Idea | Reduces |
|---|---|---|
| **Random Forest** (04) | Average many trees trained on random subsets | Variance |
| **Gradient Boosting** (09) | Sequentially build trees that correct previous errors | Bias |

Both XGBoost and LightGBM — the dominant models for tabular data in industry and competitions — are ensembles of decision trees. Understanding a single tree is essential for understanding them.

---

## 4. Pros and Cons

| Pros | Cons |
|------|------|
| Highly interpretable — can visualize and explain | Prone to overfitting without pruning |
| No feature scaling required | Unstable: small data change → different tree |
| Handles numerical and categorical features | Greedy: not globally optimal |
| Captures non-linear relationships and interactions | Axis-aligned splits: poor on diagonal boundaries |
| Built-in feature importance | High variance (single tree) |
| Fast prediction | Outperformed by ensembles on almost every task |

---

## 5. Connections

### From Logistic Regression (02)
Logistic regression can only draw linear boundaries. Decision trees can draw arbitrary (axis-aligned) boundaries. The tradeoff: trees gain flexibility but lose stability and the probabilistic interpretation.

### To Quant Finance
- **Interpretable trading rules**: "IF RSI > 70 AND volume > 2x average THEN sell" — regulators can audit this
- **Feature interaction detection**: Trees naturally find interactions (momentum only works when volatility is low)
- **Danger**: Trees overfit financial data extremely easily. A tree that memorizes "buy on the 3rd Tuesday of March" will look great in backtest and fail live.

### To Later Topics
- **Random Forest (04)**: Bagging many trees → reduces variance
- **Gradient Boosting (09)**: Boosting many trees → reduces bias
- **Neural Networks (10+)**: Can be seen as learning soft, differentiable splits instead of hard axis-aligned ones

---

## 6. Key Equations Summary

| Equation | Name |
|----------|------|
| $G(S) = 1 - \sum_k p_k^2$ | Gini Impurity |
| $H(S) = -\sum_k p_k \log_2 p_k$ | Entropy |
| $IG = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$ | Information Gain |
| $R_\alpha(T) = R(T) + \alpha|T|$ | Cost-Complexity Pruning |
