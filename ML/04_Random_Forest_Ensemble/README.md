# Random Forest and Ensemble Methods

## 0. Algorithm Identity Card

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Averages or votes of many decision trees, each trained on bootstrapped data with random feature subsets |
| **Loss function** | Same local split criteria as trees: Gini/Entropy for classification, variance reduction for regression |
| **Optimization** | Train each tree independently, then aggregate predictions |
| **Bias-variance management** | Bagging reduces variance; random feature subsets de-correlate trees and reduce variance further |
| **Key trick** | Average many unstable but low-bias trees to produce a much more stable predictor |

**Historical era**: Era 3 — Ensemble Methods (Breiman, 1996/2001)

**VC dimension intuition**: A single deep tree has extremely high complexity. A forest still uses deep trees, but averaging prevents that flexibility from turning directly into wild prediction variance.

---

## 1. What Problem Does This Solve?

A single decision tree is flexible and interpretable, but it is also unstable. Change the training set slightly, and the learned splits can change a lot. That means the tree has **low bias** but **high variance**.

Random Forest fixes this by asking:

> What if we train many different trees, make their mistakes less correlated, and then average them?

That is the whole method:

- **Bootstrap the data** so each tree sees a different sample
- **Randomize the candidate features** at each split so trees do not all make the same decisions
- **Average or vote** across trees

The result is usually:

- much better generalization than a single tree
- strong performance on tabular data with little tuning
- built-in feature importance and out-of-bag validation

The price is interpretability. A forest is usually much better than one tree, but much harder to explain as a compact set of rules.

---

## 2. Mathematical Foundation

### 2.1 Bagging: Bootstrap Aggregating

Suppose we have a high-variance learner, like a deep decision tree. We train $M$ copies of it on different bootstrap samples:

$$
\mathcal{D}^{(1)}, \mathcal{D}^{(2)}, \dots, \mathcal{D}^{(M)}
$$

where each bootstrap sample is drawn **with replacement** from the original dataset of size $n$.

For regression, the ensemble prediction is:

$$
\hat{f}_{ens}(x) = \frac{1}{M}\sum_{m=1}^M \hat{f}^{(m)}(x)
$$

For classification, use majority vote or average class probabilities.

The key point is that averaging keeps the bias roughly similar while shrinking variance.

### 2.2 Why Averaging Works

If each base model has variance $\sigma^2$ and pairwise correlation $\rho$, then the variance of the average is:

$$
\mathrm{Var}_{ens} = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2
$$

This equation is the heart of Random Forest.

- If trees are highly correlated ($\rho \approx 1$), averaging helps little
- If trees are weakly correlated, averaging helps a lot
- As $M \to \infty$, variance approaches $\rho \sigma^2$

So the real objective is not just "more trees". It is:

> build many trees whose errors are not too correlated

### 2.3 Bootstrap and Out-of-Bag Samples

When drawing $n$ times with replacement from $n$ original samples, the probability a point is never selected is:

$$
\left(1 - \frac{1}{n}\right)^n \to \frac{1}{e} \approx 0.368
$$

So each bootstrap sample contains:

- about **63.2% unique samples**
- about **36.8% out-of-bag (OOB) samples**

Those OOB samples act like a free validation set. For each training point, predict it using only trees that did not include it, then aggregate those predictions to estimate generalization error.

This is one reason Random Forest is practical: you often get a decent validation estimate without setting aside a separate split.

### 2.4 Random Forest = Bagging + Random Feature Subsets

Bagging alone still leaves a problem: if one feature is very strong, many trees will split on it near the root, making trees look similar and increasing correlation.

Random Forest fixes this:

- at each split, instead of testing all $p$ features
- test only a random subset of size $m$

Common heuristics:

- classification: $m \approx \sqrt{p}$
- regression: $m \approx p/3$

This increases diversity across trees and lowers $\rho$ in the ensemble variance formula.

### 2.5 Prediction and Feature Importance

#### Prediction

- **Classification**: majority vote or mean predicted probability
- **Regression**: average of tree outputs

#### Mean Decrease in Impurity (MDI)

Each time a feature is used to split, it reduces impurity. Sum those reductions over the forest:

$$
\mathrm{Importance}(j) = \sum_{\text{splits on } j} \frac{n_{node}}{n_{total}} \cdot \Delta \mathrm{Impurity}
$$

This is fast, but biased toward high-cardinality or continuous features.

#### Permutation Importance

Shuffle feature $j$ after training and measure how much performance drops. If the drop is large, the feature mattered.

This is usually more trustworthy than MDI, especially when comparing features of different types.

---

## 3. Why This Matters

Random Forest is one of the most important classical ML algorithms because it teaches three major ideas:

1. **Ensembling can beat a single strong model**
2. **Reducing variance is often more valuable than squeezing bias lower**
3. **Randomization is not noise for its own sake; it is a tool for de-correlation**

This is also the conceptual bridge to later methods:

- **Gradient Boosting (09)** also uses trees, but reduces bias sequentially instead of reducing variance by averaging
- **Stacking** combines different model families rather than many copies of one family
- **Deep ensembles** in neural nets reuse the same logic: multiple diverse models can generalize better than one

---

## 4. Pros and Cons

| Pros | Cons |
|------|------|
| Much lower variance than a single tree | Less interpretable than one tree |
| Strong default model for tabular data | Larger memory and slower inference |
| Handles non-linearities and interactions well | Cannot extrapolate outside observed target range |
| Little preprocessing needed | Feature importance can be misleading if used carelessly |
| Built-in OOB validation | Often beaten by gradient boosting on structured tabular tasks |
| Parallelizable training | Standard bagging is unsafe for time-ordered financial data |

---

## 5. Connections

### From Decision Trees (03)

Decision trees overfit because they are unstable. Random Forest keeps the expressive tree structure but suppresses the instability by averaging many de-correlated trees.

### To Quant Finance

- useful for **feature ranking** and **alpha signal screening**
- captures interactions without heavy feature engineering
- robust when you want a strong baseline quickly

But there is a major warning:

> Standard bootstrap sampling breaks time order.

For finance, use time-aware validation schemes such as walk-forward evaluation or purged cross-validation. Otherwise, performance estimates can be badly inflated.

### To Gradient Boosting (09)

Random Forest mainly reduces **variance**.
Gradient Boosting mainly reduces **bias**.

That contrast is the main conceptual difference to carry forward.

---

## 6. Key Equations Summary

| Equation | Name |
|----------|------|
| $\mathrm{Var}_{ens} = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$ | Ensemble variance |
| $\left(1 - \frac{1}{n}\right)^n \to \frac{1}{e}$ | OOB fraction |
| $\hat{f}_{ens}(x) = \frac{1}{M}\sum_{m=1}^M \hat{f}^{(m)}(x)$ | Ensemble average |
| $m \approx \sqrt{p}$ (classification), $m \approx p/3$ (regression) | Feature subset heuristic |

---

## 7. What To Remember

If you remember only one sentence, remember this:

> Random Forest works because it averages many strong but unstable trees, and random feature subsets stop those trees from all making the same mistakes.

