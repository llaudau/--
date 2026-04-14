# Machine Learning Learning Roadmap

A structured, progressive journey from classical ML to modern deep learning architectures.
Special focus on **L-CNN for Lattice QCD** and **Quantitative Finance / Trading Strategy** applications.

---

## Part 0: Introduction — The ML Mindset

*Read this before studying any algorithm. It provides the mental framework that ties everything together.*

### 0.1 What Is ML Actually Trying to Solve?

**The fundamental problem**: Given some data, learn a function $f: X \to Y$ that **generalizes** to new, unseen data.

The keyword is **generalize**. This is what separates ML from physics:

- **Physics**: You have a known law ($F=ma$). Data is used to *measure* parameters of that law.
- **ML**: You don't know the law. Data is used to *discover* the function, hoping it works on data you haven't seen.

That single sentence — "learn a function that generalizes" — is the seed from which everything in ML grows.

### 0.2 The Three Ingredients of Any ML Algorithm

Every ML algorithm answers the same three questions:

| Ingredient | Question | Example (Linear Regression) |
|---|---|---|
| **1. Hypothesis space** $\mathcal{H}$ | What family of functions can $f$ be? | All linear functions $f(x) = w^Tx + b$ |
| **2. Loss function** $L$ | How do we measure if a function is good? | MSE: $\sum (y_i - f(x_i))^2$ |
| **3. Optimization** | How do we find the best $f$ in $\mathcal{H}$? | Normal equation or gradient descent |

When you encounter any new algorithm, **always ask these three questions**. Once you can answer them, you understand the algorithm.

### 0.3 The Central Tension: Bias vs Variance

This is the **single most important concept in ML**. Everything else is a response to this problem.

$$\text{Test Error} = \underbrace{\text{Bias}^2}_{\text{model too simple}} + \underbrace{\text{Variance}}_{\text{model too sensitive to data}} + \underbrace{\text{Noise}}_{\text{irreducible}}$$

- **High bias** = your hypothesis space is too small to contain the truth (linear model trying to fit sin(x))
- **High variance** = your hypothesis space is so flexible it memorizes the training data instead of learning the pattern (degree-15 polynomial)

**Every advance in ML history is essentially an attempt to manage this tradeoff.**

### 0.4 The Core to Remember for Each Algorithm

When studying any algorithm, distill it down to **5 things**:

| # | Question | Why it matters |
|---|---|---|
| 1 | **What is the hypothesis space?** (What functions can it represent?) | Tells you what problems it can/can't solve |
| 2 | **What is the loss function?** | Tells you what it's optimizing for |
| 3 | **How is it optimized?** (Closed-form? GD? Greedy?) | Tells you about training cost and behavior |
| 4 | **How does it manage bias-variance?** (Regularization, ensembling, etc.) | Tells you why it generalizes |
| 5 | **What's the key intuition / "trick"?** | The one-line summary |

Fill this out for every algorithm as you learn it. By algorithm 28, you'll have a coherent map of the entire field.

### 0.5 The Master Mental Model

```
                        ML PROBLEM
                            |
                            v
              +-----------------------------+
              |  Function f: X -> Y         |
              |  that generalizes well      |
              +-------------+---------------+
                            |
            +---------------+---------------+
            v               v               v
       Hypothesis        Loss        Optimization
        space H         L(f, data)    procedure
            |               |               |
            +-------+-------+-------+-------+
                    v               v
              Bias-Variance    Generalization
               Tradeoff         (test error)
                    |               |
                    +-------+-------+
                            v
                  Regularization /
                  Model Selection /
                  Validation
```

**Every ML algorithm is a different choice for the boxes in this diagram.**

---

## Part I: Why So Many Algorithms? — A Historical Development

ML is **not** a random list of algorithms. It's a chain of "X has a problem -> invent Y to fix it -> Y has its own problem -> invent Z..." Here's the actual story.

### Era 1: Statistical Origins (1800s-1960s)

**Question**: "Can we describe data with a simple formula?"

- **Linear Regression** (Gauss, 1809) — the original. Assumes $y$ is a linear function of $x$.
- **Logistic Regression** (Berkson, 1944) — but what if $y$ is binary (yes/no)? Wrap linear in a sigmoid.
- **Naive Bayes** — what if features are categorical (text)? Use Bayes' theorem with an independence shortcut.

**Limitation**: All assume some specific form (linear, Gaussian, independent features). Real data isn't like that.

### Era 2: Non-parametric & Geometric (1960s-1990s)

**Question**: "Can we drop the assumptions about data distribution?"

- **KNN** (1951) — Don't model anything! Just remember the data. Predict like the closest neighbors.
- **Decision Trees** (CART, 1984) — Split the feature space recursively. Captures non-linearity, no scaling needed.
- **SVM** (1992) — But trees are unstable. SVMs find the *maximum margin* boundary, with kernels for non-linearity.

**Limitation**: KNN dies in high dimensions; trees overfit; SVMs are slow on big data.

### Era 3: Ensemble Methods (1990s-2010s)

**Question**: "If one model is unstable, can many models together be better?"

Key insight: **average many weak models to get a strong one**.

- **Bagging / Random Forest** (Breiman, 1996/2001) — Train many trees on bootstrap samples, average. Reduces **variance**.
- **Boosting / Gradient Boosting / XGBoost** (1997-2016) — Train models sequentially, each correcting the previous. Reduces **bias**. **Still the king of tabular data today.**

**Limitation**: Trees can't handle raw images, audio, or text well. They need hand-crafted features.

### Era 4: Representation Learning / Deep Learning (2010s-now)

**Question**: "Can the model learn its own features instead of needing humans to design them?"

This is the deep learning revolution.

- **MLPs** — universal approximator, but hard to train.
- **Backprop + GPUs + ReLU + Dropout** (2012) — finally made deep nets trainable.
- **CNNs** — exploit spatial structure (translation symmetry) in images.
- **L-CNN** — exploit gauge symmetry in lattice fields (LQCD application!).
- **RNN/LSTM** — exploit temporal structure in sequences.
- **Transformer** (2017) — replace recurrence with attention. Now dominates NLP, vision, and more.
- **Diffusion models, Normalizing flows** — generate new samples, not just predict.

**The big idea of deep learning**: Don't engineer features by hand. Let the model learn a *representation* of the data layer by layer.

### The Pattern

Every era is a response to the limitations of the previous one. **The history is a logical sequence, not a list.**

---

## Part II: How Do We Judge if an Algorithm Is "Good"?

### The Validation Framework

```
Total Data
   +-- Training set     -> Fit the model
   +-- Validation set   -> Tune hyperparameters (e.g., choose lambda for Ridge)
   +-- Test set         -> Final unbiased evaluation (touch ONLY ONCE)
```

**Cross-validation**: Rotate which fold is the validation set, average the results. Standard: 5-fold or 10-fold CV.

### Metrics by Task Type

| Task | Common metrics | What they mean |
|---|---|---|
| **Regression** | MSE, MAE, $R^2$ | Average prediction error |
| **Binary classification** | Accuracy, Precision, Recall, F1, ROC-AUC | Tradeoffs between false positives / negatives |
| **Multi-class** | Accuracy, confusion matrix, log-loss | Per-class performance |
| **Ranking** | NDCG, MAP | Are top predictions correct? |
| **Generative** | Likelihood, FID, sample quality | How realistic are generated samples? |

### Quant-specific Evaluation

Standard ML metrics aren't enough for trading:
- **Sharpe ratio**: return / volatility
- **Maximum drawdown**: worst peak-to-trough loss
- **Out-of-sample performance** with proper time-series CV (purged k-fold)

### The Most Important Principle

> **A model is only as good as its performance on data it has never seen, in conditions similar to deployment.**

Train accuracy means nothing. A degree-15 polynomial has zero train error and is useless.

### The Three Truths

1. **The goal is generalization, not fitting training data.** Always evaluate on held-out data.
2. **Every algorithm trades bias for variance.** Different algorithms make this tradeoff differently.
3. **No Free Lunch Theorem**: No single best algorithm for all problems. The right choice depends on data structure, sample size, and the task. *That* is why so many algorithms exist.

---

## Part III: Mathematical Foundations for ML

### The Four Pillars

#### Pillar 1: Linear Algebra (most important)

Everything in ML is a matrix operation. A neural network is just repeated matrix multiplications with non-linearities:

$$f(x) = \sigma(W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2) + b_3)$$

**Key concepts**: Matrix multiplication as linear maps, eigendecomposition ($A = V\Lambda V^T$), SVD ($A = U\Sigma V^T$), positive semi-definite matrices, rank and null space.

#### Pillar 2: Calculus & Optimization

In physics you minimize action ($\delta S = 0$). In ML you minimize loss functions. Same math, different objects.

**Key concepts**:
- **Gradient + chain rule = backpropagation**. If you understand $\frac{\partial}{\partial x}f(g(h(x))) = f' \cdot g' \cdot h'$, you understand how neural networks train.
- **Convexity**: Convex loss = one minimum (linear models). Non-convex = many local minima (neural nets).
- **Gradient descent variants**: SGD, momentum, Adam — the "solvers" of ML.

#### Pillar 3: Probability & Statistics

In physics experiments, you compute confidence intervals and $\chi^2$. In ML, the usage is different:

| Physics statistics | ML statistics |
|---|---|
| Confidence intervals for parameters | Rarely — we care about prediction, not parameter values |
| $\chi^2$ test, p-values | Rarely — we use train/test split instead |
| Error propagation | Backpropagation (same chain rule, different purpose) |
| Gaussian error model | Many distributions: Bernoulli, Categorical, mixtures |
| MLE for 2-3 params | MLE for millions of params (= training a neural net!) |

**Key concepts**:
- **MLE = minimizing loss**: Linear regression + Gaussian noise -> MLE = minimize MSE. Logistic regression -> MLE = minimize cross-entropy. All of training is just MLE.
- **Bayes' theorem** as a way of thinking: prior + data = posterior. Regularization IS adding a prior!
- **Expectation and Variance**: For bias-variance tradeoff and why SGD works.

#### Pillar 4: Information Theory

$$H(p) = -\sum_i p_i \log p_i \quad \text{(entropy = uncertainty)}$$

$$H(p,q) = -\sum_i p_i \log q_i \quad \text{(cross-entropy = the standard classification loss)}$$

$$D_{KL}(p\|q) = \sum_i p_i \log \frac{p_i}{q_i} \quad \text{(KL divergence = distance between distributions)}$$

Critical for understanding classification loss, VAEs, normalizing flows, and diffusion models.

### Top 10 Mathematical Ideas for ML (ranked by importance)

| Rank | Concept | Where it appears |
|---|---|---|
| 1 | **Chain rule / backprop** | All of deep learning |
| 2 | **Gradient descent** | Training everything |
| 3 | **Matrix multiplication** | Every layer, every transformation |
| 4 | **MLE = minimizing loss** | All supervised learning |
| 5 | **Bias-variance tradeoff** | Model selection, regularization |
| 6 | **Eigendecomposition / SVD** | PCA, spectral methods |
| 7 | **Cross-entropy / KL divergence** | Classification, VAE, flows |
| 8 | **Convexity** | Linear models vs neural nets |
| 9 | **Bayes' theorem** | Bayesian methods, regularization intuition |
| 10 | **Monte Carlo estimation** | SGD, MCMC, generative models |

### New Math Introduced by Each Algorithm

The math builds on itself — each algorithm adds one or two new tools:

| Algorithm | New math concept |
|---|---|
| 01 Linear Regression | Normal equation, gradient descent, convexity |
| 02 Logistic Regression | MLE, cross-entropy, sigmoid |
| 03 Decision Tree | Information entropy, greedy algorithms |
| 05 SVM | Lagrangian duality, kernels, RKHS |
| 08 PCA | Eigendecomposition, SVD |
| 10 MLP | Chain rule = backpropagation |
| 13 CNN | Convolution as an operation |
| 14 L-CNN | Group theory + convolutions (from gauge theory!) |
| 18 VAE | KL divergence, variational inference |
| 20 Normalizing Flow | Change of variables, Jacobian determinants |
| 22 Attention | Softmax, scaled dot-product |

---

## How to Use This Repository

Each numbered folder contains:
- `README.md` - Mathematical foundation, statistical background, pros/cons
- `plan.md` - Sub-plan and learning steps for that algorithm
- `examples/` - Python implementations with practical examples

Study in order (01 -> 28). Each builds on prior knowledge.

**For each algorithm, fill out this card:**

| Question | Answer |
|---|---|
| Hypothesis space? | *(what functions can it represent?)* |
| Loss function? | *(what does it optimize?)* |
| Optimization method? | *(how does it find the best function?)* |
| Bias-variance management? | *(how does it generalize?)* |
| Key trick? | *(one-line summary)* |

---

## Roadmap Overview

### Level 1: Classical ML Foundations
| # | Algorithm | Quant Relevance |
|---|-----------|-----------------|
| 01 | [Linear Regression](01_Linear_Regression/) | Factor models, baseline |
| 02 | [Logistic Regression](02_Logistic_Regression/) | Classification signals |
| 03 | [Decision Tree](03_Decision_Tree/) | Rule-based strategies |
| 04 | [Random Forest & Ensemble](04_Random_Forest_Ensemble/) | Feature importance, alpha |
| 05 | [SVM](05_SVM/) | Non-linear classification |
| 06 | [KNN](06_KNN/) | Pattern matching |
| 07 | [Naive Bayes](07_Naive_Bayes/) | Sentiment classification |
| 08 | [PCA](08_PCA/) | Risk factor decomposition |
| 09 | [Gradient Boosting](09_Gradient_Boosting/) | Top quant competition model |

### Level 2: Neural Network Basics
| # | Algorithm | Key Concept |
|---|-----------|-------------|
| 10 | [MLP](10_MLP/) | Universal approximation |
| 11 | [Backprop & Optimizers](11_Backprop_Optimizers/) | SGD, Adam, learning rates |
| 12 | [Regularization](12_Regularization/) | Dropout, BatchNorm, L1/L2 |

### Level 3: Convolutional Neural Networks
| # | Algorithm | LQCD / Quant Relevance |
|---|-----------|------------------------|
| 13 | [CNN Fundamentals](13_CNN/) | Foundation for L-CNN |
| 14 | **[L-CNN](14_L-CNN/)** | **Gauge-equivariant CNNs for lattice QCD** |
| 15 | [ResNet](15_ResNet/) | Deep architectures, skip connections |

### Level 4: Sequence & Temporal Models
| # | Algorithm | Quant Relevance |
|---|-----------|-----------------|
| 16 | [RNN/LSTM/GRU](16_RNN_LSTM/) | Time-series forecasting |
| 17 | [TCN](17_TCN/) | Efficient sequence modeling |

### Level 5: Generative Models
| # | Algorithm | LQCD / Quant Relevance |
|---|-----------|------------------------|
| 18 | [Autoencoder/VAE](18_Autoencoder_VAE/) | Config compression / Anomaly detection |
| 19 | [GAN](19_GAN/) | Gauge field generation / Synthetic data |
| 20 | **[Normalizing Flow](20_Normalizing_Flow/)** | **Trivializing maps for lattice sampling** |
| 21 | [Diffusion Model](21_Diffusion_Model/) | Lattice config generation |

### Level 6: Attention & Transformers
| # | Algorithm | Key Application |
|---|-----------|-----------------|
| 22 | [Attention Mechanism](22_Attention/) | Foundation for Transformer |
| 23 | [Transformer](23_Transformer/) | NLP, sequence modeling |
| 24 | [Vision Transformer](24_ViT/) | Lattice as image patches |

### Level 7: Reinforcement Learning (Quant Focus)
| # | Algorithm | Quant Relevance |
|---|-----------|-----------------|
| 25 | [Q-Learning/DQN](25_RL_DQN/) | **Portfolio optimization, execution** |
| 26 | [Policy Gradient](26_RL_Policy_Gradient/) | **Trading strategy optimization** |

### Level 8: Special Topics
| # | Algorithm | LQCD Relevance |
|---|-----------|----------------|
| 27 | [GNN](27_GNN/) | **Lattice as graph structure** |
| 28 | [PINN](28_PINN/) | **Embedding physics constraints** |

---

## Key Algorithms for LQCD
- **14_L-CNN**: Gauge-equivariant convolutions on lattice gauge fields
- **20_Normalizing_Flow**: Trivializing maps for efficient lattice sampling
- **27_GNN**: Treating the lattice as a graph
- **28_PINN**: Physics-informed loss functions

## Key Algorithms for Quant
- **09_Gradient_Boosting**: XGBoost/LightGBM for tabular alpha signals
- **16_RNN_LSTM**: Time-series prediction
- **23_Transformer**: NLP for news/sentiment, factor modeling
- **25_RL_DQN** & **26_RL_Policy_Gradient**: Portfolio/execution optimization
