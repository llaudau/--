# Sub-Plan: Naive Bayes

## Status: Not Started

---

## 0. Algorithm Identity Card (from Introduction framework)

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Generative model: $P(y|x) \propto P(x|y)P(y)$ with conditionally independent features |
| **Loss function** | Implicit: maximize posterior probability (equivalent to minimizing 0-1 loss under Bayes-optimal rule) |
| **Optimization** | Counting! Parameters are estimated by counting frequencies in training data. No iterative optimization. |
| **Bias-variance management** | High bias (independence assumption is almost always wrong), very low variance (few parameters). Laplace smoothing prevents zero-probability issues. |
| **Key trick** | Independence assumption reduces parameter estimation from $O(d^K)$ to $O(dK)$ — makes high-dimensional problems tractable. Despite wrong assumptions, class *ranking* is often correct. |

**Historical era**: Era 1 — Statistical Origins (Bayes 1763, applied to ML in 1990s). A generative model based purely on probability theory.

**Generalization perspective**: Ng & Jordan (2001) proved: NB converges to its (biased) asymptotic error with $O(\log n)$ samples, while logistic regression needs $O(n)$. NB wins with small data, LR wins with large data. This is the bias-variance tradeoff in action — NB's strong assumption (high bias) means it needs less data (low variance).

**What problem does this solve?** Extremely fast, works with very small datasets, handles high-dimensional sparse data (text) naturally. The go-to baseline for text classification.

**What problem does it create?** Cannot learn feature interactions. Probabilities are poorly calibrated. Outperformed by discriminative models when data is abundant. -> Motivates logistic regression (02) as the discriminative counterpart.

**New math introduced**: Bayes' theorem as a computational tool, generative vs discriminative models (a fundamental dichotomy), Laplace smoothing, connection between Bayesian priors and regularization.

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

- **Bayes' Theorem**:
  $P(C_k | x) = \frac{P(x | C_k) P(C_k)}{P(x)}$
  - Posterior $\propto$ Likelihood $\times$ Prior
  - $P(x)$ is a normalizing constant (same for all classes)
- **Naive assumption**: Features are conditionally independent given the class
  $P(x | C_k) = \prod_{j=1}^d P(x_j | C_k)$
  - This simplifies estimation from exponential to linear in $d$
  - Almost always violated in practice — but still works surprisingly well!
- **Decision rule**: $\hat{y} = \arg\max_k P(C_k) \prod_j P(x_j | C_k)$
  - In log space: $\hat{y} = \arg\max_k [\log P(C_k) + \sum_j \log P(x_j | C_k)]$

#### Variants by feature distribution

- **Gaussian Naive Bayes**: Continuous features
  - $P(x_j | C_k) = \frac{1}{\sqrt{2\pi \sigma_{jk}^2}} \exp\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)$
  - Parameters: mean $\mu_{jk}$ and variance $\sigma_{jk}^2$ per feature per class
- **Multinomial Naive Bayes**: Count/frequency features (text classification)
  - $P(x_j | C_k) = \frac{N_{jk} + \alpha}{N_k + \alpha d}$ (with Laplace smoothing $\alpha$)
  - Models word counts in documents
- **Bernoulli Naive Bayes**: Binary features
  - $P(x_j | C_k) = p_{jk}^{x_j} (1-p_{jk})^{1-x_j}$
  - Models presence/absence of features
- **Complement Naive Bayes**: Handles imbalanced classes better
  - Uses complement of each class for parameter estimation

### 1.2 Statistical Foundation

- **Generative vs Discriminative models**:
  - Naive Bayes is **generative**: models $P(x|C_k)$ and $P(C_k)$, then uses Bayes rule
  - Logistic Regression is **discriminative**: directly models $P(C_k|x)$
  - Ng & Jordan (2001): NB converges faster with less data, but LR has lower asymptotic error
- **Why "Naive" works**: Even if probability estimates are biased, the ranking of classes can still be correct (Domingos & Pazzani, 1997)
- **Calibration**: NB probabilities are often poorly calibrated (pushed toward 0 and 1)
  - Use Platt scaling or isotonic regression to calibrate
- **Laplace smoothing**: $\alpha > 0$ prevents zero probabilities for unseen feature values
  - $\alpha = 1$: Laplace smoothing
  - $\alpha < 1$: Lidstone smoothing
- **Connection to log-linear models**: Log of Naive Bayes posterior is linear in features (for exponential family distributions)

### 1.3 Historical Context

- Thomas Bayes (1763) — Bayes' theorem
- Used in spam filtering since 1990s (Paul Graham, 2002: "A Plan for Spam")
- Baseline for text classification, medical diagnosis

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Extremely fast training and prediction | Independence assumption rarely holds |
| Works well with small training sets | Poor probability calibration |
| Handles high-dimensional data (text) | Cannot learn feature interactions |
| Robust to irrelevant features | Continuous features: Gaussian assumption may not hold |
| Good baseline for text classification | Outperformed by discriminative models with enough data |
| Easy to update with new data (online learning) | Zero-frequency problem without smoothing |

### 1.5 Connections

- **To Quant**: Sentiment analysis on financial news/reports (Multinomial NB on TF-IDF), spam/fraud detection, quick baseline for any classification task
- **To later topics**: Bayesian thinking reappears in Bayesian Neural Networks; Generative model framework connects to VAE (18) and normalizing flows (20)

---

## 2. Planned Examples

### Example 1: `examples/text_classification/`
**Goal**: Spam detection or 20-newsgroup classification with Multinomial NB
- Implement Multinomial Naive Bayes from scratch
- Text preprocessing: tokenization, TF-IDF (or bag of words)
- Train on spam dataset or 20 newsgroups
- Compare with sklearn.naive_bayes.MultinomialNB
- Show effect of Laplace smoothing parameter
- **Files**: `main.py`, `naive_bayes_from_scratch.py`

### Example 2: `examples/financial_news_sentiment/`
**Goal**: Classify financial news headlines as positive/negative for market
- Use financial news dataset (or synthetic headlines with labeled sentiment)
- Features: TF-IDF of headline text
- Train Gaussian NB (on numeric features) and Multinomial NB (on text)
- Evaluate: accuracy, precision, recall on held-out data
- Discuss: how sentiment scores could feed into a trading signal
- **Files**: `main.py`, `sentiment_classifier.py`, `data/`

---

## 3. Learning Steps

1. **Theory**: Understand Bayes' theorem and the naive independence assumption
   - Work through a simple example by hand (e.g., spam with 3 words)
   - Understand why it's a generative model
2. **Implement from scratch**: Multinomial NB + Gaussian NB with numpy
   - Parameter estimation (counting + smoothing)
   - Log-space computation to avoid underflow
3. **Example 1**: Text classification
4. **Example 2**: Financial news sentiment (quant application)
5. **Reflect**: When does NB outperform more complex models? (Small data, high-dimensional text)

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $P(C_k|x) \propto P(C_k) \prod_j P(x_j|C_k)$ | Naive Bayes decision rule |
| $P(x_j|C_k) = \frac{N_{jk} + \alpha}{N_k + \alpha d}$ | Multinomial NB with smoothing |
| $P(x_j|C_k) = \mathcal{N}(\mu_{jk}, \sigma_{jk}^2)$ | Gaussian NB |

---

## 5. Prerequisites

- Probability: Bayes' theorem, conditional probability, independence
- Logistic Regression (02): for comparison (generative vs discriminative)
