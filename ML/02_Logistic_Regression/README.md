# Logistic Regression

## 0. Algorithm Identity Card

| Question | Answer |
|----------|--------|
| **Hypothesis space** | Linear decision boundary with sigmoid: $f(x) = \sigma(w^Tx + b)$ |
| **Loss function** | Binary cross-entropy: $-\frac{1}{n}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ |
| **Optimization** | Gradient descent (no closed-form) |
| **Bias-variance management** | L2/L1 regularization on weights |
| **Key trick** | Sigmoid maps linear output to probability; cross-entropy is convex |

---

## 1. From Linear Regression to Classification

Linear regression predicts any real number. But what if the target is binary — yes/no, up/down, spam/not-spam?

Naive approach: fit $f(x) = w^Tx + b$ and threshold at 0.5. Problem: the output can be -3.7 or 42.1, which are meaningless as probabilities.

**Solution**: Wrap the linear output in a **sigmoid function** that squashes everything to $(0, 1)$:

$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

Now the output is a valid probability. This is logistic regression.

---

## 2. Mathematical Foundation

### 2.1 The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
- Maps $\mathbb{R} \to (0, 1)$ — valid probability
- $\sigma(0) = 0.5$ — the decision boundary
- $\sigma(-z) = 1 - \sigma(z)$ — symmetric
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ — this makes gradient computation elegant
- Monotonically increasing — preserves ordering

### 2.2 The Log-Odds (Logit) Interpretation

Inverting the sigmoid:

$$\log \frac{p}{1-p} = w^Tx + b$$

The left side is the **log-odds** (logit). This says: logistic regression models the log-odds as a linear function of features. Each coefficient $w_j$ tells you: a one-unit increase in $x_j$ changes the log-odds by $w_j$.

This is why it's called logistic *regression* — the relationship between features and log-odds is linear, just like in linear regression.

### 2.3 Decision Boundary

The model predicts class 1 when $P(y=1|x) > 0.5$, i.e., when $w^Tx + b > 0$.

The decision boundary is the hyperplane $w^Tx + b = 0$ — exactly a linear boundary in feature space. Logistic regression is a **linear classifier**.

### 2.4 Loss Function: Binary Cross-Entropy

#### What is MSE and why not use it here?

MSE (Mean Squared Error) is the average of squared residuals — the same $\chi^2$-like quantity from physics lab (but with equal weights):

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

MSE works well for regression (linear regression uses it). But for classification with sigmoid output, MSE has two problems:

1. **Non-convex landscape**: MSE + sigmoid creates multiple local minima. Gradient descent may get stuck. Cross-entropy + sigmoid is **convex** — guaranteed global minimum.

2. **Weak gradient signal**: When the model is confidently wrong (true label $y=1$, predicted $p=0.01$):
   - MSE loss: $(1 - 0.01)^2 = 0.98$ — moderate penalty
   - Cross-entropy loss: $-\log(0.01) = 4.6$ — severe penalty

   Cross-entropy punishes confident mistakes exponentially via $-\log(p)$. MSE is too "forgiving" — the gradient is small when the model is confident, so it learns slowly from its worst mistakes. Cross-entropy keeps pushing hard until the model fixes them.

**Derivation from Maximum Likelihood**:

Each data point $(x_i, y_i)$ with $y_i \in \{0, 1\}$ has likelihood:

$$P(y_i | x_i; w) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

where $\hat{p}_i = \sigma(w^T x_i + b)$.

Log-likelihood over all data:

$$\ell(w) = \sum_{i=1}^n [y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)]$$

Negative log-likelihood = **binary cross-entropy loss**:

$$\mathcal{L}(w) = -\frac{1}{n} \sum_{i=1}^n [y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i)]$$

**Key insight**: Minimizing cross-entropy = Maximum Likelihood Estimation. This connects probability theory to optimization — the same idea powers all of deep learning.

### 2.5 Gradient

$$\nabla_w \mathcal{L} = \frac{1}{n} X^T(\hat{p} - y)$$

where $\hat{p} = \sigma(Xw + b)$.

Notice: **this has the same form as linear regression's gradient** $\nabla \mathcal{L} = \frac{1}{n} X^T(\hat{y} - y)$. The only difference is that $\hat{p}$ passes through a sigmoid. This elegance is not a coincidence — both are Generalized Linear Models (GLMs).

### 2.6 Why Cross-Entropy Is Convex

The Hessian of the cross-entropy loss is:

$$H = \frac{1}{n} X^T D X, \quad D = \text{diag}(\hat{p}_i(1-\hat{p}_i))$$

Since $0 < \hat{p}_i < 1$, all diagonal entries of $D$ are positive, so $H$ is positive semi-definite. **The loss is convex** — gradient descent is guaranteed to find the global minimum. No worry about local minima (unlike neural networks).

### 2.7 Regularization

Same as linear regression, but even more important because logistic regression can suffer from **perfect separation** — when a hyperplane perfectly separates the classes, the weights diverge to $\pm\infty$.

$$\mathcal{L}_{reg} = \mathcal{L} + \lambda\|w\|_2^2 \quad \text{(L2/Ridge)}$$

$$\mathcal{L}_{reg} = \mathcal{L} + \lambda\|w\|_1 \quad \text{(L1/Lasso — sparse feature selection)}$$

In sklearn, the parameter `C` = $1/\lambda$ (inverse regularization). Smaller `C` = stronger regularization.

### 2.8 Multi-Class Extension: Softmax

Binary logistic regression handles two classes: $P(y=1|x) = \sigma(w^Tx + b)$, and $P(y=0|x) = 1 - P(y=1|x)$. What if you have $K$ classes (e.g., digit recognition 0-9, or market regime: bull/bear/sideways)?

#### Step 1: Give each class its own weights

Each class $k$ gets its own weight vector $w_k$ and bias $b_k$. Compute a "score" (logit) for each class:

$$z_k = w_k^T x + b_k, \quad k = 1, 2, \ldots, K$$

#### Step 2: Convert scores to probabilities with softmax

$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

Properties:
- All probabilities are positive ($e^{z_k} > 0$ always)
- They sum to 1: $\sum_k P(y=k|x) = 1$
- Largest logit gets the highest probability
- For $K=2$, softmax reduces to sigmoid (verify: $\frac{e^{z_1}}{e^{z_1} + e^{z_0}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)$)

#### Connection to statistical mechanics

The softmax IS the **Boltzmann distribution**:

$$P(\text{state}_k) = \frac{e^{-\beta E_k}}{Z}, \quad Z = \sum_j e^{-\beta E_j}$$

The logits $z_k$ play the role of $-\beta E_k$. The denominator $\sum e^{z_j}$ is the partition function $Z$. If you've done stat mech, you already understand softmax — the model assigns probability to each class (state) based on its negative energy (logit). Lower energy = higher probability = the model thinks this class is more likely.

#### Loss: Categorical Cross-Entropy

For one-hot encoded labels (e.g., $y = [0, 0, 1]$ for class 2):

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}$$

Since $y_{ik}$ is 1 only for the true class $c_i$, this simplifies to:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \log \hat{p}_{i, c_i}$$

Meaning: for each sample, look at the predicted probability for the **correct** class, and take $-\log$ of it. Confident and correct ($\hat{p} \approx 1$) gives loss $\approx 0$. Confident and wrong ($\hat{p} \approx 0$) gives loss $\to \infty$.

#### Gradient

$$\nabla_{w_k} \mathcal{L} = \frac{1}{n} X^T(\hat{p}_k - y_k)$$

Same elegant form as binary logistic regression — each class weight gets updated by the difference between predicted and actual probability.

#### Decision boundaries

Softmax produces **linear** decision boundaries. Between any two classes $j$ and $k$, the boundary is where $P(y=j|x) = P(y=k|x)$, which gives $w_j^Tx + b_j = w_k^Tx + b_k$ — a hyperplane. See Fig 6 in the visual guide.

#### Why this matters for deep learning

The output layer of virtually every classification neural network is:

$$\text{learned features} \xrightarrow{W, b} \text{logits} \xrightarrow{\text{softmax}} \text{probabilities} \xrightarrow{\text{cross-entropy}} \text{loss}$$

When you study neural networks (algorithm 10), the only new part is what comes *before* the softmax — the learned feature extraction. The output layer and loss function are exactly what you're learning here.

---

## 3. Statistical Foundation

### 3.1 MLE Connection

Training logistic regression = maximizing likelihood = minimizing cross-entropy. This is the bridge between statistics and ML:

| Statistical view | ML view |
|---|---|
| Maximum Likelihood Estimation | Minimizing cross-entropy loss |
| Fisher Information matrix | Hessian of the loss |
| Wald test for coefficients | Not typically used in ML |
| Asymptotic normality of MLE | Not relied on — use cross-validation instead |

### 3.2 Generalized Linear Model (GLM)

Logistic regression is part of the GLM family:

| Component | Linear Regression | Logistic Regression |
|-----------|-------------------|---------------------|
| Distribution | Gaussian | Bernoulli |
| Link function | Identity: $\eta = \mu$ | Logit: $\eta = \log\frac{p}{1-p}$ |
| Mean function | $\mu = \eta$ | $p = \sigma(\eta)$ |
| Loss | MSE | Cross-entropy |

### 3.3 Evaluation Metrics for Classification

Unlike regression ($R^2$, MSE), classification needs specialized metrics:

**Confusion Matrix**:
```
                 Predicted
              |  Pos  |  Neg  |
Actual  Pos   |  TP   |  FN   |
        Neg   |  FP   |  TN   |
```

**Derived metrics**:
- **Accuracy** = $(TP + TN) / (TP + TN + FP + FN)$ — misleading with imbalanced classes!
- **Precision** = $TP / (TP + FP)$ — "of those I predicted positive, how many are correct?"
- **Recall** = $TP / (TP + FN)$ — "of all actual positives, how many did I catch?"
- **F1** = $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — harmonic mean

**ROC Curve and AUC**:

Logistic regression outputs a probability $p \in (0,1)$. To make a yes/no decision, you need a **threshold**: predict class 1 if $p > t$. But what threshold?
- If false negatives are costly (cancer screening): use lower $t$ — catch more positives, accept more false alarms
- If false positives are costly (spam filter): use higher $t$ — be conservative

The ROC curve shows **all possible threshold choices at once**.

For each threshold $t$ from 0 to 1, compute two rates:
- **TPR** (True Positive Rate = Recall) = $TP / (TP + FN)$ — "of all actual positives, how many did I catch?"
- **FPR** (False Positive Rate) = $FP / (FP + TN)$ — "of all actual negatives, how many did I falsely flag?"

Plot TPR (y-axis) vs FPR (x-axis). A good model hugs the top-left corner (high TPR, low FPR).

**AUC** (Area Under the ROC Curve):
- AUC = 0.5: random guessing (diagonal line). Model is useless.
- AUC = 1.0: perfect separation at some threshold.
- AUC = 0.9 (our Fig 5): For 90% of randomly chosen positive/negative pairs, the model assigns higher probability to the positive one.

AUC evaluates the model's **ranking ability** independent of threshold choice. In quant: "does the model rank good stocks above bad stocks?" is more important than predicting exact returns.

*Physics analogy*: This is exactly the signal efficiency vs background rejection curve in particle physics. Low trigger threshold = catch all signal (high TPR) but lots of background (high FPR). High threshold = pure sample (low FPR) but miss real events (low TPR). ROC shows this full tradeoff.

**Calibration**: Are the predicted probabilities meaningful?
- If model says 70% probability for 100 events, about 70 should actually be positive
- Logistic regression is generally well-calibrated (unlike Naive Bayes or Random Forest)

### 3.4 Generalization Theory

**VC dimension** of logistic regression = $p + 1$ (same as linear classifier), where $p$ = number of features.

From the VC bound: $R(f) \leq \hat{R}(f) + O(\sqrt{(p+1)/n})$

Need $n \gg p$ for good generalization. With regularization, the effective VC dimension is smaller (SRM principle from SVM theory applies here too).

---

## 4. Historical Context

- **Berkson (1944)**: Coined "logit", proposed the model for bioassay analysis
- **Cox (1958)**: Formalized as a regression framework
- **1960s-1980s**: Standard tool in epidemiology, social sciences, credit scoring
- **Today**: Still widely used as baseline; softmax layer = logistic regression on top of neural network features

---

## 5. Pros and Cons

| Pros | Cons |
|------|------|
| Outputs calibrated probabilities | Linear decision boundary only |
| Highly interpretable (log-odds per feature) | Cannot capture non-linear patterns without feature engineering |
| Convex loss — guaranteed global optimum | Sensitive to multicollinearity |
| Fast training, scales to large datasets | Performance ceiling on complex tasks |
| Well-understood theory (MLE, GLM, VC) | Requires careful feature selection/engineering |
| Strong baseline — always try it first | Assumes features contribute independently to log-odds |

---

## 6. Connections

### From Linear Regression (01)
Same gradient form, same regularization framework. Only change: sigmoid wraps the output, MSE becomes cross-entropy.

### To Quant Finance
- Predicting market direction (up/down) with probability output
- Credit scoring: $P(\text{default} | \text{features})$
- Signal classification: combining multiple alpha signals into a buy/sell probability
- Coefficient interpretation: which features drive the prediction?

### To Later Topics
- **Softmax (10, 23)**: Multi-class logistic regression is the output layer of virtually every classification neural network
- **Cross-entropy loss**: The standard loss for all classification in deep learning — introduced here
- **MLE = training**: This principle (introduced here) extends to all neural networks
- **GLM framework**: Connects to Poisson regression, ordinal regression, etc.

---

## 7. Key Equations Summary

| Equation | Name |
|----------|------|
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid function |
| $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | Sigmoid derivative |
| $\log\frac{p}{1-p} = w^Tx + b$ | Logit / log-odds |
| $\mathcal{L} = -\frac{1}{n}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | Binary cross-entropy |
| $\nabla_w \mathcal{L} = \frac{1}{n}X^T(\hat{p} - y)$ | Gradient |
| $P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_j e^{w_j^Tx}}$ | Softmax (multi-class) |
