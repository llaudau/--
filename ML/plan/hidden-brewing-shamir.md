# ML Learning Roadmap - Master Plan

## Context
Build a comprehensive, self-contained ML learning repository organized from foundational algorithms to advanced architectures. Each algorithm gets its own folder with theory notes (math/stats foundation, background, pros/cons) and at least one practical Python example. Special emphasis on **L-CNN for Lattice QCD** and **Quant/Strategy applications**.

---

## Directory Structure & Algorithm Progression

### Level 1: Foundations (Classical ML)

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 1 | Linear Regression | `01_Linear_Regression/` | - | Baseline factor models |
| 2 | Logistic Regression | `02_Logistic_Regression/` | - | Classification signals |
| 3 | Decision Trees | `03_Decision_Tree/` | - | Rule-based strategies |
| 4 | Random Forest & Ensemble Methods | `04_Random_Forest_Ensemble/` | - | Feature importance, alpha signals |
| 5 | Support Vector Machines | `05_SVM/` | - | Non-linear classification |
| 6 | K-Nearest Neighbors | `06_KNN/` | - | Pattern matching |
| 7 | Naive Bayes | `07_Naive_Bayes/` | - | Sentiment/text classification |
| 8 | Principal Component Analysis (PCA) | `08_PCA/` | Dimension reduction for lattice configs | Risk factor decomposition |
| 9 | Gradient Boosting (XGBoost/LightGBM) | `09_Gradient_Boosting/` | - | Top quant competition model |

### Level 2: Neural Network Basics

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 10 | Feedforward Neural Network (MLP) | `10_MLP/` | Function approximation on lattice observables | Non-linear signal combination |
| 11 | Backpropagation & Optimization (SGD, Adam) | `11_Backprop_Optimizers/` | Training foundation | Training foundation |
| 12 | Regularization (Dropout, BatchNorm, L1/L2) | `12_Regularization/` | Overfitting control | Overfitting control |

### Level 3: Convolutional Neural Networks

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 13 | CNN Fundamentals | `13_CNN/` | Foundation for L-CNN | Image-based features (charts) |
| 14 | **Equivariant / Lattice CNN (L-CNN)** | `14_L-CNN/` | **Core: gauge-equivariant CNNs on lattice gauge fields** | Symmetry-aware architectures |
| 15 | ResNet & Deep CNN Architectures | `15_ResNet/` | Deep feature extraction | - |

### Level 4: Sequence & Temporal Models

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 16 | Recurrent Neural Networks (RNN/LSTM/GRU) | `16_RNN_LSTM/` | Temporal correlations in MC chains | Time-series forecasting |
| 17 | Temporal Convolutional Networks (TCN) | `17_TCN/` | - | Efficient sequence modeling |

### Level 5: Generative Models

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 18 | Autoencoders (AE/VAE) | `18_Autoencoder_VAE/` | Lattice config compression & generation | Anomaly detection |
| 19 | Generative Adversarial Networks (GAN) | `19_GAN/` | Gauge field generation | Synthetic data augmentation |
| 20 | Normalizing Flows | `20_Normalizing_Flow/` | **Key: lattice field sampling, trivializing maps** | Density estimation for returns |
| 21 | Diffusion Models | `21_Diffusion_Model/` | Lattice config generation | Synthetic market data |

### Level 6: Attention & Transformers

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 22 | Attention Mechanism | `22_Attention/` | Foundation for Transformer | Foundation for Transformer |
| 23 | Transformer | `23_Transformer/` | Sequence-to-sequence for lattice observables | Alpha factor modeling, NLP for news |
| 24 | Vision Transformer (ViT) | `24_ViT/` | Lattice field as image patches | Chart/image analysis |

### Level 7: Reinforcement Learning (Quant Focus)

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 25 | Q-Learning & DQN | `25_RL_DQN/` | - | **Portfolio optimization, execution** |
| 26 | Policy Gradient (PPO/A2C) | `26_RL_Policy_Gradient/` | - | **Trading strategy optimization** |

### Level 8: Special Topics

| # | Algorithm | Folder | LQCD Relevance | Quant Relevance |
|---|-----------|--------|----------------|-----------------|
| 27 | Graph Neural Networks (GNN) | `27_GNN/` | **Lattice as graph structure** | Relationship/correlation networks |
| 28 | Physics-Informed Neural Networks (PINN) | `28_PINN/` | **Embedding physics constraints** | - |

---

## Per-Algorithm Folder Structure

Each `XX_AlgorithmName/` folder will contain:

```
XX_AlgorithmName/
├── README.md              # Math/stats foundation, background, pros & cons
├── plan.md                # Sub-plan for learning this algorithm
└── examples/
    └── example_name/      # At least 1 Python example
        ├── main.py
        └── ...
```

The `README.md` for each algorithm will cover:
1. **Mathematical Foundation** - core equations, loss functions, optimization
2. **Statistical Background** - assumptions, distributions, convergence
3. **Historical Context** - when/why it was developed
4. **Pros and Cons** - strengths, weaknesses, when to use/avoid
5. **Relevance to LQCD / Quant** (where applicable)

---

## Implementation Plan (this step)

**What to do now:**
1. Create all 28 top-level algorithm folders
2. Create `examples/` subdirectory in each
3. Create a master `README.md` at `/home/khw/Documents/Git_repository/ML/README.md` with the roadmap overview
4. Create a placeholder `plan.md` in each algorithm folder (sub-plan to be filled when we study each one)

**Verification:** Run `tree` on the ML directory to confirm structure.
