# Sub-Plan: Principal Component Analysis (PCA)

## Status: Not Started

---

## 1. README.md — Theory Content Outline

### 1.1 Mathematical Foundation

- **Goal**: Find orthogonal directions of maximum variance in the data
  - Project $d$-dimensional data onto $k < d$ dimensions while preserving as much variance as possible
- **Centering**: Subtract mean: $\tilde{X} = X - \bar{X}$
- **Covariance matrix**: $\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$

#### Eigendecomposition approach
- Eigendecompose $\Sigma$: $\Sigma = V \Lambda V^T$
  - $V = [v_1, v_2, \ldots, v_d]$: eigenvectors (principal components)
  - $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$: eigenvalues ($\lambda_1 \geq \lambda_2 \geq \ldots$)
- **Principal components**: $v_1$ is the direction of maximum variance, $v_2$ is orthogonal to $v_1$ with max remaining variance, etc.
- **Projection**: $Z = \tilde{X} V_k$ where $V_k = [v_1, \ldots, v_k]$
- **Variance explained**: $\text{ratio}_i = \lambda_i / \sum_j \lambda_j$
- **Reconstruction**: $\hat{X} = Z V_k^T + \bar{X}$ (lossy for $k < d$)

#### SVD approach (more numerically stable)
- $\tilde{X} = U \Sigma_s V^T$ (Singular Value Decomposition)
- Connection: eigenvalues of covariance $= \sigma_i^2 / (n-1)$
- PCA via SVD avoids explicitly forming $\tilde{X}^T\tilde{X}$
- Truncated SVD for large datasets (randomized algorithms)

#### Kernel PCA
- Apply kernel trick to perform PCA in feature space
- $K_{ij} = \phi(x_i)^T \phi(x_j)$
- Eigendecompose kernel matrix $K$ instead of covariance
- Captures non-linear structure (e.g., unrolling Swiss roll)

### 1.2 Statistical Foundation

- **PCA as Maximum Variance**: Maximizes $\text{Var}(Xv) = v^T \Sigma v$ subject to $\|v\| = 1$
  - Lagrangian: $v^T \Sigma v - \lambda(v^T v - 1)$ -> eigenvector equation $\Sigma v = \lambda v$
- **PCA as Minimum Reconstruction Error**: Minimizes $\|X - X V_k V_k^T\|_F^2$
  - Eckart-Young theorem: best rank-$k$ approximation in Frobenius norm
- **Choosing $k$** (number of components):
  - Cumulative variance explained > threshold (e.g., 95%)
  - Scree plot: elbow method
  - Kaiser's rule: keep components with $\lambda > 1$ (for standardized data)
  - Cross-validation on downstream task
- **Assumptions and limitations**:
  - Assumes linear relationships (kernel PCA for non-linear)
  - Sensitive to feature scaling — must standardize first!
  - Directions of max variance may not be most informative (supervised PCA, LDA as alternative)
- **Probabilistic PCA (PPCA)**: Latent variable model
  - $x = W z + \mu + \epsilon$, $z \sim \mathcal{N}(0, I)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$
  - MLE solution recovers PCA directions
  - Connects to Factor Analysis and VAE (18)

### 1.3 Historical Context

- Karl Pearson (1901) — geometric formulation
- Hotelling (1933) — statistical formulation
- Ubiquitous in data analysis, signal processing, genomics, finance

### 1.4 Pros and Cons

| Pros | Cons |
|------|------|
| Reduces dimensionality, removes noise | Linear method (use kernel PCA for non-linear) |
| Removes multicollinearity | Sensitive to feature scaling |
| Fast and well-understood | Max variance != max information for supervised tasks |
| Enables visualization of high-dim data | Components hard to interpret (linear combos of all features) |
| Noise reduction via low-rank approximation | Assumes Gaussian-like distributions for best results |
| No hyperparameters (except $k$) | New data must be projected using training eigenvectors |

### 1.5 Connections

- **To LQCD**: Dimensionality reduction for lattice gauge configurations — each config is a high-dimensional vector; PCA reveals dominant modes of fluctuation; connects to spectral analysis of Dirac operator
- **To Quant**: PCA on return covariance matrix reveals statistical risk factors (1st PC $\approx$ market factor, 2nd $\approx$ sector, etc.); PCA for yield curve modeling (level, slope, curvature); noise reduction in alpha signals
- **To later topics**: PCA is a linear autoencoder (18); Probabilistic PCA connects to VAE; Feature extraction connects to CNN (13) learned features

---

## 2. Planned Examples

### Example 1: `examples/visualization_and_reconstruction/`
**Goal**: PCA on a high-dimensional dataset with visualization
- Implement PCA from scratch (eigendecomposition + SVD)
- Apply to MNIST digits: 784D -> 2D visualization, colored by digit label
- Show variance explained curve and reconstruction at different $k$
- Visualize what each principal component "looks like" (reshape to 28x28)
- Compare with sklearn.decomposition.PCA
- **Files**: `main.py`, `pca_from_scratch.py`

### Example 2: `examples/risk_factor_decomposition/`
**Goal**: PCA on stock return covariance matrix to extract risk factors
- Compute daily returns for a portfolio of stocks (synthetic or real data)
- Compute covariance matrix, apply PCA
- Interpret first few PCs: PC1 = market, PC2 = sector rotation, PC3 = size, etc.
- Show variance explained by each factor
- Use PCA-based factors in a regression model for risk attribution
- Discuss: PCA for yield curve modeling (level, slope, curvature)
- **Files**: `main.py`, `risk_pca.py`, `data/`

### Example 3: `examples/lattice_config_analysis/` (LQCD connection)
**Goal**: PCA on lattice gauge field configurations (simplified)
- Generate toy 2D lattice configurations (e.g., random SU(2) or U(1) matrices flattened)
- Apply PCA to identify dominant modes of variation
- Visualize how top PCs correspond to physical modes
- Discuss connection to eigenvalues of the transfer matrix
- **Files**: `main.py`, `lattice_pca.py`

---

## 3. Learning Steps

1. **Theory**: Derive PCA from maximum variance and minimum reconstruction perspectives
   - Understand eigendecomposition and SVD connection
   - Work through a 2D example by hand
2. **Implement from scratch**: PCA with numpy (both eigen and SVD)
   - Centering, eigendecomposition, projection, reconstruction
3. **Example 1**: MNIST visualization and reconstruction
4. **Example 2**: Risk factor decomposition (quant application)
5. **Example 3**: Lattice configuration analysis (LQCD connection)
6. **Reflect**: When is PCA sufficient? When do we need non-linear methods (kernel PCA, autoencoders)?

---

## 4. Key Equations to Memorize

| Equation | Name |
|----------|------|
| $\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$ | Covariance matrix |
| $\Sigma v = \lambda v$ | Eigenvalue equation |
| $Z = \tilde{X} V_k$ | Projection to $k$ dims |
| Variance ratio $= \lambda_i / \sum_j \lambda_j$ | Variance explained |
| $\tilde{X} = U \Sigma_s V^T$ | SVD |

---

## 5. Prerequisites

- Linear algebra: eigenvalues/eigenvectors, SVD, orthogonality, matrix norms
- Statistics: variance, covariance, correlation
- For LQCD example: basic understanding of lattice gauge fields
