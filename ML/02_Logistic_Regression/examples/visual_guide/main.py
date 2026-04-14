"""
Logistic Regression — Visual Guide
====================================
Generates figures demonstrating core concepts:
  1. Sigmoid function and decision boundary
  2. Why cross-entropy, not MSE
  3. Decision boundary on 2D data (linear vs non-linear data)
  4. Regularization effect on decision boundary
  5. ROC curve and AUC
  6. Softmax: multi-class extension

Run:  python main.py   (use the venv at ML/venv/)
All figures saved to ./figures/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
np.random.seed(42)

PYTHON = "../../venv/bin/python"  # relative hint for users


# ---------------------------------------------------------------------------
# Logistic regression from scratch
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def cross_entropy(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_train(X, y, lr=0.1, n_iters=500, lam=0.0):
    """Train logistic regression with gradient descent. Returns (w, b, loss_history)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(n_iters):
        z = X @ w + b
        p = sigmoid(z)
        losses.append(cross_entropy(y, p))
        dw = (1 / n) * X.T @ (p - y) + 2 * lam * w
        db = (1 / n) * np.sum(p - y)
        w -= lr * dw
        b -= lr * db
    return w, b, losses


def logistic_predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def make_linear_2d(n=200):
    """Linearly separable 2-class data."""
    X0 = np.random.randn(n // 2, 2) + np.array([-1, -1])
    X1 = np.random.randn(n // 2, 2) + np.array([1.5, 1.5])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def make_moons(n=300, noise=0.2):
    """Two interleaving half-circles — non-linearly separable."""
    n0, n1 = n // 2, n - n // 2
    t0 = np.linspace(0, np.pi, n0)
    t1 = np.linspace(0, np.pi, n1)
    X0 = np.column_stack([np.cos(t0), np.sin(t0)]) + np.random.randn(n0, 2) * noise
    X1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5]) + np.random.randn(n1, 2) * noise
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    return X, y


# ===================================================================
# Figure 1: Sigmoid function
# ===================================================================
def fig1_sigmoid():
    z = np.linspace(-8, 8, 300)
    sig = sigmoid(z)
    dsig = sig * (1 - sig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: sigmoid
    ax = axes[0]
    ax.plot(z, sig, "b-", lw=2.5, label=r"$\sigma(z) = \frac{1}{1+e^{-z}}$")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.fill_between(z, 0, sig, where=(z > 0), alpha=0.1, color="blue")
    ax.set_xlabel("z = w$^T$x + b")
    ax.set_ylabel(r"$\sigma(z)$")
    ax.set_title("Sigmoid Function")
    ax.legend(fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.annotate("predict class 1", xy=(3, 0.8), fontsize=11, color="blue")
    ax.annotate("predict class 0", xy=(-7, 0.15), fontsize=11, color="red")
    ax.annotate("decision\nboundary", xy=(0.2, 0.52), fontsize=10, color="gray")

    # Right: sigmoid derivative
    ax = axes[1]
    ax.plot(z, dsig, "r-", lw=2.5, label=r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\sigma'(z)$")
    ax.set_title("Sigmoid Derivative (max at z=0)")
    ax.legend(fontsize=11)

    fig.suptitle("Figure 1: The Sigmoid Function", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_sigmoid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig1_sigmoid.png")


# ===================================================================
# Figure 2: Why cross-entropy, not MSE
# ===================================================================
def fig2_loss_comparison():
    p_pred = np.linspace(0.001, 0.999, 500)

    # For a single sample with y=1
    mse_y1 = (1 - p_pred) ** 2
    ce_y1 = -np.log(p_pred)

    # For a single sample with y=0
    mse_y0 = p_pred ** 2
    ce_y0 = -np.log(1 - p_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(p_pred, mse_y1, "b-", lw=2, label="MSE (y=1)")
    ax.plot(p_pred, ce_y1, "r-", lw=2, label="Cross-Entropy (y=1)")
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Loss")
    ax.set_title("True label y = 1")
    ax.legend()
    ax.set_ylim(0, 5)
    ax.annotate("CE penalizes\nconfident wrong\npredictions much\nmore severely",
                xy=(0.05, 3.5), fontsize=9, color="red")

    ax = axes[1]
    ax.plot(p_pred, mse_y0, "b-", lw=2, label="MSE (y=0)")
    ax.plot(p_pred, ce_y0, "r-", lw=2, label="Cross-Entropy (y=0)")
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Loss")
    ax.set_title("True label y = 0")
    ax.legend()
    ax.set_ylim(0, 5)

    fig.suptitle("Figure 2: Cross-Entropy vs MSE Loss — Why CE is preferred",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_loss_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig2_loss_comparison.png")


# ===================================================================
# Figure 3: Decision boundary — linear data vs non-linear (moons)
# ===================================================================
def _plot_decision_boundary(ax, X, y, w, b, title=""):
    """Plot data and linear decision boundary."""
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = logistic_predict_proba(grid, w, b).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 21), cmap="RdYlBu", alpha=0.6)
    ax.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", s=20, alpha=0.6, edgecolors="k", lw=0.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", s=20, alpha=0.6, edgecolors="k", lw=0.3)
    ax.set_title(title)


def fig3_decision_boundary():
    X_lin, y_lin = make_linear_2d(200)
    X_moon, y_moon = make_moons(300, noise=0.2)

    w_lin, b_lin, _ = logistic_train(X_lin, y_lin, lr=0.5, n_iters=300)
    w_moon, b_moon, _ = logistic_train(X_moon, y_moon, lr=0.5, n_iters=300)

    # Compute accuracies
    pred_lin = (logistic_predict_proba(X_lin, w_lin, b_lin) > 0.5).astype(int)
    acc_lin = np.mean(pred_lin == y_lin)
    pred_moon = (logistic_predict_proba(X_moon, w_moon, b_moon) > 0.5).astype(int)
    acc_moon = np.mean(pred_moon == y_moon)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _plot_decision_boundary(axes[0], X_lin, y_lin, w_lin, b_lin,
                            f"Linearly Separable (acc={acc_lin:.1%})")
    _plot_decision_boundary(axes[1], X_moon, y_moon, w_moon, b_moon,
                            f"Non-linear (Moons) — Linear Fails (acc={acc_moon:.1%})")

    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[1].set_xlabel("x₁")

    fig.suptitle("Figure 3: Logistic Regression Decision Boundary — Strength and Limitation",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_decision_boundary.png")


# ===================================================================
# Figure 4: Gradient descent convergence + comparison with LR gradient form
# ===================================================================
def fig4_training():
    X, y = make_linear_2d(200)

    # Train with different learning rates
    lrs = [0.01, 0.1, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for lr in lrs:
        _, _, losses = logistic_train(X, y, lr=lr, n_iters=200)
        ax.plot(losses, lw=2, label=f"lr = {lr}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Convergence (different learning rates)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: Regularization comparison
    ax = axes[1]
    lambdas = [0, 0.01, 0.1, 1.0]
    for lam in lambdas:
        _, _, losses = logistic_train(X, y, lr=0.5, n_iters=200, lam=lam)
        ax.plot(losses, lw=2, label=f"λ = {lam}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Effect of L2 Regularization")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 4: Training Logistic Regression", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_training.png")


# ===================================================================
# Figure 5: ROC Curve and AUC
# ===================================================================
def fig5_roc_curve():
    # Generate data with some overlap for interesting ROC
    np.random.seed(7)
    n = 300
    X0 = np.random.randn(n // 2, 2) + np.array([-0.5, -0.5])
    X1 = np.random.randn(n // 2, 2) + np.array([1.0, 1.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # Shuffle and split
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]

    w, b, _ = logistic_train(X_train, y_train, lr=0.5, n_iters=300)
    probs = logistic_predict_proba(X_test, w, b)

    # Compute ROC
    thresholds = np.linspace(0, 1, 200)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = np.sum((pred == 1) & (y_test == 1))
        fp = np.sum((pred == 1) & (y_test == 0))
        fn = np.sum((pred == 0) & (y_test == 1))
        tn = np.sum((pred == 0) & (y_test == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

    # AUC by trapezoidal rule (sort by FPR)
    sorted_pairs = sorted(zip(fprs, tprs))
    fprs_s = [p[0] for p in sorted_pairs]
    tprs_s = [p[1] for p in sorted_pairs]
    auc = np.trapezoid(tprs_s, fprs_s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: ROC curve
    ax = axes[0]
    ax.plot(fprs_s, tprs_s, "b-", lw=2.5, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.fill_between(fprs_s, 0, tprs_s, alpha=0.15, color="blue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Right: Probability distribution by class
    ax = axes[1]
    ax.hist(probs[y_test == 0], bins=20, alpha=0.6, color="red", label="Class 0 (negative)",
            density=True, edgecolor="white")
    ax.hist(probs[y_test == 1], bins=20, alpha=0.6, color="blue", label="Class 1 (positive)",
            density=True, edgecolor="white")
    ax.axvline(0.5, color="k", ls="--", lw=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability P(y=1)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Class")
    ax.legend()

    fig.suptitle("Figure 5: ROC Curve and Predicted Probability Distribution",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig5_roc_curve.png")


# ===================================================================
# Figure 6: Softmax — Multi-class decision boundary
# ===================================================================
def fig6_softmax():
    """3-class classification with softmax."""
    np.random.seed(42)
    n_per_class = 100
    # Three clusters
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2, 0])
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([2, 0])
    X2 = np.random.randn(n_per_class, 2) * 0.8 + np.array([0, 2.5])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    # One-hot
    K = 3
    Y_oh = np.zeros((len(y), K))
    Y_oh[np.arange(len(y)), y] = 1

    # Train softmax with gradient descent
    n, d = X.shape
    W = np.zeros((d, K))
    B = np.zeros(K)
    lr = 0.1
    for _ in range(500):
        logits = X @ W + B
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        dW = (1 / n) * X.T @ (probs - Y_oh)
        dB = (1 / n) * (probs - Y_oh).sum(axis=0)
        W -= lr * dW
        B -= lr * dB

    # Decision boundary
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    logits_grid = grid @ W + B
    exp_grid = np.exp(logits_grid - logits_grid.max(axis=1, keepdims=True))
    probs_grid = exp_grid / exp_grid.sum(axis=1, keepdims=True)
    pred_grid = np.argmax(probs_grid, axis=1).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: decision regions
    ax = axes[0]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    cmap = plt.cm.colors.ListedColormap(colors)
    ax.contourf(xx, yy, pred_grid, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.3)
    for k, (c, label) in enumerate(zip(colors, ["Class 0", "Class 1", "Class 2"])):
        ax.scatter(X[y == k, 0], X[y == k, 1], c=c, s=20, alpha=0.7,
                   edgecolors="k", lw=0.3, label=label)
    ax.set_title("Softmax: 3-Class Decision Regions")
    ax.legend()
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    # Right: probability heatmap for one class
    ax = axes[1]
    prob_class2 = probs_grid[:, 2].reshape(xx.shape)
    cs = ax.contourf(xx, yy, prob_class2, levels=20, cmap="Greens")
    fig.colorbar(cs, ax=ax, label="P(class 2)")
    ax.scatter(X[y == 2, 0], X[y == 2, 1], c="#2ecc71", s=20, edgecolors="k", lw=0.3)
    ax.set_title("P(Class 2 | x) — Probability Heatmap")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    fig.suptitle("Figure 6: Softmax — Multi-class Logistic Regression",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_softmax.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig6_softmax.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating Logistic Regression visual guide...\n")
    fig1_sigmoid()
    fig2_loss_comparison()
    fig3_decision_boundary()
    fig4_training()
    fig5_roc_curve()
    fig6_softmax()
    print(f"\nAll figures saved to {FIGDIR.resolve()}")
