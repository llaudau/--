"""
Linear Regression — Visual Guide
=================================
Generates figures demonstrating core concepts:
  1. OLS fit vs data (simple 1D)
  2. Normal equation vs Gradient Descent convergence
  3. Overfitting with polynomial features
  4. Ridge / Lasso regularization effect
  5. Bias-Variance tradeoff

Run:  python main.py
All figures saved to ./figures/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Helper: generate noisy data from a true function
# ---------------------------------------------------------------------------
def make_data(n=50, noise=0.3):
    """y = 0.8x + 0.3 + noise (true linear relationship)."""
    X = np.sort(np.random.uniform(0, 2, n))
    y = 0.8 * X + 0.3 + np.random.randn(n) * noise
    return X, y


def make_nonlinear_data(n=60, noise=0.25):
    """y = sin(2*pi*x) + noise (true nonlinear relationship)."""
    X = np.sort(np.random.uniform(0, 1, n))
    y = np.sin(2 * np.pi * X) + np.random.randn(n) * noise
    return X, y


# ===================================================================
# Figure 1: Simple OLS fit — the physics-experiment view vs ML view
# ===================================================================
def fig1_ols_fit():
    X, y = make_data()
    # Add bias column
    X_b = np.column_stack([np.ones_like(X), X])
    # Normal equation
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    x_line = np.linspace(0, 2, 100)
    y_line = beta[0] + beta[1] * x_line

    # Residuals
    y_pred = X_b @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: fit with residuals shown
    ax = axes[0]
    ax.scatter(X, y, c="steelblue", s=30, alpha=0.7, label="Data")
    ax.plot(x_line, y_line, "r-", lw=2,
            label=f"OLS: y = {beta[1]:.3f}x + {beta[0]:.3f}")
    for xi, yi, yp in zip(X, y, y_pred):
        ax.plot([xi, xi], [yi, yp], "r-", alpha=0.15, lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"OLS Fit  (R² = {r2:.4f})")
    ax.legend()

    # Right: residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Residual (y - ŷ)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals  (mean={residuals.mean():.4f}, std={residuals.std():.3f})")

    fig.suptitle("Figure 1: OLS Fit and Residual Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_ols_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig1_ols_fit.png")


# ===================================================================
# Figure 2: Gradient Descent convergence vs Normal Equation
# ===================================================================
def fig2_gradient_descent():
    X, y = make_data(n=100)
    X_b = np.column_stack([np.ones_like(X), X])
    n = len(y)

    # Normal equation solution
    beta_exact = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    # Gradient descent
    lr = 0.1
    n_iters = 60
    beta_gd = np.array([0.0, 0.0])  # start at origin
    history = [beta_gd.copy()]
    loss_history = []

    for _ in range(n_iters):
        grad = -(2 / n) * X_b.T @ (y - X_b @ beta_gd)
        beta_gd = beta_gd - lr * grad
        history.append(beta_gd.copy())
        mse = np.mean((y - X_b @ beta_gd)**2)
        loss_history.append(mse)

    history = np.array(history)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: parameter trajectory
    ax = axes[0]
    ax.plot(history[:, 0], history[:, 1], "o-", ms=3, lw=1, color="steelblue",
            alpha=0.7, label="GD path")
    ax.plot(history[0, 0], history[0, 1], "gs", ms=10, label="Start (0, 0)")
    ax.plot(beta_exact[0], beta_exact[1], "r*", ms=15, label="Normal Eq solution")
    ax.set_xlabel("β₀ (intercept)")
    ax.set_ylabel("β₁ (slope)")
    ax.set_title("Parameter Trajectory")
    ax.legend()

    # Right: loss convergence
    ax = axes[1]
    ax.plot(range(1, n_iters + 1), loss_history, "o-", ms=3, lw=1.5,
            color="steelblue")
    mse_exact = np.mean((y - X_b @ beta_exact)**2)
    ax.axhline(mse_exact, color="r", ls="--", lw=1.5, label=f"OLS MSE = {mse_exact:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Loss Convergence  (lr = {lr})")
    ax.legend()
    ax.set_yscale("log")

    fig.suptitle("Figure 2: Gradient Descent vs Normal Equation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_gradient_descent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig2_gradient_descent.png")


# ===================================================================
# Figure 3: Overfitting with polynomial features
# ===================================================================
def fig3_overfitting():
    X, y = make_nonlinear_data(n=20, noise=0.25)
    x_plot = np.linspace(0, 1, 200)

    degrees = [1, 3, 15]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, deg in zip(axes, degrees):
        # Polynomial feature matrix
        X_poly = np.column_stack([X**d for d in range(deg + 1)])
        x_poly_plot = np.column_stack([x_plot**d for d in range(deg + 1)])

        # Fit (use pseudo-inverse for numerical stability)
        beta = np.linalg.pinv(X_poly) @ y
        y_plot = x_poly_plot @ beta

        # Train MSE
        y_pred = X_poly @ beta
        mse = np.mean((y - y_pred)**2)

        ax.scatter(X, y, c="steelblue", s=30, alpha=0.7, zorder=3)
        ax.plot(x_plot, np.sin(2 * np.pi * x_plot), "g--", lw=1.5,
                alpha=0.5, label="True: sin(2πx)")
        ax.plot(x_plot, y_plot, "r-", lw=2, label=f"Degree {deg}")
        ax.set_xlabel("x")
        ax.set_title(f"Degree {deg}  (MSE = {mse:.4f})")
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("y")
    fig.suptitle(
        "Figure 3: Overfitting — Low degree (underfit) vs High degree (overfit)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_overfitting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3_overfitting.png")


# ===================================================================
# Figure 4: Ridge vs Lasso regularization
# ===================================================================
def fig4_regularization():
    X, y = make_nonlinear_data(n=25, noise=0.25)
    deg = 12  # high degree to see regularization effect
    X_poly = np.column_stack([X**d for d in range(deg + 1)])
    x_plot = np.linspace(0, 1, 200)
    x_poly_plot = np.column_stack([x_plot**d for d in range(deg + 1)])

    # Standardize columns (except intercept) for fair regularization
    means = X_poly[:, 1:].mean(axis=0)
    stds = X_poly[:, 1:].std(axis=0) + 1e-10
    X_std = X_poly.copy()
    X_std[:, 1:] = (X_poly[:, 1:] - means) / stds
    x_std_plot = x_poly_plot.copy()
    x_std_plot[:, 1:] = (x_poly_plot[:, 1:] - means) / stds

    lambdas = [0, 1e-4, 1e-2, 1.0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, lam in enumerate(lambdas):
        # Ridge
        I = np.eye(deg + 1)
        I[0, 0] = 0  # don't regularize intercept
        beta_ridge = np.linalg.inv(X_std.T @ X_std + lam * I) @ X_std.T @ y
        y_ridge = x_std_plot @ beta_ridge

        ax = axes[0, i]
        ax.scatter(X, y, c="steelblue", s=20, alpha=0.6, zorder=3)
        ax.plot(x_plot, np.sin(2 * np.pi * x_plot), "g--", lw=1, alpha=0.4)
        ax.plot(x_plot, y_ridge, "r-", lw=2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Ridge  λ = {lam}")
        if i == 0:
            ax.set_ylabel("y")

        # Lasso via coordinate descent
        beta_lasso = _lasso_cd(X_std, y, lam, max_iter=5000)
        y_lasso = x_std_plot @ beta_lasso

        ax = axes[1, i]
        ax.scatter(X, y, c="steelblue", s=20, alpha=0.6, zorder=3)
        ax.plot(x_plot, np.sin(2 * np.pi * x_plot), "g--", lw=1, alpha=0.4)
        ax.plot(x_plot, y_lasso, "r-", lw=2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Lasso  λ = {lam}")
        n_zero = np.sum(np.abs(beta_lasso[1:]) < 1e-6)
        ax.set_xlabel(f"{n_zero}/{deg} coefficients = 0")
        if i == 0:
            ax.set_ylabel("y")

    fig.suptitle(
        "Figure 4: Ridge (L2) vs Lasso (L1) Regularization  —  Degree 12 polynomial",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_regularization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig4_regularization.png")


def _lasso_cd(X, y, lam, max_iter=3000, tol=1e-6):
    """Simple coordinate descent for Lasso."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = y - X @ beta + X[:, j] * beta[j]
            z_j = X[:, j] @ r_j / n
            if j == 0:  # don't regularize intercept
                beta[j] = z_j
            else:
                beta[j] = np.sign(z_j) * max(0, abs(z_j) - lam) / (
                    np.sum(X[:, j] ** 2) / n
                )
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


# ===================================================================
# Figure 5: Bias-Variance Tradeoff — vary model complexity
# ===================================================================
def fig5_bias_variance():
    np.random.seed(0)
    degrees = list(range(1, 16))
    n_train, n_test = 25, 200
    n_runs = 100

    x_test = np.linspace(0, 1, n_test)
    y_true = np.sin(2 * np.pi * x_test)

    train_mses = {d: [] for d in degrees}
    test_mses = {d: [] for d in degrees}

    for _ in range(n_runs):
        X_tr, y_tr = make_nonlinear_data(n=n_train, noise=0.25)
        for deg in degrees:
            X_poly = np.column_stack([X_tr**d for d in range(deg + 1)])
            x_poly_test = np.column_stack([x_test**d for d in range(deg + 1)])

            try:
                beta = np.linalg.pinv(X_poly) @ y_tr
                y_pred_train = X_poly @ beta
                y_pred_test = x_poly_test @ beta

                train_mses[deg].append(np.mean((y_tr - y_pred_train)**2))
                test_mses[deg].append(np.mean((y_true - y_pred_test)**2))
            except Exception:
                train_mses[deg].append(np.nan)
                test_mses[deg].append(np.nan)

    avg_train = [np.nanmean(train_mses[d]) for d in degrees]
    avg_test = [np.nanmedian(test_mses[d]) for d in degrees]  # median is robust

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, avg_train, "o-", lw=2, label="Train MSE", color="steelblue")
    ax.plot(degrees, avg_test, "s-", lw=2, label="Test MSE", color="tomato")
    ax.set_xlabel("Polynomial Degree (model complexity)")
    ax.set_ylabel("MSE")
    ax.set_title("Figure 5: Bias-Variance Tradeoff")
    ax.set_ylim(0, min(1.0, max(avg_test) * 1.2))
    ax.axvline(degrees[np.argmin(avg_test)], color="gray", ls=":", lw=1.5,
               label=f"Optimal degree = {degrees[np.argmin(avg_test)]}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_bias_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig5_bias_variance.png")


# ===================================================================
# Figure 6: Ridge regularization path — coefficient shrinkage
# ===================================================================
def fig6_ridge_path():
    X, y = make_nonlinear_data(n=30, noise=0.25)
    deg = 10
    X_poly = np.column_stack([X**d for d in range(deg + 1)])
    # Standardize
    means = X_poly[:, 1:].mean(axis=0)
    stds = X_poly[:, 1:].std(axis=0) + 1e-10
    X_std = X_poly.copy()
    X_std[:, 1:] = (X_poly[:, 1:] - means) / stds

    lambdas = np.logspace(-6, 3, 100)
    coefs = []
    for lam in lambdas:
        I = np.eye(deg + 1)
        I[0, 0] = 0
        beta = np.linalg.inv(X_std.T @ X_std + lam * I) @ X_std.T @ y
        coefs.append(beta[1:])  # exclude intercept

    coefs = np.array(coefs)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j in range(coefs.shape[1]):
        ax.plot(lambdas, coefs[:, j], lw=1.5, label=f"β_{j+1}")
    ax.set_xscale("log")
    ax.set_xlabel("λ (regularization strength)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Figure 6: Ridge Regularization Path — Coefficients vs λ")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_ridge_path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig6_ridge_path.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating Linear Regression visual guide...\n")
    fig1_ols_fit()
    fig2_gradient_descent()
    fig3_overfitting()
    fig4_regularization()
    fig5_bias_variance()
    fig6_ridge_path()
    print(f"\nAll figures saved to {FIGDIR.resolve()}")
