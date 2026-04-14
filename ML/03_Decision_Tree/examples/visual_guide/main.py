"""
Decision Tree — Visual Guide
==============================
Figures:
  1. How a tree partitions 2D space (depth 1 vs 3 vs unlimited)
  2. Gini vs Entropy splitting criterion comparison
  3. Overfitting: train vs test accuracy as depth increases
  4. Tree visualization on real data (breast cancer)
  5. Feature importance
  6. Decision tree vs Logistic regression on non-linear data

Run:  python main.py  (use ML/venv/)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
np.random.seed(42)


def _plot_boundary(ax, clf, X, y, title="", h=0.02):
    """Plot decision boundary for 2D data."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(["#e74c3c", "#3498db"]))
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", s=15, alpha=0.6,
               edgecolors="k", lw=0.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", s=15, alpha=0.6,
               edgecolors="k", lw=0.3)
    ax.set_title(title)


# ===================================================================
# Figure 1: How depth controls partitioning
# ===================================================================
def fig1_depth_comparison():
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

    depths = [1, 3, None]
    labels = ["Depth = 1 (underfit)", "Depth = 3 (good)", "Depth = unlimited (overfit)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, depth, label in zip(axes, depths, labels):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        n_leaves = clf.get_n_leaves()
        _plot_boundary(ax, clf, X, y, f"{label}\nacc={acc:.1%}, {n_leaves} leaves")
        ax.set_xlabel("x₁")

    axes[0].set_ylabel("x₂")
    fig.suptitle("Figure 1: Decision Tree Depth — From Underfit to Overfit\n"
                 "(Notice: all boundaries are axis-aligned rectangles)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig1_depth.png")


# ===================================================================
# Figure 2: Gini vs Entropy
# ===================================================================
def fig2_gini_vs_entropy():
    p = np.linspace(0.001, 0.999, 200)

    gini = 2 * p * (1 - p)  # for binary: 1 - p^2 - (1-p)^2 = 2p(1-p)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    misclass = 1 - np.maximum(p, 1 - p)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p, gini, "b-", lw=2.5, label="Gini: 2p(1-p)")
    ax.plot(p, entropy, "r-", lw=2.5, label="Entropy: -p log p - (1-p) log(1-p)")
    ax.plot(p, misclass, "g--", lw=2, label="Misclassification: 1 - max(p, 1-p)")
    ax.set_xlabel("Fraction of class 1 (p)")
    ax.set_ylabel("Impurity")
    ax.set_title("Figure 2: Impurity Measures for Binary Classification\n"
                 "(All maximized at p=0.5, all zero at p=0 or p=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.annotate("Maximum\nimpurity\n(50/50 mix)", xy=(0.5, 0.52), fontsize=9,
                ha="center", color="gray")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_impurity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig2_impurity.png")


# ===================================================================
# Figure 3: Overfitting — train vs test accuracy by depth
# ===================================================================
def fig3_overfitting():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    depths = range(1, 21)
    train_acc, test_acc, n_leaves_list = [], [], []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_acc.append(clf.score(X_train, y_train))
        test_acc.append(clf.score(X_test, y_test))
        n_leaves_list.append(clf.get_n_leaves())

    best_depth = depths[np.argmax(test_acc)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(depths, train_acc, "o-", lw=2, ms=4, label="Train", color="steelblue")
    ax.plot(depths, test_acc, "s-", lw=2, ms=4, label="Test", color="tomato")
    ax.axvline(best_depth, color="gray", ls=":", lw=1.5,
               label=f"Best depth = {best_depth}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Test Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.88, 1.01)

    ax = axes[1]
    ax.plot(depths, n_leaves_list, "o-", lw=2, ms=4, color="steelblue")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Number of leaves")
    ax.set_title("Tree Complexity")
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 3: Overfitting in Decision Trees (Breast Cancer data)\n"
                 "(Train accuracy hits 100% quickly, but test accuracy peaks then drops)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_overfitting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_overfitting.png")


# ===================================================================
# Figure 4: Tree visualization (small tree on breast cancer)
# ===================================================================
def fig4_tree_visualization():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(clf, feature_names=data.feature_names,
              class_names=data.target_names, filled=True,
              rounded=True, fontsize=10, ax=ax,
              impurity=True, proportion=True)
    ax.set_title(f"Figure 4: Decision Tree (depth=3) on Breast Cancer Data\n"
                 f"Test accuracy = {acc:.1%} — readable by humans!",
                 fontsize=14)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_tree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_tree.png")

    # Also print text rules
    print("\n    Tree rules (text):")
    rules = export_text(clf, feature_names=list(data.feature_names), max_depth=3)
    for line in rules.split("\n")[:15]:
        print(f"    {line}")
    print("    ...")


# ===================================================================
# Figure 5: Feature importance comparison
# ===================================================================
def fig5_feature_importance():
    data = load_breast_cancer()
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(data.data, data.target)

    importances = clf.feature_importances_
    idx = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(idx)), importances[idx], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([data.feature_names[i] for i in idx])
    ax.set_xlabel("Feature Importance (total impurity reduction)")
    ax.set_title("Figure 5: Decision Tree Feature Importance\n"
                 "(Which features does the tree use to split?)")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig5_importance.png")


# ===================================================================
# Figure 6: Decision tree vs Logistic regression on non-linear data
# ===================================================================
def fig6_tree_vs_logistic():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Logistic regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_test = lr.score(X_test_s, y_test)

    # Decision tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_test = dt.score(X_test, y_test)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # LR needs scaled data for boundary plot
    lr_for_plot = LogisticRegression(max_iter=1000)
    lr_for_plot.fit(X_train, y_train)
    _plot_boundary(axes[0], lr_for_plot, X_test, y_test,
                   f"Logistic Regression (test acc={lr_test:.1%})\nCan only draw a straight line")
    _plot_boundary(axes[1], dt, X_test, y_test,
                   f"Decision Tree depth=5 (test acc={dt_test:.1%})\nCaptures the non-linear boundary")

    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[1].set_xlabel("x₁")

    fig.suptitle("Figure 6: Why Decision Trees Exist — Non-linear Boundaries\n"
                 "(Logistic regression fails on moons; decision tree succeeds)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_tree_vs_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig6_tree_vs_lr.png")


# ===================================================================
if __name__ == "__main__":
    print("Generating Decision Tree visual guide...\n")
    fig1_depth_comparison()
    fig2_gini_vs_entropy()
    fig3_overfitting()
    fig4_tree_visualization()
    fig5_feature_importance()
    fig6_tree_vs_logistic()
    print(f"\nAll figures saved to {FIGDIR.resolve()}")
