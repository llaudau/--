"""
Random Forest — Visual Guide
============================
Figures:
  1. Single tree vs Random Forest decision boundary
  2. Variance reduction as number of trees increases
  3. Out-of-bag score convergence
  4. Feature importance: MDI vs permutation
  5. Tree vs bagging vs random forest comparison

Run: python main.py  (use ML/venv/)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


def _plot_boundary(ax, clf, X, y, title, h=0.03):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.3, cmap=ListedColormap(["#e67e22", "#2980b9"]))
    ax.contour(xx, yy, zz, colors="k", linewidths=0.5)
    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="#d35400",
        s=18,
        alpha=0.65,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="#2471a3",
        s=18,
        alpha=0.65,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_title(title)
    ax.set_xlabel("x1")


def fig1_tree_vs_forest():
    X, y = make_moons(n_samples=350, noise=0.28, random_state=42)

    tree = DecisionTreeClassifier(max_depth=None, random_state=42)
    forest = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=42,
    )

    tree.fit(X, y)
    forest.fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_boundary(
        axes[0],
        tree,
        X,
        y,
        f"Single Tree\ntrain acc = {tree.score(X, y):.1%}",
    )
    _plot_boundary(
        axes[1],
        forest,
        X,
        y,
        f"Random Forest\ntrain acc = {forest.score(X, y):.1%}",
    )
    axes[0].set_ylabel("x2")
    fig.suptitle(
        "Figure 1: Single Tree vs Random Forest on Noisy Moons\n"
        "(The forest keeps flexibility but smooths unstable boundaries)",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_tree_vs_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig1_tree_vs_forest.png")


def fig2_variance_reduction():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    tree_counts = [1, 2, 5, 10, 20, 50, 100, 200]
    test_scores = []

    for n_trees in tree_counts:
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(tree_counts, test_scores, "o-", color="steelblue", lw=2, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of trees (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Figure 2: More Trees Usually Stabilize Performance\n"
        "(Random Forest rarely overfits by just adding trees)"
    )
    ax.grid(alpha=0.3)
    ax.set_ylim(min(test_scores) - 0.01, 1.0)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_variance_reduction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig2_variance_reduction.png")


def fig3_oob_convergence():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    # Very small forests can leave some samples without OOB predictions,
    # which is more distracting than useful in a teaching example.
    tree_counts = [20, 40, 80, 120, 200, 300]
    oob_scores = []
    test_scores = []

    for n_trees in tree_counts:
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_features="sqrt",
            oob_score=True,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        oob_scores.append(clf.oob_score_)
        test_scores.append(clf.score(x_test, y_test))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(tree_counts, oob_scores, "o-", lw=2, ms=5, label="OOB score", color="tomato")
    ax.plot(tree_counts, test_scores, "s--", lw=2, ms=5, label="Test score", color="steelblue")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Figure 3: OOB Score as Free Validation\n"
        "(OOB tracks held-out performance reasonably well)"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_oob_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_oob_convergence.png")


def fig4_importance_comparison():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    mdi = clf.feature_importances_
    perm = permutation_importance(
        clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    ).importances_mean
    idx = np.argsort(perm)[-12:]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    axes[0].barh(range(len(idx)), mdi[idx], color="#5dade2", edgecolor="white")
    axes[0].set_yticks(range(len(idx)))
    axes[0].set_yticklabels([data.feature_names[i] for i in idx])
    axes[0].set_title("MDI Importance")
    axes[0].set_xlabel("Importance")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(range(len(idx)), perm[idx], color="#ec7063", edgecolor="white")
    axes[1].set_title("Permutation Importance")
    axes[1].set_xlabel("Importance drop")
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Figure 4: Feature Importance in Random Forest\n"
        "(Permutation importance is usually the safer interpretation)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_importance_comparison.png")


def fig5_tree_bagging_forest():
    X, y = make_moons(n_samples=450, noise=0.3, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Single Tree": DecisionTreeClassifier(random_state=42),
        "Bagging Trees": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }

    scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores[name] = accuracy_score(y_test, model.predict(x_test))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        list(scores.keys()),
        list(scores.values()),
        color=["#f5b041", "#7fb3d5", "#58d68d"],
        edgecolor="white",
    )
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Figure 5: Single Tree vs Bagging vs Random Forest\n"
        "(Bagging reduces variance; random features reduce it further)"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_tree_bagging_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig5_tree_bagging_forest.png")


def main():
    print("Generating Random Forest visual guide figures...")
    fig1_tree_vs_forest()
    fig2_variance_reduction()
    fig3_oob_convergence()
    fig4_importance_comparison()
    fig5_tree_bagging_forest()
    print(f"Done. Figures saved in: {FIGDIR}")


if __name__ == "__main__":
    main()
