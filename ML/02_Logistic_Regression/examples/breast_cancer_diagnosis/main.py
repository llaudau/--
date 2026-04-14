"""
Breast Cancer Diagnosis — Real-World Logistic Regression
=========================================================
A complete ML workflow on real medical data:
  - 569 patients, 30 features (tumor measurements)
  - Binary target: malignant (1) vs benign (0)

This example demonstrates the FULL ML pipeline:
  1. Data exploration (EDA)
  2. Feature scaling (why it matters)
  3. Train/test split
  4. Model training (from scratch + sklearn)
  5. Evaluation: confusion matrix, ROC, precision-recall
  6. Feature importance from coefficients
  7. Regularization comparison (C parameter)
  8. Cross-validation

Run:  python main.py   (use ML/venv/)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay
)
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
np.random.seed(42)


# ===================================================================
# 1. Load and explore data
# ===================================================================
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0=malignant, 1=benign

    print("=" * 60)
    print("BREAST CANCER WISCONSIN DATASET")
    print("=" * 60)
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {dict(zip(data.target_names, np.bincount(y)))}")
    print(f"  malignant (0): {(y == 0).sum()}")
    print(f"  benign    (1): {(y == 1).sum()}")
    print(f"  baseline accuracy (predict majority): {y.mean():.1%}")
    print(f"\nFeature examples: {list(X.columns[:5])} ...")
    print(f"\nFirst 3 rows:\n{X.iloc[:3, :5]}")
    return X, y, data


# ===================================================================
# Figure 1: Data Exploration — feature distributions by class
# ===================================================================
def fig1_eda(X, y, data):
    # Pick 6 most important features (by name, clinically relevant)
    features = [
        "mean radius", "mean texture", "mean perimeter",
        "mean area", "mean smoothness", "mean concavity"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.ravel(), features):
        for label, color, name in [(0, "red", "Malignant"), (1, "steelblue", "Benign")]:
            ax.hist(X.loc[y == label, feat], bins=25, alpha=0.6,
                    color=color, label=name, density=True, edgecolor="white")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Figure 1: Feature Distributions by Diagnosis\n"
                 "(Red = malignant, Blue = benign — look for separation)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_eda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig1_eda.png")


# ===================================================================
# Figure 2: Why feature scaling matters
# ===================================================================
def fig2_scaling(X_train, y_train, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, do_scale, title in [
        (axes[0], False, "Without Scaling"),
        (axes[1], True, "With StandardScaler"),
    ]:
        if do_scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)
        else:
            Xtr, Xte = X_train.values, X_test.values

        # Train with different max_iter to show convergence
        scores = []
        iters_list = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        for max_iter in iters_list:
            model = LogisticRegression(max_iter=max_iter, solver="lbfgs",
                                       C=1.0, random_state=42)
            try:
                model.fit(Xtr, y_train)
                scores.append(model.score(Xte, y_test))
            except Exception:
                scores.append(np.nan)

        ax.plot(iters_list, scores, "o-", lw=2, color="steelblue")
        ax.set_xlabel("max_iter")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_ylim(0.85, 1.0)
        ax.grid(alpha=0.3)
        ax.axhline(scores[-1], color="red", ls="--", lw=1,
                    label=f"Final: {scores[-1]:.3f}")
        ax.legend()

    fig.suptitle("Figure 2: Feature Scaling — Convergence Speed\n"
                 "(Unscaled features have very different ranges, slowing optimization)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig2_scaling.png")


# ===================================================================
# Figure 3: Confusion matrix + classification report
# ===================================================================
def fig3_confusion(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Right: what the numbers mean
    ax = axes[1]
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (
        f"True Positives  (TP) = {tp:3d}  (benign correctly identified)\n"
        f"True Negatives  (TN) = {tn:3d}  (malignant correctly identified)\n"
        f"False Positives (FP) = {fp:3d}  (malignant called benign — DANGEROUS)\n"
        f"False Negatives (FN) = {fn:3d}  (benign called malignant — unnecessary worry)\n"
        f"\n"
        f"Accuracy  = (TP+TN) / Total     = {accuracy:.3f}\n"
        f"Precision = TP / (TP+FP)         = {precision:.3f}\n"
        f"Recall    = TP / (TP+FN)         = {recall:.3f}\n"
        f"F1 Score  = 2*P*R / (P+R)        = {f1:.3f}\n"
        f"\n"
        f"In cancer diagnosis, FP (missed malignant)\n"
        f"is far worse than FN. We want HIGH RECALL\n"
        f"for the malignant class."
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.axis("off")
    ax.set_title("Metrics Explained")

    fig.suptitle("Figure 3: Confusion Matrix and Classification Metrics",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_confusion.png")


# ===================================================================
# Figure 4: ROC curve + Precision-Recall curve
# ===================================================================
def fig4_roc_pr(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]  # P(benign)

    # ROC
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: ROC
    ax = axes[0]
    ax.plot(fpr, tpr, "b-", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, 0, tpr, alpha=0.1, color="blue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Right: Precision-Recall
    ax = axes[1]
    ax.plot(recall, precision, "r-", lw=2.5, label=f"LR (AUC = {pr_auc:.3f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", ls="--", lw=1,
               label=f"Random baseline ({baseline:.2f})")
    ax.fill_between(recall, 0, precision, alpha=0.1, color="red")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 4: ROC and Precision-Recall Curves\n"
                 "(Both near-perfect — logistic regression works well on this data)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_roc_pr.png")


# ===================================================================
# Figure 5: Feature importance — which tumor measurements matter?
# ===================================================================
def fig5_feature_importance(model, feature_names):
    coefs = model.coef_[0]
    # Sort by absolute value
    idx = np.argsort(np.abs(coefs))[::-1]
    top_n = 15  # show top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["steelblue" if c > 0 else "tomato" for c in coefs[idx[:top_n]]]
    bars = ax.barh(range(top_n), coefs[idx[:top_n]], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (log-odds per unit increase)")
    ax.set_title("Figure 5: Top 15 Features by Coefficient Magnitude\n"
                 "(Blue = higher value → more likely benign,  "
                 "Red = higher value → more likely malignant)")
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig5_feature_importance.png")


# ===================================================================
# Figure 6: Regularization strength comparison
# ===================================================================
def fig6_regularization(X_train_scaled, y_train, X_test_scaled, y_test, feature_names):
    C_values = np.logspace(-4, 4, 30)
    train_accs, test_accs = [], []
    n_nonzero = []

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=42)
        model.fit(X_train_scaled, y_train)
        train_accs.append(model.score(X_train_scaled, y_train))
        test_accs.append(model.score(X_test_scaled, y_test))
        n_nonzero.append(np.sum(np.abs(model.coef_[0]) > 0.01))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy vs C
    ax = axes[0]
    ax.plot(C_values, train_accs, "o-", ms=3, lw=1.5, label="Train accuracy",
            color="steelblue")
    ax.plot(C_values, test_accs, "s-", ms=3, lw=1.5, label="Test accuracy",
            color="tomato")
    ax.set_xscale("log")
    ax.set_xlabel("C = 1/λ  (larger C = less regularization)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Regularization")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.9, 1.005)
    ax.annotate("Strong\nregularization", xy=(1e-4, 0.92), fontsize=9, color="gray")
    ax.annotate("Weak\nregularization", xy=(1e3, 0.92), fontsize=9, color="gray")

    # Right: coefficient paths (L1 for sparsity)
    ax = axes[1]
    coef_paths = []
    for C in C_values:
        model = LogisticRegression(C=C, penalty="l1", solver="saga",
                                   max_iter=10000, random_state=42)
        model.fit(X_train_scaled, y_train)
        coef_paths.append(model.coef_[0].copy())
    coef_paths = np.array(coef_paths)

    # Plot top 8 features by final magnitude
    final_importance = np.abs(coef_paths[-1])
    top_idx = np.argsort(final_importance)[-8:]
    for i in top_idx:
        ax.plot(C_values, coef_paths[:, i], lw=1.5, label=feature_names[i])
    ax.set_xscale("log")
    ax.set_xlabel("C = 1/λ")
    ax.set_ylabel("Coefficient")
    ax.set_title("Lasso (L1) Coefficient Path")
    ax.legend(fontsize=7, loc="upper left")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 6: Regularization — C controls model complexity\n"
                 "(L1/Lasso drives unimportant features to exactly zero)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_regularization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig6_regularization.png")


# ===================================================================
# Cross-validation
# ===================================================================
def cross_validation_report(X_scaled, y):
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold)")
    print("=" * 60)
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"Fold accuracies: {scores}")
    print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

    auc_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
    print(f"AUC scores:      {auc_scores}")
    print(f"Mean AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    return scores, auc_scores


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Breast Cancer Diagnosis with Logistic Regression\n")

    # 1. Load data
    X, y, data = load_data()

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # 3. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train model
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # 5. Generate figures
    print("\nGenerating figures...")
    fig1_eda(X, y, data)
    fig2_scaling(X_train, y_train, X_test, y_test)
    fig3_confusion(model, X_test_scaled, y_test, data.target_names)
    fig4_roc_pr(model, X_test_scaled, y_test)
    fig5_feature_importance(model, data.feature_names)
    fig6_regularization(X_train_scaled, y_train, X_test_scaled, y_test,
                        data.feature_names)

    # 6. Cross-validation
    X_all_scaled = scaler.fit_transform(X)
    cross_validation_report(X_all_scaled, y)

    print(f"\nAll figures saved to {FIGDIR.resolve()}")
