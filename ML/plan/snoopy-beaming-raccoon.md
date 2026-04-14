# Plan: 04_Random_Forest_Ensemble

## Context

Continuing the ML learning roadmap. Modules 01-03 (Linear Regression, Logistic Regression, Decision Tree) are complete. Next is 04_Random_Forest_Ensemble. The `plan.md` already exists with detailed theory outline. Need to create the README and example code.

## Deliverables

### 1. `04_Random_Forest_Ensemble/README.md`

Following the established 7-section structure (matching 03_Decision_Tree/README.md style):

1. **Algorithm Identity Card** — table from plan.md
2. **What Problem Does This Solve?** — Single trees are unstable; averaging many de-correlated trees fixes this
3. **Mathematical Foundation**
   - 2.1 Ensemble variance formula: $\text{Var}_{ens} = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$
   - 2.2 Bootstrap sampling and the 63.2% rule
   - 2.3 Bagging algorithm (bootstrap + aggregate)
   - 2.4 Random Forest = Bagging + random feature subsets
   - 2.5 OOB error estimation
   - 2.6 Feature importance (MDI vs Permutation)
4. **Pros and Cons** — table
5. **Connections** — from Decision Trees (03), to Gradient Boosting (09), quant applications
6. **Key Equations Summary**

### 2. `04_Random_Forest_Ensemble/examples/visual_guide/main.py`

Six figures following the exact code style from 03 (FIGDIR, function-per-figure, `if __name__` runner):

| Figure | What it shows |
|--------|--------------|
| fig1_bootstrap | Bootstrap sampling visualization — which samples are in-bag vs OOB |
| fig2_single_vs_bagging_vs_rf | Decision boundaries on moons: single tree vs bagging vs RF |
| fig3_oob_convergence | OOB error vs number of trees — shows convergence |
| fig4_variance_reduction | Train many single trees vs one RF — show instability vs stability |
| fig5_feature_importance | MDI vs Permutation importance on breast cancer data |
| fig6_ensemble_theory | Theoretical plot: ensemble variance as function of correlation $\rho$ and number of trees $M$ |

### 3. Update `04_Random_Forest_Ensemble/plan.md`

Mark status as complete.

## Files to create/modify

- **Create**: `04_Random_Forest_Ensemble/README.md`
- **Create**: `04_Random_Forest_Ensemble/examples/visual_guide/main.py`
- **Modify**: `04_Random_Forest_Ensemble/plan.md` (status update)

## Verification

```bash
cd /home/khw/Documents/Git_repository/ML
source venv/bin/activate
python 04_Random_Forest_Ensemble/examples/visual_guide/main.py
```

Should produce 6 PNG figures in `examples/visual_guide/figures/`.
