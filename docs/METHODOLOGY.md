# Methodology

This document describes the modelling choices, fixes, and evaluation protocol used in the enhanced repository.

## 1. Dengue prevalence clustering

### Objective

Discover climatic / geographic regimes among administrative regions and check whether those regimes align with historical dengue presence (`NoYes`).

### Features

Default unsupervised feature set (same intent as the original exercise):

- `humid`, `temp`, `h10pix`, `trees`, `Xmin`, `Ymin`

Rows with missing values in these columns or in `NoYes` are dropped. Features are standardized with `StandardScaler` so Euclidean distances are comparable.

### Models compared

For each \(K \in [2, 12]\):

| Model | Role |
| --- | --- |
| **K-Means++** (`n_init=20` during search, `50` for final fit) | Spherical clusters; primary selector |
| **Gaussian Mixture** (full covariance) | Soft, elliptical clusters |
| **Agglomerative** (Ward linkage via sklearn default) | Hierarchical baseline |

### Model selection

- **Primary:** maximize silhouette score for K-Means  
- **Reported:** inertia elbow curve; Calinski–Harabasz index  
- Final labels come from K-Means at \(K^\star\)

Using silhouette avoids the original notebook’s inconsistency (narrative \(K=14\), fitted \(K=19\)).

### Visualization & profiling

- 2D **PCA** embedding of the scaled matrix  
- Per-cluster means of climate features and of `NoYes` (risk profile)  
- **Geographic map** of region midpoints \(((X_{\min}+X_{\max})/2,\ (Y_{\min}+Y_{\max})/2)\) colored by cluster and by `NoYes` side-by-side

### External validation (not used for fitting)

| Metric | Meaning |
| --- | --- |
| Adjusted Rand Index | Agreement with `NoYes` beyond chance |
| Normalized Mutual Information | Shared information with `NoYes` |
| Std. of cluster-wise dengue rates | Separation of risk levels across clusters |

This keeps the workflow honest: clustering remains unsupervised, while we still quantify epidemiological relevance.

### Supervised companion (labels used only here)

After clustering is finished, we fit **logistic regression** and **random forest** on the same climate features to predict `NoYes`:

- Stratified 5-fold CV for ROC-AUC / accuracy / F1  
- Hold-out ROC + confusion matrix  
- Feature importance (`|coef|` and RF impurity)  

Cluster labels are **not** used as predictors. The companion answers: “how predictive is the climate structure the clusters are built from?”

---

## 2. Bakery basket intelligence

### Objective

Represent transactions as item baskets, find purchase occasions, and surface product affinities for merchandising.

### Preprocessing

1. Strip item names; drop `NONE` and `Adjustment`  
2. Build a binary transaction × item matrix (`Item_*` columns)  
3. Standardize columns before PCA / K-Means  

**Critical fix:** the original Colab notebook reused dengue’s `x_features` for bakery PCA and clustering, which made those steps invalid. Features are now the item indicators.

### Dimensionality reduction

`PCA(n_components=2)` on the scaled basket matrix for visualization and variance diagnostics (via `explained_variance_ratio_`).

### Clustering

- L2-normalize baskets, reduce with **TruncatedSVD** (~25 components), then K-Means++  
- Sweep \(K \in [2, 15]\); report silhouette, inertia, and Calinski–Harabasz  
- Select an interpretable \(K^\star\) with an **inertia knee** (windowed Calinski–Harabasz tie-break), not a blindly maximizing silhouette (which can drift toward the largest \(K\))  
- Interpret each cluster as a **purchase occasion** via top item counts  

The original draft explored \(K \in [80, 100]\) and reported \(K=97\), which over-fragments ~9k transactions and yields unstable “occasions”.

### Association rules (FP-Growth)

Primary engine: **FP-Growth** via `mlxtend` (Apriori-style fallback if unavailable):

- `min_support` (default 0.02)  
- `min_confidence` (default 0.35)  
- Rank by **lift**, then confidence, then support  
- Export frequent itemsets + a top-item **co-occurrence heatmap**

Rules answer questions such as “given toast, how much more likely is coffee?”

### Time-aware bakery views

Each basket keeps its first observed timestamp. From that we report:

- Transaction volume by **hour** and **weekday**  
- Top-item attachment rates by hour (seasonality heatmap)  
- Cluster / occasion **mix by hour** (when each purchase occasion dominates)

---

## 3. Parallel numerical integration (MPI)

### π via rectangle rule (`mpi/Listing12.7.c`)

Estimates

\[
\pi \approx \int_0^1 \frac{4}{1+x^2}\,dx
\]

Intervals are striped across ranks; partial sums reduce with `MPI_SUM`. Non-interactive CLI mode: `./Listing12.7 <n_intervals>`.

### Monte Carlo (`mpi/integra.c`)

Estimates

\[
I = \int_0^1 e^{x}\sin(x)\,dx
= \frac12\bigl(e(\sin 1 - \cos 1) + 1\bigr)
\]

with uniform samples on \([0,1]\). Each rank draws a share of \(N\) samples (`rand_r` with distinct seeds); sums reduce to rank 0.

Optional variance-reduction methods (CLI arg):

| Method | Estimator idea |
| --- | --- |
| `crude` | \(f(U),\ U\sim U(0,1)\) |
| `antithetic` | \(\tfrac12\bigl(f(U)+f(1-U)\bigr)\) |
| `control` | \(f(U)-c(e^{U}-(e-1))\) |
| `importance` | Sample \(Y\sim g(y)=2y\), weight \(f(Y)/g(Y)\) |

`make -C mpi vr` compares relative error vs \(N\) across methods.

### Bugs fixed in the original `integra.c`

| Issue | Impact | Fix |
| --- | --- | --- |
| `f()` computed `exp(x)*sin(x)` without `return` | Undefined estimates | Return the integrand |
| `malloc(sizeof(double *)*N)` | Wrong allocation size | Stream samples without storing full arrays |
| MPI init without work split / reduce | No real parallelism | Partition \(N\), `MPI_Reduce` |
| Torque loop wrote `1000` for every \(N\) | Fake “convergence” study | Pass the loop variable as \(N\) |

Convergence plotting (`mpi/plots.py`) overlays empirical relative error against an \(O(N^{-1/2})\) reference.

---

## 4. Reproducibility checklist

```bash
pip install -r requirements.txt
python run_analysis.py
make -C mpi convergence
```

Artifacts:

- `figures/*.png`  
- `outputs/summary.json`  
- `outputs/*_metrics.csv`, association rules, clustered tables  

Random seeds are fixed (`random_state=42`) for sklearn estimators. Monte Carlo remains stochastic by design; increasing \(N\) tightens the error.
