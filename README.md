# Dengue Risk Clustering & Bakery Basket Intelligence

**Unsupervised learning + parallel numerical computing** on two real datasets: climatic drivers of dengue prevalence, and retail purchase occasions in a bakery.

This repository turns exploratory coursework into a **reproducible analytics package**: clean data pipelines, stronger model selection, association rules, MPI Monte Carlo integration, and documentation you can hand to a collaborator or recruiter.

---

## Why this project matters

| Problem | What the models deliver |
| --- | --- |
| Where is dengue pressure structurally higher? | Climate/land-cover **clusters** of administrative regions, validated against historical presence without leaking the label into training |
| What do customers buy together? | **Purchase-occasion clusters**, PCA views of basket structure, and **association rules** (support / confidence / lift) for merchandising |
| How reliable is parallel Monte Carlo? | MPI estimators for a known integral, with an empirical **\(O(N^{-1/2})\)** convergence check |

---

## Highlights

- **Dengue clustering** with K-Means++, Gaussian Mixtures, and Agglomerative clustering; K chosen by silhouette (elbow + Calinski–Harabasz reported)
- **Geospatial dengue map** of clusters vs historical prevalence in lon/lat space
- **External validation** (Adjusted Rand Index / NMI) against `NoYes` dengue prevalence
- **Supervised companion** — logistic regression + random forest on climate features → `NoYes` (labels never used in clustering)
- **Bakery basket mining** with SVD occasions, association rules, and **time-of-day / weekday** traffic + item seasonality
- **MPI programs**: rectangle-rule \(\pi\), Monte Carlo for \(\int_0^1 e^{x}\sin(x)\,dx\), plus **strong-scaling** study
- **Tests + CI**: `pytest` and GitHub Actions for the Python pipeline and MPI smoke/scaling builds
- **Packaging**: `Dockerfile`, `environment.yml`, MIT `LICENSE`, `CITATION.cff`
- **One-command reproduction**: `python run_analysis.py` and `make -C mpi convergence`
- Datasets vendored under `data/` — **no Google Drive / Colab mount required**

---

## Quick start

```bash
# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate dengue-bakery

# Unsupervised + supervised pipeline (figures + CSV/JSON under outputs/)
python run_analysis.py

# Tests
pytest -q

# Docker (analytics pipeline)
docker build -t dengue-bakery .
docker run --rm dengue-bakery

# Interactive walkthrough
jupyter notebook Dengue_and_Bakery_Analysis.ipynb
```

### MPI numerical methods

Requires an MPI toolchain (`mpicc`, `mpirun`). **Open MPI** is recommended locally; MPICH also works on many HPC clusters.

```bash
make -C mpi all
make -C mpi run-pi      # parallel π
make -C mpi run-mc      # Monte Carlo estimate
make -C mpi convergence # sweep N + error plot → figures/monte_carlo_relative_error.png
make -C mpi scaling     # strong scaling → figures/mpi_strong_scaling.png
```

HPC / Torque helpers: `torque.sh` (π), `torque2.sh` (Monte Carlo study).

---

## Repository layout

```text
├── Dengue_and_Bakery_Analysis.ipynb   # Main analysis notebook
├── run_analysis.py                    # Headless end-to-end runner
├── analysis/                          # Reusable dengue + bakery modules
├── data/
│   ├── dengue.csv                     # DAAG dengue prevalence regions
│   └── BreadBasket_DMS.csv            # Bakery transactions
├── mpi/
│   ├── integra.c                      # Parallel Monte Carlo integration
│   ├── Listing12.7.c                  # Parallel π (rectangle rule)
│   ├── Makefile
│   └── plots.py
├── docs/                              # Design notes & methodology
├── figures/                           # Generated plots
├── outputs/                           # Metrics, rules, clustered tables
└── notebooks/archive/                 # Original Colab notebook
```

---

## Methods (short version)

### Dengue

1. Drop incomplete rows; standardize six climate/geography features  
2. Sweep \(K\) for K-Means / GMM / Agglomerative  
3. Select K-Means \(K^\star\) by maximum silhouette  
4. Embed with PCA; profile clusters by mean climate and mean `NoYes`  
5. Report ARI / NMI vs. historical dengue presence (**supervised labels unused in fit**)

### Bakery

1. Clean `NONE` / `Adjustment`; one-hot items; aggregate to transaction baskets  
2. Standardize; PCA for visualization  
3. Silhouette-guided K-Means for purchase occasions  
4. Mine frequent pairs/triples → rules ranked by lift  

### MPI Monte Carlo

Estimates

\[
I = \int_0^1 e^{x}\sin(x)\,dx
= \tfrac12\bigl(e(\sin 1 - \cos 1) + 1\bigr)
\]

with stratified work across ranks (`MPI_Reduce`). Relative error tracks the classic \(N^{-1/2}\) rate.

Full write-up: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Sample results

| Signal | Typical result from `run_analysis.py` |
| --- | --- |
| Dengue structure vs label | ARI ≈ **0.42**, NMI ≈ **0.43** (clusters unused `NoYes` at fit time) |
| Supervised companion | Climate → `NoYes` ROC-AUC ≈ **0.95 (LR)** / **0.99 (RF)** (5-fold CV); top feature `h10pix` |
| Bakery occasions | Interpretable \(K^\star\) ≈ **6** purchase occasions; peak hour ≈ **11** |
| Top association rule | **Toast → Coffee** (confidence ≈ 0.70, lift ≈ 1.47) |
| Monte Carlo | Relative error ≈ **0.03%** at \(N=10^7\) (4 ranks) |

After `python run_analysis.py` and `make -C mpi convergence` you should see:

- `figures/dengue_cluster_diagnostics.png` — elbow + multi-model silhouette  
- `figures/dengue_pca_clusters.png` — regions in PC space  
- `figures/dengue_cluster_risk.png` — dengue rate by cluster  
- `figures/dengue_geo_clusters.png` — lon/lat clusters vs historical prevalence  
- `figures/dengue_supervised_*.png` / `dengue_feature_importance.png` — supervised companion  
- `figures/bakery_*.png` — basket structure, top items, example occasion  
- `figures/bakery_temporal_traffic.png` / `bakery_hourly_item_rates.png` / `bakery_cluster_by_hour.png` — time-aware views  
- `figures/monte_carlo_relative_error.png` / `mpi_strong_scaling.png` — MC accuracy + scaling  
- `outputs/summary.json` — selected \(K\), validation scores, supervised metrics, top rule, peak hour  

---

## Data sources

- **Dengue**: Hales et al. / DAAG `dengue` (administrative-region climate & prevalence, 1961–1990)  
- **Bakery**: *The Bread Basket* (Edinburgh) transaction log — `BreadBasket_DMS.csv`

Citations and field dictionaries: [`docs/DATA.md`](docs/DATA.md).

---

## What improved over the original draft

| Area | Before | Now |
| --- | --- | --- |
| Data access | Hard-coded Google Drive paths | Local `data/` + documented sources |
| Bakery features | Reused dengue column names (broken PCA/K-Means) | Correct `Item_*` basket matrix |
| Model selection | Inconsistent K (e.g. text said 14, code used 19; bakery K≈97) | Silhouette-selected K in a stable range |
| Algorithms | K-Means only | K-Means++ / GMM / Agglomerative + association rules |
| MPI Monte Carlo | Missing `return` in integrand, unused MPI parallelism, wrong \(N\) in Torque loop | Correct integrand, per-rank sampling, `MPI_Reduce`, convergence plot |
| Docs | None | README + methodology + data notes |

---

## Requirements

- Python 3.10+  
- Packages in `requirements.txt`  
- Optional: MPICH/Open MPI for `mpi/` targets  

---

## License & authorship

Coursework originally developed for *Métodos Computacionales Avanzados* (Universidad de los Andes). Enhanced as a reproducible portfolio project.

**Author:** Juan David Prada Malagón  
**License:** [MIT](LICENSE) — see also [`CITATION.cff`](CITATION.cff) for software + dengue data citation.
