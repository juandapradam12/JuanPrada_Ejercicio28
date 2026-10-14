# Unsupervised Learning Lab

A reproducible lab for **unsupervised learning and data mining**: clustering, association rules, model selection, and a parallel Monte Carlo companion.

Case studies included:

1. **Dengue / climate** — structure in regional climate & land-cover features  
2. **Bakery baskets** — purchase occasions and item affinities  
3. **MPI Monte Carlo** — parallel integration with variance reduction  

Originally coursework for *Métodos Computacionales Avanzados*; reworked as a portfolio-ready analytics package.

---

## Why this project matters

Real datasets rarely arrive with clean labels. This lab asks three practical questions and answers them with structure you can see.

| Question | What we learn |
| --- | --- |
| Where is dengue pressure structurally higher? | Climate/land-cover regimes that align with historical presence — without using that label to form the clusters |
| What do customers buy together? | Purchase occasions and product affinities a bakery can act on |
| How reliable is the numerics? | Parallel Monte Carlo with an empirical $O(N^{-1/2})$ check and variance-reduction bake-off |

### 1. Climate regimes, not just a dengue label

Unsupervised clusters carve the map into climatic regimes. Side-by-side with historical prevalence, the structure is visible — and only afterward do we ask how predictive those same features are.

<p align="center">
  <img src="figures/dengue_geo_clusters.png" alt="Dengue clusters vs historical prevalence on a geographic map" width="900" />
</p>

<p align="center">
  <img src="figures/dengue_cluster_risk.png" alt="Mean dengue prevalence by unsupervised cluster" width="520" />
  &nbsp;
  <img src="figures/dengue_supervised_roc.png" alt="Hold-out ROC for supervised companion models" width="360" />
</p>

### 2. Baskets that behave like occasions

Tickets become occasions: coffee-led mornings, pastry combos, tea-and-snack patterns. FP-Growth surfaces affinities; the clock shows *when* those occasions dominate.

<p align="center">
  <img src="figures/bakery_item_cooccurrence.png" alt="Top bakery item co-occurrence heatmap" width="420" />
  &nbsp;
  <img src="figures/bakery_hourly_item_rates.png" alt="Top-item basket rates by hour of day" width="460" />
</p>

<p align="center">
  <img src="figures/bakery_temporal_traffic.png" alt="Bakery traffic by hour and weekday" width="900" />
</p>

### 3. Numbers you can defend

The MPI companion is not decoration: it shows convergence, compares variance-reduction strategies, and measures parallel speedup.

<p align="center">
  <img src="figures/monte_carlo_variance_reduction.png" alt="Monte Carlo variance reduction comparison" width="480" />
  &nbsp;
  <img src="figures/mpi_strong_scaling.png" alt="MPI strong scaling wall time and speedup" width="480" />
</p>

---

## Highlights

- Dengue clustering with K-Means++ / GMM / Agglomerative, silhouette selection, geo map, and external validation (ARI / NMI)
- Supervised companion (LR + RF) on climate → `NoYes` — labels unused during clustering
- Bakery occasions via TruncatedSVD + K-Means, **FP-Growth** rules, co-occurrence and time-of-day views
- MPI Monte Carlo for $\int_{0}^{1} e^{x}\sin(x)\,dx$ with antithetic / control / importance sampling and strong scaling
- Tests, GitHub Actions, `Dockerfile`, `environment.yml`, MIT license, vendored data under `data/`

---

## Quick start

```bash
# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate unsupervised-learning-lab

# Unsupervised + supervised pipeline (figures + CSV/JSON under outputs/)
python run_analysis.py

# Tests
pytest -q

# Docker (analytics pipeline)
docker build -t unsupervised-learning-lab .
docker run --rm unsupervised-learning-lab

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
make -C mpi vr          # variance-reduction comparison plot
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
2. Sweep $K$ for K-Means / GMM / Agglomerative  
3. Select K-Means $K^\star$ by maximum silhouette  
4. Embed with PCA; profile clusters by mean climate and mean `NoYes`  
5. Report ARI / NMI vs. historical dengue presence (**supervised labels unused in fit**)

### Bakery

1. Clean `NONE` / `Adjustment`; one-hot items; aggregate to transaction baskets  
2. Standardize; PCA for visualization  
3. Silhouette-guided K-Means for purchase occasions  
4. Mine frequent pairs/triples → rules ranked by lift  

### MPI Monte Carlo

Estimates

$$
I = \int_0^1 e^{x}\sin(x)\,dx = \frac{1}{2}\big(e(\sin 1 - \cos 1) + 1\big)
$$

with stratified work across ranks (`MPI_Reduce`). Relative error tracks the classic $N^{-1/2}$ rate.

Full write-up: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Sample results

| Signal | Typical result from `run_analysis.py` |
| --- | --- |
| Dengue structure vs label | ARI ≈ **0.42**, NMI ≈ **0.43** (clusters unused `NoYes` at fit time) |
| Supervised companion | Climate → `NoYes` ROC-AUC ≈ **0.95 (LR)** / **0.99 (RF)** (5-fold CV); top feature `h10pix` |
| Bakery occasions | Interpretable $K^{\star} \approx 6$ purchase occasions; peak hour $\approx 11$ |
| Top association rule | FP-Growth: **Toast → Coffee** (confidence ≈ 0.70, lift ≈ 1.47) |
| Monte Carlo VR | Antithetic $\approx 0.005\%$ error at $N=10^{6}$ vs crude $\approx 0.065\%$ |

After `python run_analysis.py` and `make -C mpi convergence` you should see:

- `figures/dengue_cluster_diagnostics.png` — elbow + multi-model silhouette  
- `figures/dengue_pca_clusters.png` — regions in PC space  
- `figures/dengue_cluster_risk.png` — dengue rate by cluster  
- `figures/dengue_geo_clusters.png` — lon/lat clusters vs historical prevalence  
- `figures/dengue_supervised_*.png` / `dengue_feature_importance.png` — supervised companion  
- `figures/bakery_*.png` — basket structure, top items, example occasion  
- `figures/bakery_temporal_traffic.png` / `bakery_hourly_item_rates.png` / `bakery_cluster_by_hour.png` — time-aware views  
- `figures/monte_carlo_relative_error.png` / `mpi_strong_scaling.png` — MC accuracy + scaling  
- `outputs/summary.json` — selected $K$, validation scores, supervised metrics, top rule, peak hour  

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
| MPI Monte Carlo | Missing `return` in integrand, unused MPI parallelism, wrong $N$ in Torque loop | Correct integrand, per-rank sampling, `MPI_Reduce`, convergence plot |
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
**Repository:** [`UnsupervisedLearningLab`](https://github.com/juandapradam12/UnsupervisedLearningLab)  
**License:** [MIT](LICENSE) — see also [`CITATION.cff`](CITATION.cff) for software + dengue data citation.
