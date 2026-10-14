# Data dictionary

## `data/dengue.csv`

Derived from the **DAAG** `dengue` dataset (Hales et al., *The Lancet*, 2002): for ~2000 administrative regions, climate / tree-cover summaries and whether dengue was recorded between 1961 and 1990.

| Column | Description |
| --- | --- |
| `humid` | Average vapour density, 1961–1990 |
| `humid90` | 90th percentile of `humid` |
| `temp` | Average temperature, 1961–1990 |
| `temp90` | 90th percentile of `temp` |
| `h10pix` | Max `humid` within a 10-pixel radius |
| `h10pix90` | Max `humid90` within a 10-pixel radius |
| `trees` | Percent tree cover |
| `trees90` | 90th percentile of `trees` |
| `NoYes` | Dengue observed? (`1` = yes) — **validation only** |
| `Xmin`, `Xmax` | Longitude bounds |
| `Ymin`, `Ymax` | Latitude bounds |

Source CSV: [Rdatasets DAAG/dengue](https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/dengue.csv).

## `data/BreadBasket_DMS.csv`

Transaction log from **The Bread Basket** bakery (Edinburgh).

| Column | Description |
| --- | --- |
| `Date` | Transaction date (`YYYY-MM-DD`) |
| `Time` | Transaction time (`HH:MM:SS`) |
| `Transaction` | Basket / ticket id |
| `Item` | Product name (multiple rows per transaction) |

Rows labeled `NONE` or `Adjustment` are removed before basket construction.

Public copies are widely mirrored (e.g. Kaggle “Transactions from a bakery”). This repo vendors a copy under `data/` for offline reproducibility.
