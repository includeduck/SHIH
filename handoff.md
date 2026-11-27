# Handoff · Task 2 · Clustering — for Rayyan

Hi Rayyan 👋 — this is everything you need to do **Section 4 · Task 2 · Clustering (25 marks)** without having to read the rest of the repo. Your full spec is also in [PLAN.md §5](PLAN.md) and issue #1; this file adds where things stand and the exact conventions your notebook must follow so it drops straight into the final submission.

**Deadline:** the assignment is due **Sun 06 Dec 2026, 11:59 PM**. Wasay needs your PR merged by **Fri 4 Dec** at the latest to rebuild the final notebook and report — earlier is better (the plan aimed for Thu 3 Dec).

---

## 1. Where the project stands

Everything except Task 2 is done and merged into `main`:

| Part | Status | Where |
|---|---|---|
| Section 2 · Data Preparation | ✅ merged | `src/prep.py`, `notebooks/01_data_prep.ipynb` |
| Task 1 · Classification | ✅ merged | `notebooks/02_task1_classification.ipynb` |
| **Task 2 · Clustering** | ⏳ **yours** | `notebooks/03_task2_clustering.ipynb` (to create) |
| Task 3 · Neural Network + final ranking | ✅ merged | `notebooks/04_task3_neural_net.ipynb` |
| Final notebook + report | ✅ built with a Task 2 placeholder | `superhero_project.ipynb`, `report/superhero_report.pdf` |

Once your notebook is on `main`, Wasay runs one script (`tools/build_final.py`) and your cells replace the placeholder in the final notebook and the report. **You don't touch anything except your own notebook and its figures.**

Results so far, for context (not needed for Task 2): KNN is the best classifier (test Macro-F1 0.526), then Network A 0.455, Network B 0.421, Decision Tree 0.408, Baseline 0.378. The six scores turn out to be fairly weak predictors of hero vs villain — interesting to keep in mind when you look at the alignment mix of your clusters (Q8).

---

## 2. Get set up (≈15 min)

1. **Accept the collaborator invite:** https://github.com/includeduck/SHIH/invitations (it's still pending — you need it to push a branch).
2. Clone and branch:
   ```bash
   git clone https://github.com/includeduck/SHIH.git
   cd SHIH
   git switch -c rayyan/task2-clustering
   ```
3. Python **3.11 or 3.12** (TensorFlow is not needed for Task 2, so this is light):
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate      macOS/Linux/WSL:  source .venv/bin/activate
   pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
   ```
   (`pip install -r requirements.txt` also works if you want the exact full stack incl. TensorFlow.)
4. `data/train.csv` and `data/test.csv` are already in the repo — don't download or modify them.

---

## 3. Your data: Clean Set A

Already built for you by `src/prep.py` (steps 1–4 of the brief). Load it like this:

```python
from src.prep import FEATURES, SHORT_NAMES, load_raw, clean
set_a = clean(*load_raw())["set_a"]
```

| Fact | Value |
|---|---|
| Rows | **1072** (cleaned train.csv, every character) |
| Columns | `name`, the 6 `*_score` columns, `alignment` |
| `FEATURES` | `intelligence_score, strength_score, speed_score, durability_score, power_score, combat_score` |
| `SHORT_NAMES` | maps them to `INT, STR, SPD, DUR, POW, COM` (use for T7 / G10 / G11 labels) |
| Missing scores | still **NaN on purpose** — only 9 cells: strength 7, durability 1, power 1. Fill them with the median imputer in step 1. |
| `alignment` | Good 588 · Bad 328 · Neutral 93 · blank 63 (blank = `NaN`; use `.fillna("blank")` for T8) |
| Score scale | 5–100 (0s were "unknown" and are now NaN) |

**test.csv is not used anywhere in Task 2.** `alignment` and `name` are **never** inputs — alignment is only looked at in the very last step (T8 / Q8), after the clusters exist.

Shared helpers you can import:

```python
from src.style import CLUSTER_COLORS, cluster_color, save, bar_labels
# cluster_color(1) -> tab10 colour 1 (blue), cluster_color(2) -> colour 2 (orange), ...
# save(fig, "G05_elbow.png") -> writes figures/G05_elbow.png
# bar_labels(ax, bars, fmt="{:.1f}") -> writes each bar's value on top of it
```

---

## 4. What to build — step by step

Use `random_state=42` everywhere it exists. All metrics are computed on the **scaled** data.

| # | Step | Exact code / rule |
|---|---|---|
| 1 | **Fill & scale** | `SimpleImputer(strategy="median")` → `StandardScaler()`, fitted on Clean Set A (a 2-step `Pipeline` is fine). Call the result `X_scaled`. |
| 2 | **K-Means for k = 2…8** | `KMeans(n_clusters=k, n_init=10, random_state=42)`; record `inertia_` and `silhouette_score(X_scaled, labels)` for each k. |
| 3 | **Choose k** | Highest Silhouette. Tie (same to **3 decimals**) → the **smaller** k. Describe the Elbow, but it does **not** change the choice. |
| 4 | **Agglomerative** | `AgglomerativeClustering(n_clusters=k, linkage="ward")` with the same k. |
| 5 | **Score both** | `silhouette_score` ↑, `davies_bouldin_score` ↓, `calinski_harabasz_score` ↑ on `X_scaled`, plus `adjusted_rand_score(kmeans_labels, agglo_labels)`. |
| 6 | **Renumber clusters** | For each algorithm separately: mean of the 6 scores per cluster on the **original 0–100 scale** (use the median-imputed, *unscaled* values). Lowest average → **Cluster 1**, next → Cluster 2, … Cluster *i* always uses `cluster_color(i)`. |
| 7 | **PCA (plots only)** | `PCA(n_components=2, random_state=42)` fitted **once** on `X_scaled`, reused for both scatter plots. |
| 8 | **Explain** | Profiles, a few example `name`s per cluster, the PCA plots — **then** the alignment mix (T8). |

Dendrogram: `scipy.cluster.hierarchy.linkage(X_scaled, method="ward")` → `dendrogram(Z, truncate_mode="lastp", p=30)`, with a dashed horizontal line where the tree splits into k groups, e.g. `y = (Z[-k, 2] + Z[-k + 1, 2]) / 2`.

### Graphs (save each with `save(fig, ...)` using these exact filenames)

| ID | What | Filename |
|---|---|---|
| **G5** | Elbow: inertia vs k (2–8) | `G05_elbow.png` |
| **G6** | Silhouette vs k, final k marked | `G06_silhouette.png` |
| **G7** | Ward dendrogram, last 30 merges, dashed cut line at k | `G07_dendrogram.png` |
| **G8** | PCA scatter — K-Means. Axes `PC1 (xx.x %)`, `PC2 (xx.x %)` from `explained_variance_ratio_` | `G08_pca_kmeans.png` |
| **G9** | PCA scatter — Agglomerative, same projection and axes as G8 | `G09_pca_agglo.png` |
| **G10** | Cluster profiles — K-Means: x = 6 scores, one bar per cluster, y = average 0–100 | `G10_profiles_kmeans.png` |
| **G11** | Cluster profiles — Agglomerative, same format | `G11_profiles_agglo.png` |

### Tables

| ID | Content |
|---|---|
| **T6** | Algorithm · k · Silhouette ↑ · Davies–Bouldin ↓ · Calinski–Harabasz ↑ · Wins (of 3), plus an "ARI between the two" row |
| **T7** | One per algorithm: Cluster · Size · INT · STR · SPD · DUR · POW · COM · **a short name you give** (e.g. "Street-level fighters", "Cosmic powerhouses") |
| **T8** | % Good / Bad / Neutral / blank in each cluster (both algorithms) |

### Comparison rules (use these words in your answers)

- **Better separated** = wins at least 2 of the 3 metrics.
- **ARI bands:** ≥ 0.70 high · 0.40–0.69 moderate · < 0.40 low agreement. Back it up with the profiles.
- **Silhouette bands:** 0.71–1.00 strong · 0.51–0.70 reasonable · 0.26–0.50 weak · ≤ 0.25 no real structure.
- **Easier to interpret** = the algorithm whose profiles (G10/G11) differ more clearly.
- **Never** use accuracy / precision / recall for clustering.

### Questions to answer (in your notebook, final numbers)

- **Q5** Which k did you choose? What does the Elbow say?
- **Q6** Did the two algorithms find similar groups? Use ARI and profiles.
- **Q7** Which grouping is better separated, and which is easier to interpret?
- **Q8** Describe each cluster in plain words. Is any cluster mostly villains?

---

## 5. Notebook conventions — please follow exactly

The final notebook is assembled automatically, so a few details matter:

1. **File:** `notebooks/03_task2_clustering.ipynb`.
2. **First cell (markdown)** — same shape as the other sections:
   ```markdown
   # Superhero Intelligence System
   ## Section 4 · Task 2 · Clustering (25 marks)
   **Find groups of similar characters — no labels allowed.**
   ```
   (The `# Superhero…` line is dropped automatically in the final build; keep the `##` heading.)
3. **Second cell (code) — setup.** Copy these first lines *exactly*; the build script recognises and removes them:
   ```python
   import os, sys
   if os.path.basename(os.getcwd()) == "notebooks":   # run from the repo root so data/ paths work
       os.chdir("..")
   sys.path.insert(0, os.getcwd())

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import StandardScaler
   # ... your other sklearn / scipy imports ...

   from src.prep import FEATURES, SHORT_NAMES, load_raw, clean
   from src.style import CLUSTER_COLORS, cluster_color, save, bar_labels
   ```
   Every `from src.…` import must be on **its own line starting with `from src.`** (no `import src.prep`).
4. **Don't overwrite shared names.** In the final notebook your section runs *between* Task 1 and Task 3, which still need these. Never assign to: `FEATURES`, `SHORT_NAMES`, `TARGET_MAP`, `CLASS_NAMES`, `load_raw`, `clean`, `xy`, `HERO`, `VILLAIN`, `MODEL_COLORS`, `CLUSTER_COLORS`, `cluster_color`, `CM_LABELS`, `FIG_DIR`, `plot_confusion`, `save`, `bar_labels`, `RANDOM_STATE`, `np`, `pd`, `plt`. Your own variables (`X_scaled`, `km_labels`, `Z`, …) are fine.
5. **Numbering and captions.** Every graph title starts with its ID (`"G5 · Elbow curve"`), every table has a markdown heading cell right above it (`**Table T6 · Clustering metrics**`), and **directly under each G/T** put a markdown cell with 1–2 sentences in this format:
   ```markdown
   *G5:* what it shows, with the actual numbers … and what you conclude.
   ```
6. **Every graph:** title, labelled axes, legend if there's more than one colour. Clusters in `cluster_color(i)` order — Cluster 1 = first tab10 colour in *every* plot.
7. **Answers** at the end in one markdown cell, same style as Task 1:
   ```markdown
   ### Answers

   **Q5 · Which k did you choose? What does the Elbow say?**
   …
   ```
8. **Run it top to bottom from a fresh kernel** (Kernel → Restart & Run All) and **commit it with outputs**. It must not need anything except `data/`, `src/` and the imports above.

For reference, `notebooks/02_task1_classification.ipynb` is a good model of the layout and caption style.

---

## 6. Rules that cost marks if broken

- [ ] Inputs are **exactly the 6 scores** — never `overall_score`, `name`, `alignment` or any other column.
- [ ] `alignment` is only used **after** clustering (T8, Q8).
- [ ] test.csv is not used at all.
- [ ] Imputer + scaler fitted on Clean Set A; all metrics on the scaled data.
- [ ] `random_state=42` in KMeans and PCA.
- [ ] k chosen by highest Silhouette (ties → smaller k), Agglomerative uses the same k.
- [ ] Clusters renumbered by average score (lowest = Cluster 1), consistent colours everywhere.
- [ ] G5–G11, T6–T8 all present, numbered, captioned; Q5–Q8 answered with real numbers.

---

## 7. Submitting your part

```bash
git add notebooks/03_task2_clustering.ipynb figures/G05_*.png figures/G06_*.png figures/G07_*.png \
        figures/G08_*.png figures/G09_*.png figures/G10_*.png figures/G11_*.png
git commit -m "Add Task 2 clustering: K-Means vs Agglomerative (G5-G11, T6-T8, Q5-Q8)"
git push -u origin rayyan/task2-clustering
```

Then open a PR into `main` and write `Closes #1` in its description. Please **don't** edit `src/`, the other notebooks, `superhero_project.ipynb`, `report/` or `tools/` — if you need something changed there, say so in the PR or on issue #1 and Wasay will do it.

After your PR is merged, Wasay will:
1. run `tools/build_final.py` to put your section into `superhero_project.ipynb`,
2. re-run the whole notebook and re-export `report/superhero_report.pdf`,
3. zip and submit `24I-3071_24I-3000_Assignment03.zip`.

Questions → comment on **issue #1** (https://github.com/includeduck/SHIH/issues/1). Thanks!
