# Superhero Intelligence System — Implementation Plan

**Assignment 03 · Machine Learning Project** — "Can six ability scores tell a hero from a villain?"
**Due:** Sunday **06 Dec 2026, 11:59 PM** (5 min late = −10%, ≤1 h = −20%, ≤12 h = −50%, >12 h = 0).
**Team:** Muhammad Wasay Tariq (24I-3071, `includeduck`) · Syed Muhammad Rayyan Hassan (`rayyanhasan899`)
**Repo:** https://github.com/includeduck/SHIH

| Section | Marks | Owner |
|---|---|---|
| 2 · Data Preparation | 10 | Wasay |
| 3 · Task 1 · Classification (Baseline, Decision Tree, KNN) | 25 | Wasay |
| 4 · Task 2 · Clustering (K-Means, Agglomerative) | 25 | **Rayyan** |
| 5 · Task 3 · Neural Network (Network A, B) + final ranking | 25 | Wasay |
| Integration, final notebook, report assembly | — | Wasay |

**Split by marks:** Wasay 60/85 = **70.6 %**, Rayyan 25/85 = **29.4 %**.
Task 2 is the cleanest cut: it only needs Clean Set A, never touches `test.csv`, and none of the other tasks depend on its output.

---

## 0. Ground rules that apply to everyone (lose marks if broken)

- **Exactly 6 inputs:** `intelligence_score, strength_score, speed_score, durability_score, power_score, combat_score`. Never `overall_score` (leaks the answer), text columns, `has_*` flags, or anything else. `name` is a label only.
- **Do not merge** `train.csv` and `test.csv`. **Do not** make your own train/test split (the only split allowed is the 80/20 validation split inside train.csv for Task 3).
- `test.csv` is opened for scoring **once**, at the end of Task 1 and Task 3. Never `.fit()` / `.fit_transform()` on test data. Never choose settings from test scores.
- **Clustering never sees `alignment`** until after the clusters exist (step 8 of Task 2).
- `random_state=42` everywhere it exists; networks use `tf.keras.utils.set_random_seed(42)`.
- Tools allowed: pandas, scikit-learn, SciPy, matplotlib (seaborn ok), TensorFlow/Keras.
- Not allowed anywhere: class weights, oversampling/undersampling, extra features, tuning on test.csv. Task 3 also: no dropout, no extra layers/neurons, no changed settings.
- **Fixed colours** (import from `src/style.py`, never hard-code differently):
  Hero/Good `#1F5FBF` · Villain/Bad `#D62839` · Baseline `#9E9E9E` · Decision Tree `#2E7D32` · KNN `#F9A825` · Network A `#6A1B9A` · Network B `#00838F` · Clusters: `tab10` in order (Cluster 1 = 1st colour).
- **Every graph/table:** title, labelled axes, legend if >1 colour, and **1–2 sentences underneath** (what it shows + what you conclude). Number them **G1–G16** and **T1–T11** exactly as the brief does.
- **Confusion matrix format:** rows = actual, columns = predicted, order Good (Hero) then Bad (Villain), count in every cell. Villain (1) is the positive class.

## 1. Repo layout

```
SHIH/
├── superhero_project.ipynb      # FINAL deliverable (assembled by Wasay)
├── data/
│   ├── train.csv                # from huggingface.co/datasets/jrtec/Superheroes
│   └── test.csv
├── src/
│   ├── prep.py                  # Section 2 — shared cleaning (Wasay, ships first)
│   └── style.py                 # colours, plot helpers, confusion-matrix helper
├── notebooks/
│   ├── 01_data_prep.ipynb       # Wasay
│   ├── 02_task1_classification.ipynb   # Wasay
│   ├── 03_task2_clustering.ipynb       # Rayyan
│   └── 04_task3_neural_net.ipynb       # Wasay
├── figures/                     # G1–G16 exported as PNG (G12 drawn by hand)
├── report/                      # report with Q1–Q12, tables, figures
├── requirements.txt
└── PLAN.md
```

Notebooks are kept one-per-owner so we never have merge conflicts inside `.ipynb` JSON. Read data **only** with `pd.read_csv("data/train.csv")` / `pd.read_csv("data/test.csv")` (paths relative to the repo root, as the brief requires).

### Workflow
- Branch per piece: `wasay/prep`, `wasay/task1`, `wasay/task3`, `rayyan/task2-clustering`, `wasay/final`. Open a PR into `main`; the other person skims it before merge.
- Don't edit files the other person owns. If you need a change in `src/prep.py` or `src/style.py`, ask in the PR / issue.
- Environment: **Python 3.11–3.12** (TensorFlow does not support 3.14 yet). Install with `pip install -r requirements.txt` (pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, tensorflow, jupyter).
  - **Wasay (WSL, done):** venv at `~/venvs/shih` (Python 3.12.16, TF 2.20, sklearn 1.7.2), Jupyter kernel **"Python 3.12 (shih)"**. Run from the repo: `cd /mnt/c/Projects/Projects/SHIS && source ~/venvs/shih/bin/activate && jupyter lab` — or open the folder in VS Code via WSL and pick that kernel.
  - **Rayyan:** any Python 3.11/3.12 → `python -m venv .venv`, activate, `pip install -r requirements.txt`. Task 2 doesn't need TensorFlow, so `pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter` is enough if TF gives trouble.

## 2. Interface contract (`src/prep.py`) — the only thing Task 2 depends on

```python
FEATURES = ["intelligence_score", "strength_score", "speed_score",
            "durability_score", "power_score", "combat_score"]

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]          # train, test (step 1 input)
def clean(train, test) -> dict:
    # returns {
    #   "set_a":       DataFrame  # cleaned train.csv, ALL alignments, cols: name, FEATURES, alignment  (NaNs still present)
    #   "set_b_train": DataFrame  # cleaned train.csv, Good/Bad only, + target y (Good=0, Bad=1)
    #   "set_b_test":  DataFrame  # cleaned test.csv,  Good/Bad only, + target y
    #   "t1": ..., "t2": ..., "nan_counts": ...   # tables for the report
    # }
```

NaNs are **left in** on purpose — each task fills them with a median imputer inside its own Pipeline (step 5 is training-data only).

---

## 3. Section 2 · Data Preparation (10 marks) — **Wasay** — target: Sat 28 Nov

Facts already checked on the real files: train.csv = 1158 rows × 83 cols, test.csv = 290 × 83. Alignment values are exactly `Good / Bad / Neutral / ""`. Scores have **0s but no blanks** (≈85–91 zeros per column in train, ≈20 in test). Raw files have no duplicate names and no names shared between files — still run and report the steps.

1. **Download** `train.csv`, `test.csv` from Hugging Face into `data/` and commit them (both of us use identical files).
2. **Step 1 · Inspect** → **T1**: per file — shape, column names, dtypes, missing counts. Also, per score column, how many 0s and how many blanks became NaN.
3. **Step 2 · Remove duplicates**: same `name` twice in a file → keep first. Name in both files → drop from train.csv. "Captain X" and "Captain X (Film)" are different — exact match only.
4. **Step 3 · Mark missing**: `pd.to_numeric(errors="coerce")` on the 6 scores; blanks and `0` → `NaN`.
5. **Step 4 · Remove weak rows**: drop rows with **≥ 3** of the 6 scores missing.
6. Steps 1–4 applied identically to both files (one function, called twice).
7. **Clean Set A** = train after steps 1–4, all alignments (alignment kept aside, not an input).
8. **Clean Set B** = train and test after steps 1–4, only Good/Bad; `y = {Good: 0, Bad: 1}`. Report Neutral and blank rows removed per file.
9. **T2** row counts: Loaded · After step 2 · After step 4 · Clean Set A (train only) · Clean Set B — for both files.
10. Short markdown under T1/T2; a diagram-free explanation of why median / why train-only / why ≥3 rule.
11. Write `src/style.py`: `COLORS`, `CLUSTER_COLORS = plt.cm.tab10.colors`, `plot_confusion(y_true, y_pred, title, ax)` in the required format.

**Done when:** `from src.prep import load_raw, clean` works from any notebook and T1/T2 render. Merge to `main` ASAP — this unblocks Rayyan.

## 4. Section 3 · Task 1 · Classification (25 marks) — **Wasay** — target: Mon 30 Nov

Data: Clean Set B. Train on all of train.csv, test on test.csv once. Main metric **Macro-F1**.

| Model | Pipeline | Grid | Fixed |
|---|---|---|---|
| Baseline | `DummyClassifier` | — | `strategy="most_frequent"` |
| Decision Tree | `SimpleImputer(median)` → `DecisionTreeClassifier` | `max_depth`: 3,5,7,10,None × `min_samples_leaf`: 1,5,10 (15) | `criterion="gini"`, `random_state=42` |
| KNN | `SimpleImputer(median)` → `StandardScaler` → `KNeighborsClassifier` | `n_neighbors`: 3,5,7,9,11,15 × `weights`: uniform, distance (12) | `metric="euclidean"` |

Search: `GridSearchCV(pipe, grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="f1_macro", refit=True)`.
Baseline: give it the imputer too (harmless) or fit directly — it ignores features.

Steps: load X (6 scores) / y → build pipelines → grid search → refit on all train → predict test.csv once → `classification_report(y_test, y_pred, digits=3)`.

Deliverables
- **G1** class counts: Good/Bad bars for train and test side by side, % labels.
- **G2** confusion matrix DT · **G3** confusion matrix KNN (required format).
- **G4** grouped bars: Accuracy, Macro-Precision, Macro-Recall, Macro-F1 for the 3 models, y 0–1.
- **T3** class counts and % per file · **T4** best params + best CV macro-F1 · **T5** results table (Train Macro-F1 on full train.csv; everything else on test.csv; Accuracy, P/R/F1 per class, Macro-F1).
- Winner rules: highest test Macro-F1; gap < 0.01 → higher Villain recall wins; "useful" only if > Baseline; overfits if Train − Test Macro-F1 > 0.10.
- **Q1** winner with numbers · **Q2** beat Baseline by how much · **Q3** harder class (from CMs) · **Q4** overfitting (train–test gap).

Save the 3 fitted models' test metrics (Accuracy, Macro-F1, Villain recall) to a dict / `results/task1_test.json` — Task 3's T11/G16 need them.

## 5. Section 4 · Task 2 · Clustering (25 marks) — **Rayyan** — target: Thu 3 Dec (PR open)

Data: **Clean Set A** from `src.prep.clean(...)["set_a"]`. Inputs: the 6 scores only. No test.csv. `alignment` not touched until step 8.

**Step-by-step**
1. `X_raw = set_a[FEATURES]`; `X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_raw))` (or a 2-step Pipeline). All metrics are computed on this scaled `X`.
2. **K-Means** for k = 2…8: `KMeans(n_clusters=k, n_init=10, random_state=42)`; record `inertia_` and `silhouette_score(X, labels)`.
3. **Choose k** = highest Silhouette; tie to 3 decimals → smaller k. Describe the Elbow, but it does **not** change the choice.
4. **Agglomerative:** `AgglomerativeClustering(n_clusters=k, linkage="ward")` with the same k.
5. **Score both:** `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` (on scaled X) + `adjusted_rand_score(kmeans_labels, agglo_labels)`.
6. **Renumber** clusters for each algorithm: compute each cluster's mean of the 6 scores on the original 0–100 scale (use the median-imputed, unscaled values); lowest average → Cluster 1, next → Cluster 2 … Use `tab10` colour i for Cluster i in every plot.
7. **PCA** `PCA(n_components=2, random_state=42)` fit **once** on scaled X, used for both scatter plots.
8. **Explain:** profiles, a few example `name`s per cluster, PCA plots, **then** the alignment mix.

**Graphs**
- **G5** Elbow: inertia vs k (2–8).
- **G6** Silhouette vs k, final k marked.
- **G7** Dendrogram: `scipy.cluster.hierarchy.linkage(X, method="ward")` → `dendrogram(Z, truncate_mode="lastp", p=30)`; dashed horizontal line where the tree splits into k groups (e.g. `y = (Z[-k, 2] + Z[-k+1, 2]) / 2`).
- **G8** PCA scatter K-Means · **G9** PCA scatter Agglomerative — same projection; axes "PC1 (xx.x %)", "PC2 (xx.x %)" from `explained_variance_ratio_`.
- **G10** profiles K-Means · **G11** profiles Agglomerative: x = 6 scores, one bar per cluster, y = average 0–100.

**Tables**
- **T6** Algorithm · k · Silhouette ↑ · Davies–Bouldin ↓ · Calinski–Harabasz ↑ · Wins (of 3), plus an "ARI between the two" row.
- **T7** (one per algorithm) Cluster · Size · INT · STR · SPD · DUR · POW · COM · short name you give (e.g. "Low-power civilians", "All-round powerhouses").
- **T8** % Good / Bad / Neutral / blank in each cluster (both algorithms).

**Comparison rules:** better separated = wins ≥ 2 of 3 metrics; agreement via ARI bands (≥ 0.70 high, 0.40–0.69 moderate, < 0.40 low) backed by profiles; easier to interpret = profiles (G10/G11) differ more clearly. Silhouette bands: 0.71–1 strong · 0.51–0.70 reasonable · 0.26–0.50 weak · ≤ 0.25 no real structure. Never use accuracy/precision/recall here.

**Answers:** **Q5** chosen k + what the Elbow says · **Q6** similar groups? (ARI + profiles) · **Q7** better separated vs easier to interpret · **Q8** each cluster in plain words; is any cluster mostly villains?

**Hand-off format** (so Wasay can drop it straight into the final notebook):
- `notebooks/03_task2_clustering.ipynb` runs top-to-bottom from a fresh kernel with only `from src.prep import ...` / `from src.style import ...` imports.
- Every G/T has its 1–2 sentence caption in a markdown cell right under it.
- PNGs saved as `figures/G05_elbow.png` … `figures/G11_profiles_agglo.png`.
- Q5–Q8 answers written in markdown cells at the end of the notebook (final numbers, not placeholders).

## 6. Section 5 · Task 3 · Neural Network (25 marks) — **Wasay** — target: Thu 3 Dec

Data: Clean Set B train → `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` → 80 % fit / 20 % validation. Imputer (median) + StandardScaler fit on the **80 % only**, then transform validation and test.

| | Network A | Network B |
|---|---|---|
| Layers | 6 → Dense(8, ReLU) → Dense(1, sigmoid) | 6 → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, sigmoid) |
| Trainable params | 65 (check `model.summary()`) | 257 |

Shared: TF/Keras, `tf.keras.utils.set_random_seed(42)` before building each model, loss `binary_crossentropy`, `Adam(learning_rate=0.001)`, batch 32, max 200 epochs, `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)`, threshold 0.5 (≥ 0.5 → Villain).

Evaluate train (80 %), validation, test separately: Accuracy, Macro-F1 (all three), log loss (val, test), per-class P/R/F1 + confusion matrix (test).

Deliverables
- **G12** architecture diagrams A and B (draw.io / PowerPoint — the only non-code graph).
- **G13 / G14** loss curves: x = epoch, y = BCE loss, training + validation lines, dashed vertical line at best epoch (lowest val loss).
- **G15** confusion matrices A and B (required format).
- **G16** grouped bars: Accuracy and Macro-F1 for all 5 models on test.csv.
- **T9** settings + best epoch · **T10** train/val/test results (+ per-class test P/R/F1 below it) · **T11** final ranking of all 5 models: Rank · Model · Test accuracy · Test Macro-F1 · Villain recall · Beats baseline?
- Rules: B "improved results" only if test Macro-F1 ≥ A + 0.01; overfit if train − test Macro-F1 > 0.10 or val loss rises while train loss falls. Ranking: test Macro-F1, gap < 0.01 → higher Villain recall.
- Note in the comparison that DT/KNN learned from all of train.csv while networks learned from 80 %.
- **Q9** B vs A by ≥ 0.01? why · **Q10** over/underfitting, best epoch · **Q11** did a network beat DT and KNN (numbers) · **Q12** two limitations (e.g. small, imbalanced data; 0 = unknown imputation; ability scores ignore story/context; single fixed split for networks; run-to-run variance).

## 7. Integration & submission — **Wasay** — Fri 4 – Sat 5 Dec

1. Merge Rayyan's PR, then rebuild the deliverables (WSL, from the repo root, venv active):
   ```bash
   python tools/build_final.py        # assembles superhero_project.ipynb (src/ code inlined, Task 2 slotted in)
   jupyter nbconvert --to notebook --execute --inplace superhero_project.ipynb
   jupyter nbconvert --to html --no-input --output-dir report --output superhero_report superhero_project.ipynb
   ```
   Then print `report/superhero_report.html` to `report/superhero_report.pdf` (browser → Print → Save as PDF, or headless Edge `--print-to-pdf`). Until Task 2 lands, the build inserts a marked placeholder for Section 4.
2. **Restart & Run All** from a fresh kernel; confirm all G1–G16 / T1–T11 render, numbering and colours match the brief, every figure has its caption.
3. Report (`report/`): Q1–Q12 with numbers quoted, all tables/figures, team names + roll numbers. Export PDF.
4. Final checklist pass (section 0 above) — especially: no `overall_score`, no fit on test, test opened once, alignment unseen by clustering.
5. Zip as `24I-3071_<partner-roll>_Assignment03.zip` (notebook + `data/` + report PDF + G12 image) and submit **by Sat 5 Dec night** — one day of buffer before the 6 Dec 11:59 PM deadline.

## 8. Timeline

| Date | Wasay (70 %) | Rayyan (30 %) |
|---|---|---|
| Fri 27 Nov | Repo setup, data/, requirements, invite | Accept invite, set up venv, read Section 4 of the brief |
| Sat 28 Nov | Section 2 + `src/prep.py`, `src/style.py` → merged | Start Task 2 on `main` (steps 1–3, G5, G6) |
| Sun 29 – Mon 30 Nov | Task 1 (T3–T5, G1–G4, Q1–Q4) | Agglomerative, metrics, T6, G7 |
| Tue 1 – Wed 2 Dec | Task 3 (T9, T10, G12–G15) | Renumbering, PCA, G8–G11, T7, T8 |
| Thu 3 Dec | T11, G16, Q9–Q12 | Captions, Q5–Q8, **open PR** |
| Fri 4 Dec | Review + merge Task 2, assemble final notebook | Review Wasay's notebook for rule breaks |
| Sat 5 Dec | Report, Run-All check, **submit** | Final proofread |
| Sun 6 Dec 23:59 | Hard deadline (buffer only) | |
