"""Assemble the final deliverable `superhero_project.ipynb` from the section notebooks.

    python tools/build_final.py          # run from the repo root

* Puts the sections in the brief's order: Data Prep -> Task 1 -> Task 2 -> Task 3.
* Inlines src/prep.py and src/style.py into a "Shared code" section, so the final
  notebook only needs data/ next to it (the brief's layout) - no src/ package.
* Strips each section's repo-only setup lines (chdir / sys.path / `from src...` imports).
* If notebooks/03_task2_clustering.ipynb does not exist yet, a clearly marked
  placeholder is inserted instead.

The result is written un-executed; run it top to bottom afterwards, e.g.
    jupyter nbconvert --to notebook --execute --inplace superhero_project.ipynb
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "superhero_project.ipynb"

SECTIONS = [
    ("notebooks/01_data_prep.ipynb", None),
    ("notebooks/02_task1_classification.ipynb", None),
    ("notebooks/03_task2_clustering.ipynb", "task2_placeholder"),
    ("notebooks/04_task3_neural_net.ipynb", None),
]

TEAM = [
    ("Muhammad Wasay Tariq", "24I-3071"),
    ("Syed Muhammad Rayyan Hassan", "24I-3000"),
]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


# ------------------------------------------------------------------ helpers

def module_source(path: Path) -> str:
    """Module code without its docstring and `from __future__` line."""
    src = path.read_text(encoding="utf8")
    tree = ast.parse(src)
    lines = src.splitlines()
    start = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
        start = tree.body[0].end_lineno
    body = [l for l in lines[start:] if not l.startswith("from __future__")]
    return "\n".join(body).strip() + "\n"


SETUP_PATTERNS = [
    re.compile(r"^if os\.path\.basename\(os\.getcwd\(\)\) == \"notebooks\""),
    re.compile(r"^\s+os\.chdir\(\"\.\.\"\)"),
    re.compile(r"^sys\.path\.insert\(0, os\.getcwd\(\)\)"),
    re.compile(r"^from src\."),
]


def strip_repo_setup(source: str) -> str:
    kept = [l for l in source.splitlines() if not any(p.search(l) for p in SETUP_PATTERNS)]
    while kept and not kept[0].strip():
        kept.pop(0)
    return "\n".join(kept)


TEXT_FIXES = {
    "The cleaning code lives in `src/prep.py` so every task uses exactly the same rules.":
        "The cleaning functions are defined once in the *Shared code* section above, so every task uses exactly the same rules.",
}


def clean_markdown(source: str) -> str:
    lines = [l for l in source.splitlines()
             if not l.startswith("# Superhero Intelligence System") and not l.startswith("**Team:**")]
    text = "\n".join(lines).strip()
    for old, new in TEXT_FIXES.items():
        text = text.replace(old, new)
    return text


def section_cells(path: Path) -> list:
    nb = nbf.read(path, as_version=4)
    cells = []
    for c in nb.cells:
        if c.cell_type == "markdown":
            text = clean_markdown(c.source)
            if text:
                cells.append(md(text))
        elif c.cell_type == "code":
            src = strip_repo_setup(c.source)
            if src.strip():
                cells.append(code(src))
    return cells


def task2_placeholder() -> list:
    return [md("""## Section 4 · Task 2 · Clustering (25 marks)

> **Placeholder — Task 2 (Syed Muhammad Rayyan Hassan) has not been merged yet.**
> Re-run `python tools/build_final.py` once `notebooks/03_task2_clustering.ipynb` is on `main`.

Will contain: G5 Elbow curve · G6 Silhouette vs k · G7 Dendrogram · G8/G9 PCA scatter (K-Means / Agglomerative) · G10/G11 cluster profiles · T6 clustering metrics + ARI · T7 cluster profiles · T8 alignment mix · Q5–Q8.""")]


# ------------------------------------------------------------------ build

def build() -> nbf.NotebookNode:
    team = " · ".join(f"{n} ({r})" for n, r in TEAM)
    task2_ready = (ROOT / "notebooks/03_task2_clustering.ipynb").exists()

    cells = [
        md(f"""# Superhero Intelligence System
### Can six ability scores tell a hero from a villain?
**Machine Learning · Assignment 03**
**Team:** {team}

| Section | Content | Graphs | Tables | Questions |
|---|---|---|---|---|
| 2 · Data Preparation | clean both files with fixed rules, build Clean Set A and B | — | T1, T2 | — |
| 3 · Task 1 · Classification | Baseline, Decision Tree, KNN | G1–G4 | T3–T5 | Q1–Q4 |
| 4 · Task 2 · Clustering | K-Means, Agglomerative (Ward) | G5–G11 | T6–T8 | Q5–Q8 |
| 5 · Task 3 · Neural Network | Network A, Network B, final ranking of all 5 models | G12–G16 | T9–T11 | Q9–Q12 |

Data: `data/train.csv` and `data/test.csv` from huggingface.co/datasets/jrtec/Superheroes, read separately, never merged. test.csv is used only once, for the final scores of Task 1 and Task 3.
{"" if task2_ready else chr(10) + "> **Draft:** Task 2 is still a placeholder in this version." + chr(10)}"""),
        md("""## Shared code
Setup used by every section: the six input columns, the cleaning rules of Section 2, the fixed colours from the brief, and the confusion-matrix / figure helpers. Everything uses `random_state=42`."""),
        code("""import os, sys, json, functools, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display"""),
        code("# ---- data preparation (Section 2) ----\n" + module_source(ROOT / "src/prep.py")),
        code("# ---- colours and plot helpers ----\n" + module_source(ROOT / "src/style.py")),
    ]

    for rel, fallback in SECTIONS:
        path = ROOT / rel
        if path.exists():
            cells += section_cells(path)
        elif fallback == "task2_placeholder":
            cells += task2_placeholder()
        else:
            raise FileNotFoundError(rel)

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "shih", "display_name": "Python 3.12 (shih)", "language": "python"}
    nb.metadata["title"] = "Superhero Intelligence System"
    return nb


if __name__ == "__main__":
    nb = build()
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(nb.cells)} cells)")
