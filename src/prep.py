"""Section 2 - Data Preparation.

Steps 1-4 of the brief, applied to train.csv and test.csv in exactly the same way:

    1. Inspect          -> Table T1 (shape, columns, dtypes, missing counts)
    2. Remove duplicates (same name twice in a file -> keep first;
                          name in both files -> drop it from train.csv)
    3. Mark missing     (6 scores made numeric; blanks and 0 -> NaN)
    4. Remove weak rows (3 or more of the 6 scores missing)

Step 5 (median fill) is NOT done here: each task does it inside its own
scikit-learn Pipeline so the medians are learned from training data only.
That is why the clean sets returned below still contain NaNs.

Usage from a notebook (run from the repo root):

    from src.prep import FEATURES, load_raw, clean
    train_raw, test_raw = load_raw()
    data = clean(train_raw, test_raw)
    data["set_a"], data["set_b_train"], data["set_b_test"], data["t1"], data["t2"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "intelligence_score",
    "strength_score",
    "speed_score",
    "durability_score",
    "power_score",
    "combat_score",
]
SHORT_NAMES = dict(zip(FEATURES, ["INT", "STR", "SPD", "DUR", "POW", "COM"]))

TARGET_MAP = {"Good": 0, "Bad": 1}          # 0 = Hero, 1 = Villain
CLASS_NAMES = ["Good (Hero)", "Bad (Villain)"]
MAX_MISSING = 2                             # keep rows with at most 2 of the 6 scores missing

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


# ---------------------------------------------------------------- loading

def load_raw(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    """Read the two files exactly as given. They are never merged."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


# ---------------------------------------------------------------- step 1

def inspect(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Table T1 (part 1): column names, dtypes and missing counts for both files."""
    t1 = pd.DataFrame({
        "dtype (train)": train.dtypes.astype(str),
        "missing (train)": train.isna().sum(),
        "dtype (test)": test.dtypes.astype(str),
        "missing (test)": test.isna().sum(),
    })
    t1.index.name = "column"
    return t1


def shapes(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"rows": [len(train), len(test)], "columns": [train.shape[1], test.shape[1]]},
        index=pd.Index(["train.csv", "test.csv"], name="file"),
    )


# ---------------------------------------------------------------- step 2

def remove_duplicates(train: pd.DataFrame, test: pd.DataFrame):
    """Same name twice in one file -> keep the first row.
    A name found in both files -> remove it from train.csv (test stays untouched).
    Names are compared exactly, so "Captain X" and "Captain X (Film)" stay separate."""
    train = train.drop_duplicates(subset="name", keep="first")
    test = test.drop_duplicates(subset="name", keep="first")
    train = train[~train["name"].isin(set(test["name"]))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------- step 3

def mark_missing(df: pd.DataFrame):
    """Make the 6 scores numeric and turn blanks and 0 into NaN.
    A 0 means "unknown", not "no ability". Returns (clean_df, counts_df)."""
    df = df.copy()
    counts = {}
    for col in FEATURES:
        raw = df[col]
        blank = raw.isna() | raw.astype(str).str.strip().eq("")
        num = pd.to_numeric(raw, errors="coerce")
        bad_text = num.isna() & ~blank              # non-numeric text, if any
        zero = num.eq(0)
        df[col] = num.mask(zero)
        counts[col] = {
            "0 -> NaN": int(zero.sum()),
            "blank -> NaN": int(blank.sum()),
            "text -> NaN": int(bad_text.sum()),
            "total NaN": int(df[col].isna().sum()),
        }
    return df, pd.DataFrame(counts).T.rename_axis("score")


# ---------------------------------------------------------------- step 4

def remove_weak_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every character with 3 or more of its 6 scores missing."""
    n_missing = df[FEATURES].isna().sum(axis=1)
    return df[n_missing <= MAX_MISSING].reset_index(drop=True)


# ---------------------------------------------------------------- clean sets

def _clean_alignment(s: pd.Series) -> pd.Series:
    """Strip spaces; empty text becomes NaN (= "blank" alignment)."""
    return s.map(lambda v: v.strip() if isinstance(v, str) else v).replace("", np.nan)


def _set_b(df: pd.DataFrame):
    """Good/Bad rows only, with target y: Good -> 0 (Hero), Bad -> 1 (Villain)."""
    keep = df["alignment"].isin(TARGET_MAP)
    removed = {
        "Neutral removed": int(df["alignment"].eq("Neutral").sum()),
        "blank removed": int(df["alignment"].isna().sum()),
    }
    out = df.loc[keep, ["name", *FEATURES, "alignment"]].copy()
    out["y"] = out["alignment"].map(TARGET_MAP).astype(int)
    return out.reset_index(drop=True), removed


def clean(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> dict:
    """Run steps 1-4 on both files and build Clean Set A and Clean Set B.

    Returns a dict:
        set_a        cleaned train.csv, every character (name, 6 scores, alignment) - Task 2
        set_b_train  cleaned train.csv, Good/Bad only, + y                         - Tasks 1 & 3
        set_b_test   cleaned test.csv,  Good/Bad only, + y                         - Tasks 1 & 3
        shapes       raw shape of each file                                        - T1
        t1           column / dtype / missing count for each file                  - T1
        nan_counts   0s and blanks turned into NaN, per score and file             - T1
        t2           row counts after every step                                   - T2
        set_b_removed  Neutral and blank rows dropped from each file for Set B
    NaNs are left in the score columns on purpose (step 5 happens in each Pipeline).
    """
    counts = {"Loaded": (len(train_raw), len(test_raw))}

    # step 1
    shape_tbl = shapes(train_raw, test_raw)
    t1 = inspect(train_raw, test_raw)

    # step 2
    train, test = remove_duplicates(train_raw, test_raw)
    counts["After step 2 (duplicates)"] = (len(train), len(test))

    # step 3
    train, nan_train = mark_missing(train)
    test, nan_test = mark_missing(test)
    nan_counts = pd.concat({"train.csv": nan_train, "test.csv": nan_test}, axis=1)
    counts["After step 3 (mark missing)"] = (len(train), len(test))

    # step 4
    train = remove_weak_rows(train)
    test = remove_weak_rows(test)
    counts["After step 4 (weak rows)"] = (len(train), len(test))

    for df in (train, test):
        df["alignment"] = _clean_alignment(df["alignment"])

    set_a = train[["name", *FEATURES, "alignment"]].copy()
    set_b_train, rem_train = _set_b(train)
    set_b_test, rem_test = _set_b(test)
    counts["Clean Set A"] = (len(set_a), "not used")
    counts["Clean Set B"] = (len(set_b_train), len(set_b_test))

    t2 = pd.DataFrame.from_dict(counts, orient="index", columns=["train.csv", "test.csv"])
    t2.index.name = "step"

    return {
        "set_a": set_a,
        "set_b_train": set_b_train,
        "set_b_test": set_b_test,
        "shapes": shape_tbl,
        "t1": t1,
        "nan_counts": nan_counts,
        "t2": t2,
        "set_b_removed": pd.DataFrame({"train.csv": rem_train, "test.csv": rem_test}),
    }


def xy(set_b: pd.DataFrame):
    """X = the 6 scores (NaNs kept), y = 0/1. Use for Tasks 1 and 3."""
    return set_b[FEATURES], set_b["y"]
