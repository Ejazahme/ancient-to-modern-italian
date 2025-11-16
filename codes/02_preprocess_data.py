import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

A1_INPUT = DATA_DIR / "synthetic_approach1.jsonl"
A2_INPUT = DATA_DIR / "synthetic_approach2.jsonl"

A1_CATEGORIES = ["lexical", "syntactic", "semantic"]
A2_PERIODS = ["early_1260", "middle_1310", "late_1360"]


def load_jsonl(path: Path):
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if "source" not in df.columns or "target" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["source", "target"]).copy()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    df = df[(df["source"].str.len() > 10) & (df["target"].str.len() > 10)]

    df = df[df["source"] != df["target"]]

    before = len(df)
    df = df.drop_duplicates(subset=["source", "target"])
    after = len(df)
    if after < before:
        pass

    return df


def split_df(df: pd.DataFrame, label: str):
    if df.empty:
        return None, None, None

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if len(df) < 10:
        train_df = df
        val_df = df.iloc[0:0].copy()
        test_df = df.iloc[0:0].copy()
    else:
        train_df, temp = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)
    return train_df, val_df, test_df


def save_jsonl(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write("\n")
    


def process_approach1():
    df = load_jsonl(A1_INPUT)
    if df.empty:
        return

    df = clean_df(df)

    if "category" not in df.columns:
        return

    for cat in A1_CATEGORIES:
        sub = df[df["category"].str.lower() == cat].copy()
        if sub.empty:
            continue
        train_df, val_df, test_df = split_df(sub, f"Approach1/{cat}")
        if train_df is None:
            continue

        base = OUTPUT_DIR / "approach1" / cat
        save_jsonl(train_df, base / "train.jsonl")
        save_jsonl(val_df, base / "val.jsonl")
        save_jsonl(test_df, base / "test.jsonl")


def process_approach2():
    df = load_jsonl(A2_INPUT)
    if df.empty:
        return

    df = clean_df(df)

    if "period" not in df.columns:
        return

    for period in A2_PERIODS:
        sub = df[df["period"] == period].copy()
        if sub.empty:
            continue
        train_df, val_df, test_df = split_df(sub, f"Approach2/{period}")
        if train_df is None:
            continue

        base = OUTPUT_DIR / "approach2" / period
        save_jsonl(train_df, base / "train.jsonl")
        save_jsonl(val_df, base / "val.jsonl")
        save_jsonl(test_df, base / "test.jsonl")


def main():
    process_approach1()
    process_approach2()
    

if __name__ == "__main__":
    main()
