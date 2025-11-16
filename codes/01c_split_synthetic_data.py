import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/synthetic/synthetic_all.jsonl")
OUTPUT_A1 = Path("data/synthetic_approach1.jsonl")
OUTPUT_A2 = Path("data/synthetic_approach2.jsonl")

def main():
    if not INPUT_FILE.exists():
        return

    approach1_records, approach2_records = [], []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            cat = record.get("category", "").lower().strip()

            if cat in {"lexical", "syntactic", "semantic"}:
                approach1_records.append(record)
            elif cat == "temporal":
                approach2_records.append(record)

    if approach1_records:
        c1 = Counter(r.get("category", "unknown") for r in approach1_records)

    if approach2_records:
        c2 = Counter(r.get("period", "unknown") for r in approach2_records)

    def save_records(records, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as out:
            for rec in records:
                json.dump(rec, out, ensure_ascii=False)
                out.write("\n")

    save_records(approach1_records, OUTPUT_A1)
    save_records(approach2_records, OUTPUT_A2)


if __name__ == "__main__":
    main()
