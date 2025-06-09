# preprocess.py
# --------------
# Reads reports.xlsx, enumerates all summary‐pairs per report, and writes pairs.json.

import pandas as pd
import itertools
import json

def make_pairwise_dataset(excel_path: str, json_path: str):
    df = pd.read_excel(excel_path)
    records = []
    for _, row in df.iterrows():
        report_id = row["report_id"]
        # collect (label, text, rank) for non-null summaries
        summaries = []
        for i in range(1, 5):
            txt = row.get(f"summary_{i}")
            rank = row.get(f"rank_{i}")
            if pd.notna(txt) and pd.notna(rank):
                summaries.append((f"S{i}", str(txt).strip(), int(rank)))
        # all unordered pairs
        for (id1, txt1, r1), (id2, txt2, r2) in itertools.combinations(summaries, 2):
            # lower rank number = better
            better = "A" if r1 < r2 else "B"
            records.append({
                "report_id": report_id,
                "question": "Which of the following two summaries is better?",
                "summary_a": txt1,
                "summary_b": txt2,
                "answer": better
            })
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    make_pairwise_dataset("reports.xlsx", "pairs.json")
    print("→ Wrote pairs.json")
