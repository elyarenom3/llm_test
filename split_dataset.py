# split_dataset.py
# ----------------
# Splits reports into train/val/test (70/15/15) at the report level,
# then filters pairs.json into train.json, val.json, test.json.

import pandas as pd
import json
from sklearn.model_selection import train_test_split

def split_reports(excel_path, pairs_json,
                  train_json, val_json, test_json,
                  seed=42):
    df = pd.read_excel(excel_path)
    report_ids = df["report_id"].unique().tolist()

    train_ids, temp_ids = train_test_split(
        report_ids, test_size=0.30, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=seed
    )

    with open(pairs_json, encoding="utf8") as f:
        pairs = json.load(f)

    def save_split(out_path, keep_ids):
        filtered = [ex for ex in pairs if ex["report_id"] in keep_ids]
        with open(out_path, "w", encoding="utf8") as g:
            json.dump(filtered, g, indent=2, ensure_ascii=False)

    save_split(train_json, train_ids)
    save_split(val_json,   val_ids)
    save_split(test_json,  test_ids)
    print(f"→ train: {len(train_ids)} reports → {len(json.load(open(train_json)))} pairs")
    print(f"→ val:   {len(val_ids)} reports → {len(json.load(open(val_json)))} pairs")
    print(f"→ test:  {len(test_ids)} reports → {len(json.load(open(test_json)))} pairs")

if __name__ == "__main__":
    split_reports(
        excel_path="reports.xlsx",
        pairs_json="pairs.json",
        train_json="train.json",
        val_json="val.json",
        test_json="test.json"
    )
