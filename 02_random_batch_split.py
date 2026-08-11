import json
import os

import numpy as np
import pandas as pd

FEATURE_PATH = "./outputs/01_semantic_encoding/Master_Features_244D.csv"
LABEL_PATH = "./data/Batch3_145_ID_Knee_EOL80.csv"
OUTPUT_DIR = "./outputs/02_batch_split"

SPLIT_SEED = 40

TOTAL_CELLS = 145
TRAIN_SIZE = 117
TEST_SIZE = 28

Y_COLS = ["Knee_Cycle", "EOL_Cycle"]

def select_ids(df, ids, cols=None):
    """Select rows and preserve the exact randomized ID order."""
    table = df.copy()
    table["Cell_ID"] = table["Cell_ID"].astype(int)
    table = table.set_index("Cell_ID").loc[ids].reset_index()

    if cols is not None:
        table = table[["Cell_ID"] + list(cols)]

    return table

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features = pd.read_csv(FEATURE_PATH)
    labels = pd.read_csv(LABEL_PATH)

    if "Cell_ID" not in features.columns:
        raise ValueError("Feature table must contain Cell_ID.")
    if "Cell_ID" not in labels.columns:
        raise ValueError("Label table must contain Cell_ID.")

    missing_y_cols = [c for c in Y_COLS if c not in labels.columns]
    if missing_y_cols:
        raise ValueError(f"Label table is missing target columns: {missing_y_cols}")

    features = features.copy()
    labels = labels.copy()
    features["Cell_ID"] = features["Cell_ID"].astype(int)
    labels["Cell_ID"] = labels["Cell_ID"].astype(int)

    if features["Cell_ID"].duplicated().any():
        raise ValueError("Duplicate Cell_ID values exist in the feature table.")
    if labels["Cell_ID"].duplicated().any():
        raise ValueError("Duplicate Cell_ID values exist in the label table.")

    all_ids = sorted(labels["Cell_ID"].tolist())

    if len(all_ids) != TOTAL_CELLS:
        raise ValueError(
            f"Expected exactly {TOTAL_CELLS} final cells in the label table, "
            f"but found {len(all_ids)}."
        )

    feature_ids = set(features["Cell_ID"].tolist())
    missing_features = sorted(set(all_ids) - feature_ids)
    if missing_features:
        raise ValueError(
            f"Feature table is missing these final Cell_ID values: {missing_features}"
        )

    feature_cols = [c for c in features.columns if c != "Cell_ID"]
    if len(feature_cols) != 244:
        raise ValueError(
            f"Expected 244 semantic features, but found {len(feature_cols)}."
        )

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled_ids = rng.permutation(np.asarray(all_ids, dtype=int)).tolist()

    train_ids = shuffled_ids[:TRAIN_SIZE]
    test_ids = shuffled_ids[TRAIN_SIZE:]

    if len(train_ids) != TRAIN_SIZE or len(test_ids) != TEST_SIZE:
        raise RuntimeError("Unexpected train/test split size.")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("Train/test Cell_ID overlap detected.")

    train_x_raw = select_ids(features, train_ids)
    test_x_raw = select_ids(features, test_ids)
    train_y_raw = select_ids(labels, train_ids, Y_COLS)
    test_y_raw = select_ids(labels, test_ids, Y_COLS)

    train_x_raw.to_pickle(os.path.join(OUTPUT_DIR, "train_x_raw.pkl"))
    test_x_raw.to_pickle(os.path.join(OUTPUT_DIR, "test_x_raw.pkl"))
    train_y_raw.to_pickle(os.path.join(OUTPUT_DIR, "train_y_raw.pkl"))
    test_y_raw.to_pickle(os.path.join(OUTPUT_DIR, "test_y_raw.pkl"))

    train_x_raw.to_csv(os.path.join(OUTPUT_DIR, "train_x_raw.csv"), index=False)
    test_x_raw.to_csv(os.path.join(OUTPUT_DIR, "test_x_raw.csv"), index=False)
    train_y_raw.to_csv(os.path.join(OUTPUT_DIR, "train_y_raw.csv"), index=False)
    test_y_raw.to_csv(os.path.join(OUTPUT_DIR, "test_y_raw.csv"), index=False)

    with open(
        os.path.join(OUTPUT_DIR, "train_cell_ids.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(map(str, train_ids)))

    with open(
        os.path.join(OUTPUT_DIR, "test_cell_ids.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(map(str, test_ids)))

    manifest = pd.DataFrame({
        "Cell_ID": train_ids + test_ids,
        "Subset": ["train"] * TRAIN_SIZE + ["test"] * TEST_SIZE,
    })
    manifest.to_csv(
        os.path.join(OUTPUT_DIR, "batch_manifest.csv"),
        index=False,
    )

    metadata = {
        "split_seed": SPLIT_SEED,
        "total_cells": TOTAL_CELLS,
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "feature_dimensions": len(feature_cols),
        "train_ids": train_ids,
        "test_ids": test_ids,
    }

    with open(
        os.path.join(OUTPUT_DIR, "batch_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Part 2 completed: random batch split")
    print(f"SPLIT_SEED: {SPLIT_SEED}")
    print(f"Final cells: {TOTAL_CELLS}")
    print(f"Training:    {len(train_ids)}")
    print(f"Test:        {len(test_ids)}")
    print(f"Features:    {len(feature_cols)}")
    print(f"Output:      {OUTPUT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()
