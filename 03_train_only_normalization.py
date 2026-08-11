import json
import os
import re

import numpy as np
import pandas as pd

INPUT_DIR = "./outputs/02_batch_split"
OUTPUT_DIR = "./outputs/03_normalization"

Y_COLS = ["Knee_Cycle", "EOL_Cycle"]

def scale_values(values, mn, mx):
    den = mx - mn
    if np.isclose(den, 0):
        return np.zeros_like(values, dtype=float)
    return (values - mn) / den

def get_feature_groups(columns):
    scalar = []
    groups = {
        "SampQ": [],
        "SampV": [],
        "IC": [],
        "DV": [],
        "dQdI": [],
        "dIdQ": [],
    }
    separators = []

    for c in columns:
        if c.startswith("Sep_"):
            separators.append(c)
        elif "_Start_" in c or "_End_" in c:
            scalar.append(c)
        elif "_SampQ_" in c:
            groups["SampQ"].append(c)
        elif "_SampV_" in c:
            groups["SampV"].append(c)
        elif re.search(r"_IC_\d+$", c):
            groups["IC"].append(c)
        elif re.search(r"_DV_\d+$", c):
            groups["DV"].append(c)
        elif re.search(r"_dQdI_\d+$", c):
            groups["dQdI"].append(c)
        elif re.search(r"_dIdQ_\d+$", c):
            groups["dIdQ"].append(c)
        else:
            raise ValueError(f"Unrecognized feature type: {c}")

    return scalar, groups, separators

def normalize_x(train_raw, test_raw):
    train = train_raw.copy()
    test = test_raw.copy()

    cols = [c for c in train.columns if c != "Cell_ID"]
    scalar, groups, separators = get_feature_groups(cols)

    params = {
        "scalar_columnwise": {},
        "sequence_global_by_type": {},
        "separators": separators,
    }

    for c in scalar:
        mn = float(train_raw[c].min())
        mx = float(train_raw[c].max())

        train[c] = scale_values(
            train_raw[c].to_numpy(dtype=float), mn, mx
        )
        test[c] = scale_values(
            test_raw[c].to_numpy(dtype=float), mn, mx
        )

        params["scalar_columnwise"][c] = {
            "min": mn,
            "max": mx,
        }

    for name, group_cols in groups.items():
        if not group_cols:
            continue

        train_values = train_raw[group_cols].to_numpy(dtype=float)
        mn = float(np.nanmin(train_values))
        mx = float(np.nanmax(train_values))

        train[group_cols] = scale_values(
            train_raw[group_cols].to_numpy(dtype=float), mn, mx
        )
        test[group_cols] = scale_values(
            test_raw[group_cols].to_numpy(dtype=float), mn, mx
        )

        params["sequence_global_by_type"][name] = {
            "min": mn,
            "max": mx,
            "columns": group_cols,
        }

    return train, test, params

def normalize_y(train_raw, test_raw):
    train = train_raw.copy()
    test = test_raw.copy()
    params = {}

    for c in Y_COLS:
        mn = float(train_raw[c].min())
        mx = float(train_raw[c].max())

        train[c] = scale_values(
            train_raw[c].to_numpy(dtype=float), mn, mx
        )
        test[c] = scale_values(
            test_raw[c].to_numpy(dtype=float), mn, mx
        )

        params[c] = {
            "min": mn,
            "max": mx,
        }

    return train, test, params

def validate_alignment(x_df, y_df, name):
    if not np.array_equal(
        x_df["Cell_ID"].astype(int).to_numpy(),
        y_df["Cell_ID"].astype(int).to_numpy(),
    ):
        raise ValueError(f"{name} X/Y Cell_ID order is inconsistent.")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_x_raw = pd.read_pickle(
        os.path.join(INPUT_DIR, "train_x_raw.pkl")
    )
    test_x_raw = pd.read_pickle(
        os.path.join(INPUT_DIR, "test_x_raw.pkl")
    )
    train_y_raw = pd.read_pickle(
        os.path.join(INPUT_DIR, "train_y_raw.pkl")
    )
    test_y_raw = pd.read_pickle(
        os.path.join(INPUT_DIR, "test_y_raw.pkl")
    )

    validate_alignment(train_x_raw, train_y_raw, "Training")
    validate_alignment(test_x_raw, test_y_raw, "Test")

    if len(train_x_raw) != 117 or len(test_x_raw) != 28:
        raise ValueError(
            f"Expected 117/28 cells, got "
            f"{len(train_x_raw)}/{len(test_x_raw)}."
        )

    train_feature_cols = [
        c for c in train_x_raw.columns if c != "Cell_ID"
    ]
    test_feature_cols = [
        c for c in test_x_raw.columns if c != "Cell_ID"
    ]

    if train_feature_cols != test_feature_cols:
        raise ValueError("Train/test feature columns are not identical.")
    if len(train_feature_cols) != 244:
        raise ValueError(
            f"Expected 244 features, found {len(train_feature_cols)}."
        )

    train_x, test_x, x_params = normalize_x(
        train_x_raw, test_x_raw
    )
    train_y, test_y, y_params = normalize_y(
        train_y_raw, test_y_raw
    )

    train_x.to_pickle(os.path.join(OUTPUT_DIR, "train_x.pkl"))
    test_x.to_pickle(os.path.join(OUTPUT_DIR, "test_x.pkl"))
    train_y.to_pickle(os.path.join(OUTPUT_DIR, "train_y.pkl"))
    test_y.to_pickle(os.path.join(OUTPUT_DIR, "test_y.pkl"))

    train_x.to_csv(
        os.path.join(OUTPUT_DIR, "Train_Normalized.csv"),
        index=False,
    )
    test_x.to_csv(
        os.path.join(OUTPUT_DIR, "Test_Normalized.csv"),
        index=False,
    )
    train_y.to_csv(
        os.path.join(OUTPUT_DIR, "Train_Y_Normalized.csv"),
        index=False,
    )
    test_y.to_csv(
        os.path.join(OUTPUT_DIR, "Test_Y_Normalized.csv"),
        index=False,
    )

    with open(
        os.path.join(OUTPUT_DIR, "normalization_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {"X": x_params, "Y": y_params},
            f,
            indent=2,
            ensure_ascii=False,
        )

    scalar, groups, separators = get_feature_groups(
        train_feature_cols
    )

    train_range = train_x.drop(
        columns="Cell_ID"
    ).to_numpy(dtype=float)
    test_range = test_x.drop(
        columns="Cell_ID"
    ).to_numpy(dtype=float)

    print("=" * 72)
    print("Part 3 completed: training-set-only normalization")
    print(f"Training cells: {len(train_x)}")
    print(f"Test cells:     {len(test_x)}")
    print(f"Feature dims:   {len(train_feature_cols)}")
    print(f"Scalar cols:    {len(scalar)}")
    print(
        "Sequence groups: "
        + ", ".join(
            f"{name}={len(cols)}"
            for name, cols in groups.items()
        )
    )
    print(f"Separators:     {len(separators)}")
    print(
        f"Knee train min/max: "
        f"{y_params['Knee_Cycle']['min']:.0f}/"
        f"{y_params['Knee_Cycle']['max']:.0f}"
    )
    print(
        f"EOL train min/max:  "
        f"{y_params['EOL_Cycle']['min']:.0f}/"
        f"{y_params['EOL_Cycle']['max']:.0f}"
    )
    print(
        f"Normalized train_x range: "
        f"{train_range.min():.6f} ~ {train_range.max():.6f}"
    )
    print(
        f"Normalized test_x range:  "
        f"{test_range.min():.6f} ~ {test_range.max():.6f}"
    )
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()
