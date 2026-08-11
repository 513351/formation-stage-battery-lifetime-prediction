import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# ============================================================

NUM_SAMPLES = 10

CC_DELTA_V = 0.010
CC_DELTA_Q = 0.002
CV_DELTA_I = 0.00005
CV_DELTA_Q = 0.00005

FILTER_WINDOW = 21
FILTER_POLY = 2


def smooth_curve(data):
    """Interpolate missing values and optionally apply Savitzky-Golay smoothing."""
    s = (
        pd.Series(data)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0)
    )

    if not HAS_SCIPY or len(s) < FILTER_WINDOW:
        return s.values

    try:
        window = FILTER_WINDOW if FILTER_WINDOW % 2 != 0 else FILTER_WINDOW + 1
        smoothed = savgol_filter(
            s.values,
            window_length=window,
            polyorder=FILTER_POLY,
        )
        return smoothed
    except Exception:
        return s.values


def get_adaptive_start_index(signal, min_skip=5, max_skip_ratio=0.15):
    """Determine an adaptive starting index for feature sampling."""
    n = len(signal)

    if n < 20:
        return 0

    diffs = np.abs(np.diff(signal))
    median_val = np.median(diffs)
    reference = median_val if median_val > 1e-9 else np.mean(diffs)

    if reference < 1e-9:
        reference = 1e-9

    threshold = reference * 10
    limit = int(n * max_skip_ratio)
    start_idx = min_skip

    for i in range(min_skip, limit):
        window = diffs[i:i + 3]
        if len(window) > 0 and np.all(window < threshold):
            start_idx = i
            break

    return start_idx


def get_safe_indices(deriv1, deriv2, length, signal_for_adaptive, pid):
    """
    Determine sampling indices within the valid derivative range.

    The original sampling logic is preserved:
    1. restrict sampling to valid derivative regions;
    2. combine the adaptive starting point, fixed skipping rule,
       and first valid derivative position.
    """
    valid_idx1 = np.where(
        (~np.isnan(deriv1)) & (np.abs(deriv1) > 1e-9)
    )[0]

    valid_idx2 = np.where(
        (~np.isnan(deriv2)) & (np.abs(deriv2) > 1e-9)
    )[0]

    first_valid_1 = valid_idx1[0] if len(valid_idx1) > 0 else 0
    first_valid_2 = valid_idx2[0] if len(valid_idx2) > 0 else 0
    min_positive_start = max(first_valid_1, first_valid_2)

    last_1 = valid_idx1[-1] if len(valid_idx1) > 0 else 0
    last_2 = valid_idx2[-1] if len(valid_idx2) > 0 else 0
    safe_end = min(last_1, last_2)

    if safe_end > 5:
        safe_end -= 1

    if safe_end < NUM_SAMPLES:
        safe_end = length - 1

    adaptive_start = get_adaptive_start_index(signal_for_adaptive)
    skip_fixed = 30 if length > 90 else int(length * 0.1)

    safe_start = max(
        adaptive_start,
        skip_fixed,
        min_positive_start,
    )

    if safe_start >= safe_end:
        safe_start = 0

    return np.linspace(
        safe_start,
        safe_end,
        NUM_SAMPLES,
    ).astype(int)


def calculate_derivatives(df, pid):
    """Calculate stage-dependent derivative features."""
    n = len(df)

    v = df["Voltage"].values
    q = df["Capacity"].values
    i_arr = df["Current"].values

    res1 = np.full(n, np.nan)
    res2 = np.full(n, np.nan)

    if pid == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            res1 = np.gradient(q) / np.gradient(v)
            res2 = np.gradient(v) / np.gradient(q)

    elif pid in [3, 4]:
        for i in range(n):
            for j in range(i + 1, n):
                if v[j] - v[i] >= CC_DELTA_V:
                    val = (q[j] - q[i]) / (v[j] - v[i])
                    if val > 0:
                        res1[i] = val
                    break

        for i in range(n):
            for j in range(i + 1, n):
                if q[j] - q[i] >= CC_DELTA_Q:
                    val = (v[j] - v[i]) / (q[j] - q[i])
                    if val > 0:
                        res2[i] = val
                    break

    else:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i_arr[j] - i_arr[i]) >= CV_DELTA_I:
                    denom = i_arr[j] - i_arr[i]
                    if denom != 0:
                        res1[i] = (q[j] - q[i]) / denom
                        break

        for i in range(n):
            for j in range(i + 1, n):
                if q[j] - q[i] >= CV_DELTA_Q:
                    denom = q[j] - q[i]
                    if denom != 0:
                        res2[i] = (i_arr[j] - i_arr[i]) / denom
                        break

    res1[~np.isfinite(res1)] = np.nan
    res2[~np.isfinite(res2)] = np.nan

    return res1, res2


def process_cell_folder(cell_folder):
    """Extract the full semantic vector for one cell."""
    fname = os.path.basename(cell_folder)

    match = re.search(r"Cell_(\d+)_", fname)
    cid = int(match.group(1)) if match else 0

    files = glob.glob(
        os.path.join(cell_folder, "*.csv")
    )

    seg_map = {
        int(re.findall(r"\d+", os.path.basename(f))[-1]): f
        for f in files
        if re.findall(r"\d+", os.path.basename(f))
    }

    full_vec = []
    names = []

    for pid in range(1, 6):
        d1n, d2n = (
            ("IC", "DV")
            if pid in [1, 3, 4]
            else ("dQdI", "dIdQ")
        )

        if pid not in seg_map:
            full_vec.extend([0] * 48)

            names.extend([
                f"S{pid}_Start_V",
                f"S{pid}_Start_Q",
                f"S{pid}_Start_I",
                f"S{pid}_Start_T",
            ])

            names.extend([
                f"S{pid}_{d1n}_{k + 1}"
                for k in range(10)
            ])

            names.extend([
                f"S{pid}_{d2n}_{k + 1}"
                for k in range(10)
            ])

            names.extend([
                f"S{pid}_SampQ_{k + 1}"
                for k in range(10)
            ])

            names.extend([
                f"S{pid}_SampV_{k + 1}"
                for k in range(10)
            ])

            names.extend([
                f"S{pid}_End_T",
                f"S{pid}_End_I",
                f"S{pid}_End_Q",
                f"S{pid}_End_V",
            ])

            if pid < 5:
                full_vec.append(0)
                names.append(f"Sep_{pid}")

            continue

        df = pd.read_csv(
            seg_map[pid]
        )

        col_map = {}

        for c in df.columns:
            cl = c.lower()

            if "volt" in cl:
                col_map[c] = "Voltage"
            elif "curr" in cl:
                col_map[c] = "Current"
            elif "cap" in cl:
                col_map[c] = "Capacity"
            elif "temp" in cl and "circuit" not in cl:
                col_map[c] = "Temperature"

        df.rename(
            columns=col_map,
            inplace=True,
        )

        if "Temperature" not in df.columns:
            df["Temperature"] = 25.0

        d1, d2 = calculate_derivatives(
            df,
            pid,
        )

        signal_for_adaptive = (
            df["Voltage"].values
            if pid in [3, 4]
            else df["Current"].values
        )

        idx = get_safe_indices(
            d1,
            d2,
            len(df),
            signal_for_adaptive,
            pid,
        )

        s_d1 = smooth_curve(d1)
        s_d2 = smooth_curve(d2)

        if pid in [3, 4]:
            s_d1[s_d1 < 0] = 0
            s_d2[s_d2 < 0] = 0

        v0 = df["Voltage"].iloc[0]
        q0 = df["Capacity"].iloc[0]
        i0 = df["Current"].iloc[0]
        t0 = df["Temperature"].iloc[0]

        full_vec.extend([
            v0,
            q0,
            i0,
            t0,
        ])

        names.extend([
            f"S{pid}_Start_V",
            f"S{pid}_Start_Q",
            f"S{pid}_Start_I",
            f"S{pid}_Start_T",
        ])

        full_vec.extend(s_d1[idx])
        names.extend([
            f"S{pid}_{d1n}_{k + 1}"
            for k in range(10)
        ])

        full_vec.extend(s_d2[idx])
        names.extend([
            f"S{pid}_{d2n}_{k + 1}"
            for k in range(10)
        ])

        full_vec.extend(
            df["Capacity"].values[idx]
        )
        names.extend([
            f"S{pid}_SampQ_{k + 1}"
            for k in range(10)
        ])

        full_vec.extend(
            df["Voltage"].values[idx]
        )
        names.extend([
            f"S{pid}_SampV_{k + 1}"
            for k in range(10)
        ])

        t_end = df["Temperature"].iloc[-1]
        i_end = df["Current"].iloc[-1]
        q_end = df["Capacity"].iloc[-1]
        v_end = df["Voltage"].iloc[-1]

        full_vec.extend([
            t_end,
            i_end,
            q_end,
            v_end,
        ])

        names.extend([
            f"S{pid}_End_T",
            f"S{pid}_End_I",
            f"S{pid}_End_Q",
            f"S{pid}_End_V",
        ])

        if pid < 5:
            full_vec.append(1)
            names.append(f"Sep_{pid}")

    return cid, full_vec, names


def main(input_root, output_dir):
    """Run semantic encoding for all available cell folders."""
    if not os.path.exists(input_root):
        raise FileNotFoundError(
            f"Input directory not found: {input_root}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    cell_dirs = glob.glob(
        os.path.join(input_root, "Cell_*")
    )

    print(
        f"Starting semantic feature extraction "
        f"for {len(cell_dirs)} cells..."
    )

    all_data = []

    for i, cell_dir in enumerate(cell_dirs):
        cid, values, feature_names = process_cell_folder(
            cell_dir
        )

        if values:
            pd.DataFrame({
                "Feature_Name": feature_names,
                "Value": values,
            }).to_csv(
                os.path.join(
                    output_dir,
                    f"Cell_{cid}_Final.csv",
                ),
                index=False,
            )

            row = {"Cell_ID": cid}
            row.update(
                dict(
                    zip(
                        feature_names,
                        values,
                    )
                )
            )

            all_data.append(row)

        if i % 20 == 0:
            print(
                f"Progress: {i}/{len(cell_dirs)}"
            )

    if all_data:
        df_master = (
            pd.DataFrame(all_data)
            .sort_values("Cell_ID")
        )

        save_path = os.path.join(
            output_dir,
            "Master_Features_244D.csv",
        )

        df_master.to_csv(
            save_path,
            index=False,
        )

        print(
            f"Feature extraction completed: {save_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract a 244-dimensional semantic feature vector "
            "from five pre-segmented formation stages."
        )
    )

    parser.add_argument(
        "--input-root",
        default="./data/formation_segments",
        help=(
            "Directory containing Cell_* folders with "
            "pre-segmented formation-stage CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="./outputs/01_semantic_encoding",
        help="Directory for extracted semantic features.",
    )

    args = parser.parse_args()

    main(
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
