import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

NORMALIZED_DIR = "./outputs/03_normalization"
MODEL_DIR = "./outputs/04_model"
OUTPUT_DIR = "./outputs/05_test"

DEVICE = torch.device("cpu")

TEST_SEED = 40
DROPOUT = 0.3
MC_SAMPLES = 300

Y_COLS = ["Knee_Cycle", "EOL_Cycle"]

def set_seed(seed=TEST_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BatteryDataset(Dataset):
    def __init__(self, x_df, y_df):
        if not np.array_equal(
            x_df["Cell_ID"].astype(int).to_numpy(),
            y_df["Cell_ID"].astype(int).to_numpy(),
        ):
            raise ValueError("Test X/Y Cell_ID order is inconsistent.")

        self.cell_ids = x_df["Cell_ID"].astype(int).to_numpy()

        self.x = torch.tensor(
            x_df.drop(columns="Cell_ID").to_numpy(dtype=np.float32)
        )

        self.y = torch.tensor(
            y_df[Y_COLS].to_numpy(dtype=np.float32)
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 2, 1, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return x * self.attention(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_s = nn.Conv1d(1, 16, 2, padding=1)
        self.conv_m = nn.Conv1d(1, 16, 5, padding=2)
        self.conv_l = nn.Conv1d(1, 16, 7, padding=3)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(32)
        self.attention = TemporalAttention(48)

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(48 * 32, 128)

        self.out_knee = nn.Linear(128, 1)
        self.out_eol = nn.Linear(128, 1)

    def forward(self, x):
        s = self.pool(self.relu(self.conv_s(x)))
        m = self.pool(self.relu(self.conv_m(x)))
        l = self.pool(self.relu(self.conv_l(x)))

        x = torch.cat((s, m, l), dim=1)
        x = self.attention(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc(self.dropout(x)))

        return self.out_knee(x), self.out_eol(x)

def enable_dropout(model):
    """Enable Dropout layers while keeping all other layers in eval mode."""
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()

def inverse_norm(values, mn, mx):
    return np.asarray(values, dtype=float) * (mx - mn) + mn

def calc_metrics(true, pred):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mape = np.mean(
        np.abs((true - pred) / (true + 1e-8))
    ) * 100

    return float(mae), float(rmse), float(mape)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_x = pd.read_pickle(
        os.path.join(NORMALIZED_DIR, "test_x.pkl")
    )
    test_y = pd.read_pickle(
        os.path.join(NORMALIZED_DIR, "test_y.pkl")
    )

    if len(test_x) != 28 or len(test_y) != 28:
        raise ValueError(
            f"Expected 28 test cells, got "
            f"X={len(test_x)}, Y={len(test_y)}."
        )

    if test_x.shape[1] - 1 != 244:
        raise ValueError(
            f"Expected 244 input features, "
            f"found {test_x.shape[1] - 1}."
        )

    with open(
        os.path.join(
            NORMALIZED_DIR,
            "normalization_params.json",
        ),
        "r",
        encoding="utf-8",
    ) as f:
        norm_params = json.load(f)

    y_params = norm_params["Y"]

    knee_min = float(
        y_params["Knee_Cycle"]["min"]
    )
    knee_max = float(
        y_params["Knee_Cycle"]["max"]
    )
    eol_min = float(
        y_params["EOL_Cycle"]["min"]
    )
    eol_max = float(
        y_params["EOL_Cycle"]["max"]
    )

    test_ds = BatteryDataset(
        test_x,
        test_y,
    )

    model = Model().to(DEVICE)

    model_path = os.path.join(
        MODEL_DIR,
        "trained_model_state_dict.pth",
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
        )
    )

    model.eval()
    enable_dropout(model)

    set_seed()

    cell_ids = []
    knee_true = []
    knee_mean = []
    knee_std = []

    eol_true = []
    eol_mean = []
    eol_std = []

    with torch.no_grad():
        for idx in range(len(test_ds)):
            bx, by = test_ds[idx]

            cell_id = int(
                test_ds.cell_ids[idx]
            )

            bx = bx.unsqueeze(0).to(DEVICE)

            k_samples_norm = []
            e_samples_norm = []

            for _ in range(MC_SAMPLES):
                pred_knee, pred_eol = model(bx)

                k_samples_norm.append(
                    pred_knee.item()
                )
                e_samples_norm.append(
                    pred_eol.item()
                )

            k_samples = inverse_norm(
                k_samples_norm,
                knee_min,
                knee_max,
            )
            e_samples = inverse_norm(
                e_samples_norm,
                eol_min,
                eol_max,
            )

            true_knee = float(
                inverse_norm(
                    by[0].item(),
                    knee_min,
                    knee_max,
                )
            )
            true_eol = float(
                inverse_norm(
                    by[1].item(),
                    eol_min,
                    eol_max,
                )
            )

            cell_ids.append(cell_id)

            knee_true.append(true_knee)
            knee_mean.append(
                float(np.mean(k_samples))
            )
            knee_std.append(
                float(np.std(k_samples))
            )

            eol_true.append(true_eol)
            eol_mean.append(
                float(np.mean(e_samples))
            )
            eol_std.append(
                float(np.std(e_samples))
            )

    knee_std = np.asarray(knee_std)
    eol_std = np.asarray(eol_std)

    knee_ci = 1.96 * knee_std
    eol_ci = 1.96 * eol_std

    k_mae, k_rmse, k_mape = calc_metrics(
        knee_true,
        knee_mean,
    )
    e_mae, e_rmse, e_mape = calc_metrics(
        eol_true,
        eol_mean,
    )

    predictions = pd.DataFrame({
        "Cell_ID": cell_ids,
        "Knee_Actual": knee_true,
        "Knee_MC_Mean": knee_mean,
        "Knee_MC_Std": knee_std,
        "Knee_95CI_HalfWidth": knee_ci,
        "EOL_Actual": eol_true,
        "EOL_MC_Mean": eol_mean,
        "EOL_MC_Std": eol_std,
        "EOL_95CI_HalfWidth": eol_ci,
    })

    predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions.csv",
        ),
        index=False,
    )

    report = {
        "test_cells": len(test_ds),
        "mc_samples": MC_SAMPLES,
        "inverse_normalization": (
            "training-set Knee/EOL min-max from Part 3"
        ),
        "three_sigma_filtering": False,
        "Knee": {
            "MAE": k_mae,
            "RMSE": k_rmse,
            "MAPE_percent": k_mape,
            "mean_95CI_halfwidth": float(
                np.mean(knee_ci)
            ),
        },
        "EOL": {
            "MAE": e_mae,
            "RMSE": e_rmse,
            "MAPE_percent": e_mape,
            "mean_95CI_halfwidth": float(
                np.mean(eol_ci)
            ),
        },
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "evaluation_metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("Part 5 completed: model testing")
    print(
        f"Knee | MAE={k_mae:.2f} | "
        f"RMSE={k_rmse:.2f} | "
        f"MAPE={k_mape:.2f}% | "
        f"95%CI=±{np.mean(knee_ci):.2f}"
    )
    print(
        f"EOL  | MAE={e_mae:.2f} | "
        f"RMSE={e_rmse:.2f} | "
        f"MAPE={e_mape:.2f}% | "
        f"95%CI=±{np.mean(eol_ci):.2f}"
    )
    print("No 3-sigma filtering.")
    print(
        "Predictions were inverse-normalized using "
        "training-label Min/Max only."
    )
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()
