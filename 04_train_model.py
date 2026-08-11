import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

INPUT_DIR = "./outputs/03_normalization"
OUTPUT_DIR = "./outputs/04_model"

DEVICE = torch.device("cpu")

TRAINING_SEED = 40
BATCH_SIZE = 8
EPOCHS = 500
LR = 2e-3
DROPOUT = 0.3

Y_COLS = ["Knee_Cycle", "EOL_Cycle"]

def set_seed(seed=TRAINING_SEED):
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
            raise ValueError("Training X/Y Cell_ID order is inconsistent.")

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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_x = pd.read_pickle(
        os.path.join(INPUT_DIR, "train_x.pkl")
    )
    train_y = pd.read_pickle(
        os.path.join(INPUT_DIR, "train_y.pkl")
    )

    if len(train_x) != 117 or len(train_y) != 117:
        raise ValueError(
            f"Expected 117 training cells, got "
            f"X={len(train_x)}, Y={len(train_y)}."
        )

    feature_dim = train_x.shape[1] - 1
    if feature_dim != 244:
        raise ValueError(
            f"Expected 244 input features, found {feature_dim}."
        )

    set_seed()

    train_ds = BatteryDataset(train_x, train_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = Model().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0

        for bx, by in train_loader:
            bx = bx.to(DEVICE)
            by = by.to(DEVICE)

            pred_knee, pred_eol = model(bx)

            loss = (
                criterion(pred_knee, by[:, 0:1])
                + criterion(pred_eol, by[:, 1:2])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * bx.size(0)

        epoch_loss = total / len(train_ds)
        losses.append(epoch_loss)

        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
            or epoch + 1 == EPOCHS
        ):
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Loss={epoch_loss:.6f}"
            )

    # Do not print numerical state_dict values.
    # Do not include this .pth file in a public code-only release.
    model_path = os.path.join(
        OUTPUT_DIR,
        "trained_model_state_dict.pth",
    )
    torch.save(model.state_dict(), model_path)

    pd.DataFrame({
        "Epoch": range(1, EPOCHS + 1),
        "Training_Loss": losses,
    }).to_csv(
        os.path.join(OUTPUT_DIR, "training_loss.csv"),
        index=False,
    )

    model_config = {
        "model": "Multi-scale CNN S+M+L + Temporal Attention",
        "input_features": 244,
        "training_seed": TRAINING_SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "dropout": DROPOUT,
        "device": str(DEVICE),
        "public_release_note": (
            "Numerical learned weights are saved locally in the .pth file "
            "and should be omitted from a code-only public repository."
        ),
    }

    with open(
        os.path.join(OUTPUT_DIR, "model_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            model_config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 72)
    print("Part 4 completed: model training")
    print(f"Trainable parameter count: {n_params}")
    print(f"Model state_dict saved locally: {model_path}")
    print("Numerical learned weights are NOT printed.")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()
