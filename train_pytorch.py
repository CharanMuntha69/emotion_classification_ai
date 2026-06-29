import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# Config

X_PATH = os.path.join("processed", "X_logmel.npy")
Y_PATH = os.path.join("processed", "y_valence_arousal.npy")

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3
SEED = 42
VAL_SPLIT = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Dataset

class DeamSpectrogramDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)  # (N, 128, T)
        self.y = np.load(y_path)  # (N, 2)

        if self.X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (N, 128, T). Got shape: {self.X.shape}")
        if self.y.ndim != 2 or self.y.shape[1] != 2:
            raise ValueError(f"Expected y to be (N, 2). Got shape: {self.y.shape}")

        # Normalize per-dataset (simple and effective baseline)
        # X is log-mel dB (often roughly -80..0). We'll standardize it.
        self.mean = self.X.mean()
        self.std = self.X.std() + 1e-8

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std       # (128, T)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 128, T) add channel
        y = torch.tensor(self.y[idx], dtype=torch.float32)     # (2,)
        return x, y


# Model: simple CNN regressor

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (128, T) -> (64, T/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, T/4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Make output size independent of T
            nn.AdaptiveAvgPool2d((1, 1))  # -> (64, 1, 1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),          # 64
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)       # valence, arousal
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# Metrics

def mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def pearson_corr(x, y):
    # x,y: numpy arrays shape (N,)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())) + 1e-12
    return float((x * y).sum() / denom)

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        total_loss += loss.item() * xb.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    avg_loss = total_loss / len(loader.dataset)
    avg_mae = np.mean(np.abs(all_pred - all_true))

    v_corr = pearson_corr(all_pred[:, 0], all_true[:, 0])
    a_corr = pearson_corr(all_pred[:, 1], all_true[:, 1])

    return avg_loss, float(avg_mae), v_corr, a_corr


# Train

def main():
    dataset = DeamSpectrogramDataset(X_PATH, Y_PATH)
    n_total = len(dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNNRegressor().to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    print("N total:", n_total, "N train:", n_train, "N val:", n_val)
    print("X example shape:", dataset[0][0].shape, "y example shape:", dataset[0][1].shape)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)
        val_loss, val_mae, v_corr, a_corr = evaluate(model, val_loader, loss_fn)

        print(
            f"Epoch {epoch:02d} | "
            f"train_mse={train_loss:.4f} | val_mse={val_loss:.4f} | "
            f"val_mae={val_mae:.4f} | corr_v={v_corr:.3f} | corr_a={a_corr:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "x_mean": dataset.mean,
                    "x_std": dataset.std,
                },
                os.path.join("processed", "cnn_valence_arousal.pt")
            )

    print("Saved best model to processed/cnn_valence_arousal.pt")

if __name__ == "__main__":
    main()
