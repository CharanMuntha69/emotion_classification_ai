import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import librosa

from train_pytorch import SEED

# =========================
# Config
# =========================
RAVDESS_DIR = "./RAVDESS"   # change to your ravdess folder path
PRETRAINED_PATH = "./processed/cnn_valence_arousal.pt"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
SEED = 42
VAL_SPLIT = 0.2
SR = 22050
TARGET_SEC = 4
N_MELS = 128
N_FFT = 2048
HOP = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# Dataset
# =========================
class RavdessDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []

        for actor in os.listdir(root_dir):
            actor_path = os.path.join(root_dir, actor)
            if not os.path.isdir(actor_path):
                continue

            for f in os.listdir(actor_path):
                if f.endswith(".wav"):
                    full_path = os.path.join(actor_path, f)

                    parts = f.split("-")
                    emotion_code = int(parts[2])  # third field
                    label = emotion_code - 1      # make 0–7

                    self.files.append(full_path)
                    self.labels.append(label)

        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        y, _ = librosa.load(path, sr=SR, mono=True)
        target_len = TARGET_SEC * SR

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP
        )

        logmel = librosa.power_to_db(mel, ref=np.max)

        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

        x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

# =========================
# Model
# =========================
class CNNTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# Training
# =========================
def train_model(pretrained=False):

    dataset = RavdessDataset(RAVDESS_DIR)
    n_total = len(dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CNNTransfer().to(device)

    if pretrained:
        checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        pretrained_sd = checkpoint["model_state_dict"]

        # Keep only the feature extractor weights
        feature_sd = {}
        for k, v in pretrained_sd.items():
            if k.startswith("features."):
                feature_sd[k.replace("features.", "")] = v  # strip prefix

        model.features.load_state_dict(feature_sd, strict=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        correct = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

        train_acc = correct / total

        # ---- VALIDATION ----
        model.eval()
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                pred = model(xb)
                _, predicted = torch.max(pred, 1)

                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.3f} | Val Acc = {val_acc:.3f}")

    return model

if __name__ == "__main__":
    print("Training from scratch:")
    train_model(pretrained=False)

    print("\nTraining with music pretraining:")
    train_model(pretrained=True)