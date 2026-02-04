import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.cuda.amp import autocast, GradScaler

from video_dataset import VideoDataset
from transforms import get_train_transforms, get_val_transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES_PER_VIDEO = 16
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
PATIENCE = 2   # early stopping patience

train_ds = VideoDataset(
    root_dir=f"{DATA_DIR}/train",
    transform=get_train_transforms(),
    frames_per_video=FRAMES_PER_VIDEO
)

val_ds = VideoDataset(
    root_dir=f"{DATA_DIR}/val",
    transform=get_val_transforms(),
    frames_per_video=FRAMES_PER_VIDEO
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
cnn.classifier = nn.Identity()
cnn = cnn.to(DEVICE)

for param in cnn.parameters():
    param.requires_grad = False

FEATURE_DIM = 1280

lstm = nn.LSTM(
    input_size=FEATURE_DIM,
    hidden_size=256,
    num_layers=1,
    batch_first=True
).to(DEVICE)

classifier = nn.Linear(256, 1).to(DEVICE)



criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    list(lstm.parameters()) + list(classifier.parameters()),
    lr=LR
)

scaler = GradScaler()

os.makedirs(MODEL_DIR, exist_ok=True)
best_val_loss = float("inf")
patience_counter = 0

def train_one_epoch():
    cnn.eval()
    lstm.train()
    classifier.train()

    total_loss = 0.0

    for videos, labels in train_loader:
        labels = labels.float().unsqueeze(1).to(DEVICE)

        videos = torch.stack(videos, dim=1).to(DEVICE)
        B, T, C, H, W = videos.shape
        videos = videos.view(B * T, C, H, W)

        with torch.no_grad():
            features = cnn(videos)

        features = features.view(B, T, -1)

        optimizer.zero_grad()

        with autocast():
            outputs, _ = lstm(features)
            final_output = outputs[:, -1, :]
            logits = classifier(final_output)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate():
    cnn.eval()
    lstm.eval()
    classifier.eval()

    total_loss = 0.0

    for videos, labels in val_loader:
        labels = labels.float().unsqueeze(1).to(DEVICE)

        videos = torch.stack(videos, dim=1).to(DEVICE)
        B, T, C, H, W = videos.shape
        videos = videos.view(B * T, C, H, W)

        features = cnn(videos)
        features = features.view(B, T, -1)

        outputs, _ = lstm(features)
        final_output = outputs[:, -1, :]
        logits = classifier(final_output)
        loss = criterion(logits, labels)

        total_loss += loss.item()

    return total_loss / len(val_loader)



print("Using device:", DEVICE)

for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss = train_one_epoch()
    val_loss = validate()

    epoch_time = time.time() - start_time

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "lstm_state": lstm.state_dict(),
                "classifier_state": classifier.state_dict(),
                "val_loss": val_loss
            },
            os.path.join(MODEL_DIR, "cnn_lstm_best.pth")
        )

        print("Saved new best model.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete.")
