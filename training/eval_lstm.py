import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score

from video_dataset import VideoDataset
from transforms import get_val_transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_lstm_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES_PER_VIDEO = 16
BATCH_SIZE = 4

val_ds = VideoDataset(
    root_dir=f"{DATA_DIR}/val",
    transform=get_val_transforms(),
    frames_per_video=FRAMES_PER_VIDEO
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
cnn.classifier = nn.Identity()
cnn = cnn.to(DEVICE)
cnn.eval()

for p in cnn.parameters():
    p.requires_grad = False

FEATURE_DIM = 1280

lstm = nn.LSTM(
    input_size=FEATURE_DIM,
    hidden_size=256,
    num_layers=1,
    batch_first=True
).to(DEVICE)

classifier = nn.Linear(256, 1).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
lstm.load_state_dict(checkpoint["lstm_state"])
classifier.load_state_dict(checkpoint["classifier_state"])

lstm.eval()
classifier.eval()

y_true = []
y_scores = []

with torch.no_grad():
    for videos, labels in val_loader:
        labels = labels.to(DEVICE)

        videos = torch.stack(videos, dim=1).to(DEVICE)
        B, T, C, H, W = videos.shape
        videos = videos.view(B * T, C, H, W)

        features = cnn(videos)
        features = features.view(B, T, -1)

        outputs, _ = lstm(features)
        final_output = outputs[:, -1, :]

        logits = classifier(final_output)
        probs = torch.sigmoid(logits).squeeze(1)

        y_scores.extend(probs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())


y_true = np.array(y_true)
y_scores = np.array(y_scores)

y_pred = (y_scores > 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_scores)

print(f"Video-level Accuracy: {acc:.4f}")
print(f"Video-level AUC: {auc:.4f}")
print(f"Total videos evaluated: {len(y_true)}")
