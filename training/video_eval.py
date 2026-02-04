import os
import torch
import numpy as np
from collections import defaultdict
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from transforms import get_val_transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


val_ds = DeepfakeDataset(
    root_dir=f"{DATA_DIR}/val",
    transform=get_val_transforms()
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

model = model.to(DEVICE)
model.eval()


video_scores = defaultdict(list)
video_labels = {}


with torch.no_grad():
    for images, labels, videos in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

        for p, label, vid in zip(probs, labels.cpu().numpy(), videos):
            video_scores[vid].append(p)
            video_labels[vid] = label


y_true = []
y_pred = []

for vid in video_scores:
    y_true.append(video_labels[vid])
    y_pred.append(np.mean(video_scores[vid]))   # MEAN aggregation


from sklearn.metrics import accuracy_score, roc_auc_score

y_pred_bin = [1 if p > 0.5 else 0 for p in y_pred]

print("Video-level Accuracy:", accuracy_score(y_true, y_pred_bin))
print("Video-level AUC:", roc_auc_score(y_true, y_pred))
print("Total videos evaluated:", len(y_true))