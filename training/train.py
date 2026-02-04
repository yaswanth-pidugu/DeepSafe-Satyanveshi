import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import DeepfakeDataset
from transforms import get_train_transforms, get_val_transforms
torch.backends.cudnn.benchmark = True
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "efficientnet_b0.pth")
TOTAL_EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_ds = DeepfakeDataset(
    root_dir=f"{DATA_DIR}/train",
    transform=get_train_transforms()
)

val_ds = DeepfakeDataset(
    root_dir=f"{DATA_DIR}/val",
    transform=get_val_transforms()
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print("Dataset size:", len(train_ds))
print("Starting training loop...")


model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


os.makedirs(MODEL_DIR, exist_ok=True)
start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}...")


def train_one_epoch():
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % 20 == 0:
            print(
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / len(train_loader)



@torch.no_grad()
def validate():
    model.eval()
    running_loss = 0.0

    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    return running_loss / len(val_loader)


print(f"Using device: {DEVICE}")

for epoch in range(start_epoch, TOTAL_EPOCHS):
    train_loss = train_one_epoch()
    val_loss = validate()

    print(
        f"Epoch [{epoch + 1}/{TOTAL_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        },
        CHECKPOINT_PATH
    )

print("Training complete.")