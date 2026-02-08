import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DeepSafeDataset
from model import DeepSafeModel
from tqdm import tqdm
from config_manager import cfg

BATCH_SIZE = cfg['training']['batch_size']
EPOCHS = cfg['training']['epochs']
LR = float(cfg['training']['learning_rate'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train():
    train_ds = DeepSafeDataset(split_dir=Path(cfg['paths']['splits']) / "train", transform=transform)
    val_ds = DeepSafeDataset(split_dir=Path(cfg['paths']['splits']) / "val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = DeepSafeModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for frames, labels in loop:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)


            with torch.cuda.amp.autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), cfg['paths']['weights'])
            print("Best Model Saved")
