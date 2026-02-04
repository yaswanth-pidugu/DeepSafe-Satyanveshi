import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_INTERVAL = 5       # take every 5th frame
IMG_SIZE = 224

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



def extract_frames(video_path, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()
    return frames


@torch.no_grad()
def predict_video(video_path):
    frames = extract_frames(video_path, FRAME_INTERVAL)

    if len(frames) == 0:
        raise ValueError("No frames extracted from video.")

    probs = []

    for frame in frames:
        img = Image.fromarray(frame)
        img = transform(img).unsqueeze(0).to(DEVICE)

        logits = model(img)
        prob = torch.sigmoid(logits).item()
        probs.append(prob)

    k = max(1, int(0.2 * len(probs)))  # top 20%
    top_k_probs = sorted(probs, reverse=True)[:k]
    mean_prob = float(np.mean(top_k_probs))

    return mean_prob, len(probs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_video.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    score, num_frames = predict_video(video_path)

    print(f"Frames analyzed: {num_frames}")
    print(f"Fake probability: {score:.4f}")

    if score > 0.3:
        print("Prediction: FAKE")
    else:
        print("Prediction: REAL")
