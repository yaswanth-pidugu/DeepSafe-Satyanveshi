import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "gradcam")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_INTERVAL = 10
IMG_SIZE = 224



os.makedirs(OUTPUT_DIR, exist_ok=True)


model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

model = model.to(DEVICE)
model.eval()

target_layer = model.features[-1]

activations = None
gradients = None


def save_activation(module, input, output):
    global activations
    activations = output.detach()


def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()


target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)



transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_frames(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            frames.append(frame)

        idx += 1

    cap.release()
    return frames


def generate_gradcam(frame, idx):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    model.zero_grad()
    output = model(input_tensor)
    score = torch.sigmoid(output)

    score.backward()


    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(DEVICE)

    for i, w in enumerate(pooled_grads):
        cam += w * activations[0, i]

    cam = cam.cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(OUTPUT_DIR, f"gradcam_{idx}.jpg")
    cv2.imwrite(out_path, overlay)

    print(f"Saved Grad-CAM: {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gradcam_video.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    frames = extract_frames(video_path, FRAME_INTERVAL)
    print(f"Visualizing {len(frames)} frames")

    for i, frame in enumerate(frames[:10]):  # limit to first 10 frames
        generate_gradcam(frame, i)
