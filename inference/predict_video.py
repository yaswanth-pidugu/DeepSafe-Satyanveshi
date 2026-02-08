import sys
import warnings
from pathlib import Path
from config_manager import cfg

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import torch
import cv2
import numpy as np
from training.model import DeepSafeModel
from facenet_pytorch import MTCNN

warnings.filterwarnings("ignore", category=FutureWarning)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = weights_path = cfg['paths']['weights']

model = DeepSafeModel().to(DEVICE)
model.load_state_dict(torch.load(str(weights_path), weights_only=True))
model.eval()

detector = MTCNN(margin=20, keep_all=False, device=DEVICE, image_size=224)


def get_frames(cap, indices):
    frames = []
    for f_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = detector(frame_rgb)
        if face is not None:
            frames.append((face + 1) / 2)
    return frames


def predict_best(video_path):
    vid_path = Path(video_path)
    if not vid_path.is_absolute():
        vid_path = root_dir / video_path

    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Analyze 30 frames (Two windows of 15) for better stability
    indices_1 = [int(i * (total_frames // 2 / 15)) for i in range(15)]
    indices_2 = [int(total_frames // 2 + i * (total_frames // 2 / 15)) for i in range(15)]

    probs = []
    for idx_set in [indices_1, indices_2]:
        frames = get_frames(cap, idx_set)
        if len(frames) == 15:
            video_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(video_tensor)
                probs.append(torch.sigmoid(output).item())

    cap.release()

    if not probs:
        return "ERROR", 0.0, "No faces detected."

    final_prob = np.mean(probs)


    if final_prob > cfg['inference']['threshold_fake']:
        verdict = "FAKE"
    elif final_prob < cfg['inference']['threshold_real']:
        verdict = "REAL"
    else:
        verdict = "INCONCLUSIVE (NEEDS REVIEW)"

    return verdict, final_prob, f"Analyzed {len(probs) * 15} frames."