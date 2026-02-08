import cv2
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
from config_manager import cfg

INPUT_DIR = Path(cfg['paths']['raw_data'])
OUTPUT_DIR = Path(cfg['paths']['processed_faces'])
FRAMES_PER_VIDEO = cfg['model']['frames_per_video']
IMG_SIZE = cfg['model']['image_size']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(
    margin=20,
    keep_all=False,
    post_process=False,
    device=device,
    image_size=IMG_SIZE
)

def extract_faces():
    for label in ['real', 'fake']:
        videos = list((INPUT_DIR / label).glob("*.mp4"))
        label_out = OUTPUT_DIR / label
        label_out.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(videos)} {label} videos...")
        for v_path in tqdm(videos):
            vid_name = v_path.stem
            save_path = label_out / vid_name
            save_path.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(v_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Uniformly sample frames throughout the video
            indices = [int(i * (total / FRAMES_PER_VIDEO)) for i in range(FRAMES_PER_VIDEO)]

            for i, f_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: continue

                # MTCNN requires RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract and Save
                target_file = str(save_path / f"frame_{i}.jpg")
                detector(frame_rgb, save_path=target_file)

            cap.release()