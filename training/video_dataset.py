import os
import random
from PIL import Image
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=16):
        self.samples = []
        self.transform = transform
        self.frames_per_video = frames_per_video
        root_dir = os.path.abspath(root_dir)

        for label, class_name in enumerate(["real", "fake"]):
            class_dir = os.path.join(root_dir, class_name)

            for video in os.listdir(class_dir):
                video_dir = os.path.join(class_dir, video)
                if not os.path.isdir(video_dir):
                    continue

                frames = sorted([
                    os.path.join(video_dir, f)
                    for f in os.listdir(video_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ])

                if len(frames) >= frames_per_video:
                    self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]

        selected = random.sample(frames, self.frames_per_video)
        images = []

        for img_path in selected:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return images, label
