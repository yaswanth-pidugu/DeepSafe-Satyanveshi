import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from config_manager import cfg

class DeepSafeDataset(Dataset):
    def __init__(self, split_dir, transform=None, seq_len=cfg['model']['seq_length']):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.seq_len = seq_len
        self.samples = []

        for label, class_idx in [('real', 0), ('fake', 1)]:
            class_path = self.split_dir / label
            if not class_path.exists(): continue
            for vid_folder in class_path.iterdir():
                if vid_folder.is_dir():
                    frames = sorted(list(vid_folder.glob("*.jpg")))
                    if len(frames) > 0:
                        self.samples.append((frames, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        frames = []
        for p in frame_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        current_len = len(frames)
        if current_len < self.seq_len:
            diff = self.seq_len - current_len
            frames.extend([frames[-1]] * diff)
        elif current_len > self.seq_len:
            frames = frames[:self.seq_len]

        video_tensor = torch.stack(frames)
        return video_tensor, label