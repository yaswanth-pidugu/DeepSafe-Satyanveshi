import os
from PIL import Image
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        root_dir = os.path.abspath(root_dir)
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = os.path.join(root_dir, class_name)
            for video in os.listdir(class_dir):
                video_dir = os.path.join(class_dir, video)
                for img_name in os.listdir(video_dir):
                    img_path = os.path.join(video_dir, img_name)
                    self.samples.append((img_path, label, video))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, video = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, video

