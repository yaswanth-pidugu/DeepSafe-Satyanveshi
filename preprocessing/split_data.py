import shutil
import random
from pathlib import Path
from config_manager import cfg

SOURCE_BASE = Path(cfg['paths']['processed_faces'])
SPLIT_BASE = Path(cfg['paths']['splits'])
TRAIN_RATIO = cfg['data_split']['train']
VAL_RATIO = cfg['data_split']['val']


def split_dataset():
    for label in ['real', 'fake']:
        source_dir = SOURCE_BASE / label
        video_folders = [f.name for f in source_dir.iterdir() if f.is_dir()]

        random.seed(42)
        random.shuffle(video_folders)

        train_count = int(len(video_folders) * TRAIN_RATIO)
        val_count = int(len(video_folders) * VAL_RATIO)

        train_vids = video_folders[:train_count]
        val_vids = video_folders[train_count:train_count + val_count]
        test_vids = video_folders[train_count + val_count:]

        splits = {'train': train_vids, 'val': val_vids, 'test': test_vids}

        for split_name, vids in splits.items():
            print(f"Moving {len(vids)} {label} videos to {split_name}...")
            dest_dir = SPLIT_BASE / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for vid in vids:
                shutil.copytree(source_dir / vid, dest_dir / vid)


if __name__ == "__main__":
    split_dataset()