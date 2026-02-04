import os
import random
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACES_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "faces")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)


def make_dirs():
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(SPLITS_DIR, split, label), exist_ok=True)


def split_videos(video_list):
    random.shuffle(video_list)
    total = len(video_list)

    train_end = int(SPLIT_RATIOS["train"] * total)
    val_end = train_end + int(SPLIT_RATIOS["val"] * total)

    return {
        "train": video_list[:train_end],
        "val": video_list[train_end:val_end],
        "test": video_list[val_end:]
    }


def copy_video(video_name, label, split):
    src_dir = os.path.join(FACES_DIR, label, video_name)
    dst_dir = os.path.join(SPLITS_DIR, split, label, video_name)

    shutil.copytree(src_dir, dst_dir)


def main():
    make_dirs()

    for label in ["real", "fake"]:
        label_dir = os.path.join(FACES_DIR, label)
        videos = os.listdir(label_dir)

        splits = split_videos(videos)

        for split, video_names in splits.items():
            for video_name in video_names:
                copy_video(video_name, label, split)

        print(f"{label.upper()} -> "
              f"Train: {len(splits['train'])}, "
              f"Val: {len(splits['val'])}, "
              f"Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
