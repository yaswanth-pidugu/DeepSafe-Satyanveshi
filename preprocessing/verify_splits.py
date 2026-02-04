import os

SPLITS_DIR = r'C:\Users\yashw\PycharmProjects\DeepSafe-Satyanveshi\data\splits'

def get_videos(split, label):
    return set(os.listdir(os.path.join(SPLITS_DIR, split, label)))

for label in ["real", "fake"]:
    train_videos = get_videos("train", label)
    val_videos = get_videos("val", label)
    test_videos = get_videos("test", label)

    print(f"\nLABEL: {label.upper()}")

    print("Train ∩ Val:", train_videos & val_videos)
    print("Train ∩ Test:", train_videos & test_videos)
    print("Val ∩ Test:", val_videos & test_videos)
