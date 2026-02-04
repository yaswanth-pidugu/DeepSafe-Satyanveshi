import os
import random

ORIGINAL_CELEB_DIR = r"C:\Users\yashw\Downloads\archive (5)"
ORIGINAL_FF_DIR = r"C:\Users\yashw\Downloads\archive (4)\FaceForensics++_C23"
TRAIN_RAW_DIR = r'C:\Users\yashw\PycharmProjects\DeepSafe-Satyanveshi\data\raw'


def create_symlinks():
    test_list = []
    with open(os.path.join(ORIGINAL_CELEB_DIR, r"C:\Users\yashw\Downloads\archive (5)\List_of_testing_videos.txt"), 'r') as f:
        test_list = [line.strip().split(' ')[1] for line in f]

    def link_files(src_root, folder_name, target_label, limit=None):
        files = [f for f in os.listdir(os.path.join(src_root, folder_name)) if f.endswith('.mp4')]

        files = [f for f in files if f"{folder_name}/{f}" not in test_list]

        if limit:
            files = random.sample(files, limit)

        for f in files:
            src = os.path.join(src_root, folder_name, f)
            dst = os.path.join(TRAIN_RAW_DIR, target_label, f"{folder_name}_{f}")

            # This creates the "shortcut"
            if not os.path.exists(dst):
                os.symlink(src, dst)


    link_files(ORIGINAL_FF_DIR, "original", "real")
    link_files(ORIGINAL_CELEB_DIR, "YouTube-real", "real")
    link_files(ORIGINAL_CELEB_DIR, "Celeb-real", "real")

    link_files(ORIGINAL_FF_DIR, "Deepfakes", "fake", limit=1000)
    link_files(ORIGINAL_CELEB_DIR, "Celeb-synthesis", "fake", limit=890)


create_symlinks()