import cv2
import os

#CONFIG
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "frames")
TARGET_FPS = 5
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / TARGET_FPS))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

def process_split(split_name):
    input_dir = os.path.join(RAW_DATA_DIR, split_name)
    output_dir = os.path.join(OUTPUT_DIR, split_name)

    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        if not video_file.lower().endswith(VIDEO_EXTENSIONS):
            continue

        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_dir, video_file)

        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"Processing {split_name}/{video_file}")
        extract_frames(video_path, video_output_dir)

def main():
    process_split("real")
    process_split("fake")

    print("Frame extraction complete.")

if __name__ == "__main__":
    main()
