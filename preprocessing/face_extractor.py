import os
from PIL import Image
from facenet_pytorch import MTCNN


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAMES_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "frames")
FACES_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "faces")

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    select_largest=True,
    post_process=True
)

def process_split(split_name):
    input_split_dir = os.path.join(FRAMES_DIR, split_name)
    output_split_dir = os.path.join(FACES_DIR, split_name)

    os.makedirs(output_split_dir, exist_ok=True)

    for video_folder in os.listdir(input_split_dir):
        video_input_dir = os.path.join(input_split_dir, video_folder)
        video_output_dir = os.path.join(output_split_dir, video_folder)

        os.makedirs(video_output_dir, exist_ok=True)

        for frame_file in os.listdir(video_input_dir):
            frame_path = os.path.join(video_input_dir, frame_file)

            try:
                img = Image.open(frame_path).convert("RGB")
                face = mtcnn(img)

                if face is not None:
                    face_img = Image.fromarray(
                        (face.permute(1, 2, 0).numpy() * 255).astype("uint8")
                    )
                    face_img.save(
                        os.path.join(video_output_dir, frame_file)
                    )

            except Exception as e:
                print(f"[WARN] Failed on {frame_path}: {e}")

def main():
    process_split("real")
    process_split("fake")
    print("Face extraction complete.")

if __name__ == "__main__":
    main()