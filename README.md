# 🛡️ DeepSafe-Satyanveshi: Deepfake Video Detection
DeepSafe-Satyanveshi is an end-to-end deepfake detection system designed to identify manipulated videos with high precision. It utilizes a hybrid deep learning architecture, combining EfficientNet-B0 for spatial feature extraction and a Bidirectional GRU for temporal analysis.

# 🚀 Quick Start

## **_1. Prerequisites_**
Ensure you have Python 3.10+ installed. It is highly recommended to use a high-performance GPU (NVIDIA GTX 1650 or better) for training, though inference can run on CPU.

## **_2. Installation_**
Clone the repository and set up a virtual environment:

PowerShell
# Clone the repo
git clone https://github.com/yaswanth-pidugu/DeepSafe-Satyanveshi.git

cd DeepSafe-Satyanveshi

# Create and activate virtual environment
python -m venv .venv

.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## **_3. Configuration_**
This project uses a centralized configuration system.

Navigate to the configs/ folder.

Ensure base.yaml contains the model hyperparameters.

Open local.yaml and update the paths to match your local directory structure.

## **_4. Running the Pipeline_**
Follow these steps in order:

**Step A: Preprocessing**

- Extract faces from your raw video datasets.

- PowerShell
>python -m preprocessing.face_extractor

**Step B: Data Splitting**
- Split your processed face data into training, validation, and testing sets.

- PowerShell
> python -m preprocessing.split_data


**Step C: Training**
- Train the hybrid EfficientNet-GRU model. The best weights will be saved automatically based on validation loss.

- PowerShell
> python -m training.train
 
**Step D: Evaluation & Inference**
- Generate a classification report and confusion matrix, or test a specific video.

- PowerShell
-  #Run evaluation on the test set
> python -m training.test

- #Run inference on a specific video
> python -m inference.predict_video

🏗️ Project Structure
## 🏗️ Project Structure

```text
DeepSafe-Satyanveshi/
├── config/                # Centralized YAML configurations
│   ├── base.yaml          # Global hyperparameters
│   ├── local.yaml         # Local machine paths (Git ignored)
│   └── prod.yaml          # Production/Server settings
├── inference/             # Scripts for real-world prediction
│   └── predict_video.py   # Main inference logic
├── preprocessing/         # Data cleaning and face extraction scripts
│   ├── face_extractor.py  # MTCNN face cropping
│   └── split_data.py      # Training/Val/Test partitioning
├── training/              # Model architecture and training logic
│   ├── dataset.py         # Custom PyTorch Dataset class
│   ├── model.py           # EfficientNet + Bi-GRU stack
│   ├── train.py           # Training loop with checkpoints
│   └── test.py            # Evaluation and confusion matrix
├── .gitignore             # Rules to exclude data/local configs
├── config_manager.py      # Logic for loading YAML environments
└── requirements.txt       # Project dependencies
```

**🛠️ Tech Stack**

1. Deep Learning: PyTorch (EfficientNet_B0 + Bi-GRU)

2. Computer Vision: OpenCV, facenet-pytorch (MTCNN)

3. Configuration: YAML

4. Analytics: Scikit-learn, Matplotlib, Seaborn