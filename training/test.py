import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DeepSafeDataset
from model import DeepSafeModel
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "../models/deepsafe_best.pth"
TEST_DIR = "../data/splits/test"


def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = DeepSafeDataset(split_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = DeepSafeModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    y_true = []
    y_pred = []

    print("Running Final Evaluation on Test Set")
    with torch.no_grad():
        for frames, labels in tqdm(test_loader):
            frames = frames.to(DEVICE)
            outputs = model(frames)

            preds = torch.sigmoid(outputs).cpu().numpy()
            y_pred.extend((preds > 0.5).astype(int).flatten())
            y_true.extend(labels.numpy().flatten())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Real', 'Pred Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title("DeepSafe-Satyanveshi: Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("../deepsafe_confusion_matrix.png")
    print("\nConfusion Matrix saved as 'deepsafe_confusion_matrix.png'")
    plt.show()


if __name__ == "__main__":
    evaluate()