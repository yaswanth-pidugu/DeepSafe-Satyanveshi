import torch
import torch.nn as nn
from torchvision import models


class DeepSafeModel(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepSafeModel, self).__init__()

        # 1. Feature Extractor (Backbone)
        # We use EfficientNet_B0 for high accuracy on low VRAM
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # 2. Freeze initial layers (Optional: helps training speed)
        # for param in self.feature_extractor[:6]:
        #     param.requires_grad = False

        # 3. Temporal Processor (GRU)
        # EfficientNet-B0 output is 1280 features
        self.gru = nn.GRU(input_size=1280, hidden_size=256, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.3)

        # 4. Final Classifier
        # Bidirectional GRU doubles the hidden size (256 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Output raw logit (use Sigmoid later)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, C, H, W) -> (B, 15, 3, 224, 224)
        batch_size, seq_len, c, h, w = x.shape

        # Flatten batch and sequence to pass through CNN
        x = x.view(batch_size * seq_len, c, h, w)

        # Extract features
        features = self.feature_extractor(x)  # (B*15, 1280, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (B, 15, 1280)

        # Pass sequence to GRU
        gru_out, _ = self.gru(features)

        # We only care about the output of the LAST frame in the sequence
        last_out = gru_out[:, -1, :]

        # Final prediction
        logits = self.classifier(last_out)
        return logits


if __name__ == "__main__":
    # Quick Test for VRAM check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSafeModel().to(device)
    dummy_input = torch.randn(2, 15, 3, 224, 224).to(device)  # Batch size 2
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")  # Should be [2, 1]
    print("GTX 1650 Test: Success.")