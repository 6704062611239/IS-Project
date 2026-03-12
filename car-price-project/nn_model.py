"""
Neural Network Model (Model 2) — Image Classification
โจทย์: แยกยี่ห้อรถจากรูปภาพ
Architecture: Custom CNN (ไม่ใช้ torchvision)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

CAR_BRANDS = [
    "Toyota", "Honda", "BMW", "Ford", "Chevrolet",
    "Audi", "Mercedes", "Hyundai", "Nissan", "Volkswagen"
]

# ────────────────────────────────────────────────
# IMAGE TRANSFORM (ไม่ใช้ torchvision)
# ────────────────────────────────────────────────
def transform_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


# ────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ────────────────────────────────────────────────
class CarBrandClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ────────────────────────────────────────────────
# BUILD AND SAVE
# ────────────────────────────────────────────────
def build_and_save_model(save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("=" * 55)
    print("NEURAL NETWORK — CAR BRAND IMAGE CLASSIFIER")
    print("=" * 55)
    print(f"Architecture: Custom CNN (3 Conv Blocks)")
    print(f"Classes ({len(CAR_BRANDS)}): {CAR_BRANDS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CarBrandClassifier(num_classes=len(CAR_BRANDS)).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Plot architecture
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    layers = [
        ("Input Image", "224×224×3",         "#4FC3F7"),
        ("Conv Block 1", "Conv→BN→ReLU ×2\nMaxPool | Dropout", "#81C784"),
        ("Conv Block 2", "Conv→BN→ReLU ×2\nMaxPool | Dropout", "#81C784"),
        ("Conv Block 3", "Conv→BN→ReLU\nMaxPool | Dropout",    "#81C784"),
        ("AdaptiveAvgPool", "Global Average Pooling",           "#FFB74D"),
        ("Linear (128→256)", "ReLU | Dropout(0.5)",             "#CE93D8"),
        ("Linear (256→10)", "Output: 10 ยี่ห้อรถ",              "#EF9A9A"),
        ("Softmax", "ความน่าจะเป็นแต่ละยี่ห้อ",                "#80DEEA"),
    ]
    for i, (name, detail, color) in enumerate(layers):
        y = 1 - i * 0.12
        rect = plt.Rectangle((0.1, y-0.07), 0.8, 0.09,
                               facecolor=color, edgecolor="gray", linewidth=1.5,
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.5, y-0.02, name, ha="center", va="center",
                fontsize=10, fontweight="bold", transform=ax.transAxes)
        if detail:
            ax.text(0.5, y-0.05, detail, ha="center", va="center",
                    fontsize=7.5, color="#444", transform=ax.transAxes)

    ax.set_title("Custom CNN Car Brand Classifier", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("plots/nn_architecture.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Architecture diagram saved → plots/nn_architecture.png")

    model_path = os.path.join(save_dir, "nn_car_brand.pt")
    torch.save({
        "model_state": model.state_dict(),
        "classes":     CAR_BRANDS,
        "num_classes": len(CAR_BRANDS),
        "architecture": "Custom CNN",
    }, model_path)

    print(f"\n✅ Model saved → {model_path}")
    return model


if __name__ == "__main__":
    build_and_save_model()
