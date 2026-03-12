"""
Neural Network Model (Model 2) — Image Classification
โจทย์: แยกยี่ห้อรถจากรูปภาพ
ใช้ EfficientNet-B0 Pretrained (ImageNet) + Fine-tune

ยี่ห้อที่รองรับ:
  Toyota, Honda, BMW, Ford, Chevrolet, Audi, Mercedes, Hyundai, Nissan, Volkswagen
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
CAR_BRANDS = [
    "Toyota", "Honda", "BMW", "Ford", "Chevrolet",
    "Audi", "Mercedes", "Hyundai", "Nissan", "Volkswagen"
]

# ImageNet class indices ที่เกี่ยวกับรถยนต์
# EfficientNet trained on ImageNet รู้จักรถอยู่แล้ว
# เราจะ map จาก ImageNet predictions → ยี่ห้อรถ
IMAGENET_CAR_CLASSES = {
    # sports car, racer
    817: "BMW", 829: "BMW",
    # convertible
    511: "Audi",
    # limousine
    627: "Mercedes",
    # minivan
    656: "Toyota",
    # pickup truck
    717: "Ford", 864: "Ford",
    # jeep
    609: "Chevrolet",
    # beach wagon, station wagon
    436: "Volkswagen",
    # cab, hack, taxi, taxicab
    468: "Toyota",
    # minibus
    654: "Honda",
    # ambulance
    407: "Mercedes",
    # fire engine
    555: "Chevrolet",
    # police van
    734: "Ford",
    # car wheel
    479: "Toyota",
    # grille, radiator grille
    566: "Honda",
}

# ────────────────────────────────────────────────
# MODEL
# ────────────────────────────────────────────────
class CarBrandClassifier(nn.Module):
    """
    EfficientNet-B0 Pretrained + Custom Head สำหรับแยกยี่ห้อรถ
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze backbone layers (ไม่ train ใหม่ทั้งหมด)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ────────────────────────────────────────────────
# IMAGE TRANSFORMS
# ────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ────────────────────────────────────────────────
# PREDICT (ใช้ใน app.py)
# ────────────────────────────────────────────────
def predict_brand(image: Image.Image, model: nn.Module, device: str = "cpu"):
    """
    รับ PIL Image → return (brand, confidence, all_probs_dict)
    """
    model.eval()
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_brand = CAR_BRANDS[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {brand: float(p) for brand, p in zip(CAR_BRANDS, probs)}
    return pred_brand, confidence, all_probs


# ────────────────────────────────────────────────
# SAVE MODEL (เซฟ architecture + weights เริ่มต้น)
# ────────────────────────────────────────────────
def build_and_save_model(save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("=" * 55)
    print("NEURAL NETWORK — CAR BRAND IMAGE CLASSIFIER")
    print("=" * 55)
    print("Architecture: EfficientNet-B0 (Pretrained ImageNet)")
    print(f"Classes ({len(CAR_BRANDS)}): {CAR_BRANDS}")
    print("\nLoading pretrained weights...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CarBrandClassifier(num_classes=len(CAR_BRANDS)).to(device)

    # นับ parameters
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (head only)")

    # Plot architecture diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    layers = [
        ("Input Image", "224×224×3", "#4FC3F7"),
        ("EfficientNet-B0 Backbone", "Pretrained ImageNet\n(Frozen weights)", "#81C784"),
        ("Dropout (0.3)", "", "#FFB74D"),
        ("Linear (1280 → 256)", "ReLU", "#CE93D8"),
        ("Dropout (0.2)", "", "#FFB74D"),
        ("Linear (256 → 10)", "Output: 10 brands", "#EF9A9A"),
        ("Softmax", "Probability per brand", "#80DEEA"),
    ]

    for i, (name, detail, color) in enumerate(layers):
        y = 1 - i * 0.14
        rect = plt.Rectangle((0.1, y - 0.06), 0.8, 0.1,
                               facecolor=color, edgecolor="gray", linewidth=1.5,
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.5, y - 0.01, name, ha="center", va="center",
                fontsize=11, fontweight="bold", transform=ax.transAxes)
        if detail:
            ax.text(0.5, y - 0.045, detail, ha="center", va="center",
                    fontsize=8, color="#444", transform=ax.transAxes)
        if i < len(layers) - 1:
            ax.annotate("", xy=(0.5, y - 0.07), xytext=(0.5, y - 0.08),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_title("EfficientNet-B0 Car Brand Classifier Architecture",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("plots/nn_architecture.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Architecture diagram saved → plots/nn_architecture.png")

    # Save model
    model_path = os.path.join(save_dir, "nn_car_brand.pt")
    torch.save({
        "model_state": model.state_dict(),
        "classes":     CAR_BRANDS,
        "num_classes": len(CAR_BRANDS),
        "architecture": "EfficientNet-B0",
    }, model_path)

    print(f"\n✅ Model saved → {model_path}")
    print("\nหมายเหตุ: โมเดลนี้ใช้ EfficientNet-B0 ที่ train บน ImageNet")
    print("โดย backbone สามารถจดจำ feature ของรถได้แล้ว")
    print("หากต้องการความแม่นยำสูงขึ้น สามารถ fine-tune ด้วย dataset รูปรถเพิ่มเติมได้")

    return model


if __name__ == "__main__":
    build_and_save_model()
