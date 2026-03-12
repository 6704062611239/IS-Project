"""
Neural Network Model (Model 2)
โครงสร้าง: MLP Regressor ด้วย PyTorch
  Input → BatchNorm → Dense(256) → ReLU → Dropout(0.3)
        → Dense(128) → ReLU → Dropout(0.2)
        → Dense(64)  → ReLU
        → Dense(1)   (output: price)
Dataset: Car details v4.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preparation import prepare_v4


# ────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ────────────────────────────────────────────────
class CarPriceNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ────────────────────────────────────────────────
# TRAIN LOOP
# ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(X_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)


# ────────────────────────────────────────────────
# MAIN TRAINING
# ────────────────────────────────────────────────
def train_nn_model(base_dir="data", save_dir="models",
                   epochs=100, batch_size=64, lr=1e-3):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("=" * 50)
    print("NEURAL NETWORK MODEL TRAINING")
    print("=" * 50)

    X, y, encoders, scaler, _ = prepare_v4(base_dir, save_dir)

    # Log-transform target เพื่อลด skewness
    y_log = np.log1p(y.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32),
        y_log.astype(np.float32),
        test_size=0.2, random_state=42,
    )
    # เก็บ y_test ตัวจริงสำหรับ metric
    _, _, y_train_orig, y_test_orig = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    model     = CarPriceNN(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_model_state = None
    train_losses, val_losses = [], []

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = eval_epoch(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    _, y_pred_log, y_true_log = eval_epoch(model, test_loader, criterion, device)
    y_pred = np.expm1(y_pred_log)   # inverse log
    y_true = y_test_orig

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n📊 Test Results:")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  RMSE : {rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss", color="#2196F3")
    axes[0].plot(val_losses,   label="Val Loss",   color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # Actual vs Predicted
    axes[1].scatter(y_true, y_pred, alpha=0.4, color="#9C27B0")
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")
    axes[1].set_title("Actual vs Predicted (Neural Network)")

    plt.tight_layout()
    plt.savefig("plots/nn_results.png", dpi=120)
    plt.close()
    print("  Plot saved → plots/nn_results.png")

    # Save model
    model_path = os.path.join(save_dir, "nn_model.pt")
    torch.save({
        "model_state": best_model_state,
        "input_dim": X_train.shape[1],
        "feature_names": list(X.columns),
    }, model_path)
    print(f"\n✅ Model saved → {model_path}")

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return model, metrics, list(X.columns)


if __name__ == "__main__":
    train_nn_model()
