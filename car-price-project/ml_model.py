"""
ML Ensemble Model (Model 1)
ประกอบด้วย 3 โมเดล:
  1. Random Forest Regressor
  2. Gradient Boosting Regressor
  3. XGBoost Regressor
รวมกันแบบ Voting (Stacking) ด้วย Ridge Regression เป็น meta-learner
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_preparation import prepare_v3


# ────────────────────────────────────────────────
# BUILD ENSEMBLE
# ────────────────────────────────────────────────
def build_ensemble():
    base_estimators = [
        ("rf",  RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42,
                             eval_metric="rmse", verbosity=0)),
    ]
    meta_learner = Ridge()
    ensemble = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
    )
    return ensemble


# ────────────────────────────────────────────────
# TRAIN & EVALUATE
# ────────────────────────────────────────────────
def train_ml_model(base_dir="data", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("ML ENSEMBLE MODEL TRAINING")
    print("=" * 50)

    X, y, encoders, scaler, df_clean = prepare_v3(base_dir, save_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    model = build_ensemble()
    print("\nTraining Stacking Ensemble (RF + GBR + XGB → Ridge)...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊 Test Results:")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  RMSE : {rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

    # Cross-validation R²
    cv_scores = cross_val_score(
        build_ensemble(), X, y, cv=3, scoring="r2", n_jobs=-1
    )
    print(f"  CV R² (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance (from RF base model)
    rf_model   = model.estimators_[0]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"feature": X.columns, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False)

    # Plot
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="#2196F3")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("Actual vs Predicted (ML Ensemble)")

    # Feature Importance
    axes[1].barh(feat_df["feature"][:10][::-1], feat_df["importance"][:10][::-1], color="#4CAF50")
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top 10 Feature Importances (RF)")

    plt.tight_layout()
    plt.savefig("plots/ml_results.png", dpi=120)
    plt.close()
    print("  Plot saved → plots/ml_results.png")

    # Save model
    model_path = os.path.join(save_dir, "ml_ensemble_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved → {model_path}")

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "cv_r2_mean": cv_scores.mean()}
    return model, metrics, X.columns.tolist()


if __name__ == "__main__":
    train_ml_model()
