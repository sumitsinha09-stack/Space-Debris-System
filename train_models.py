import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier, XGBRegressor

# ── LOAD DATA ─────────────────────────────────────────────
print("=" * 60)
print("  XGBOOST TRAINING PIPELINE")
print("  Space Debris Collision Prediction")
print("=" * 60)

df = pd.read_csv("training_data.csv")
print(f"\n  Loaded {len(df)} samples")
print(f"  Features: {len(df.columns) - 3}\n")

# Features the model learns from
FEATURES = [
    "miss_distance_km", "miss_x_km", "miss_y_km", "miss_z_km",
    "alt1_km", "alt2_km", "alt_diff_km", "mean_altitude_km",
    "speed1_kms", "speed2_kms", "relative_speed_kms", "approach_velocity_kms",
    "combined_cov_trace", "mahalanobis_distance",
    "both_debris", "one_debris"
]
X = df[FEATURES]

# ── Clean infinite and NaN values ─────────────────────────
import numpy as np
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
X = X.clip(-1e9, 1e9)  # clip any extreme values
print(f"  Cleaned data shape: {X.shape}")
print(f"  Any remaining nulls: {X.isnull().sum().sum()}\n")

# ── MODEL 1: Will they collide? (Binary) ──────────────────
print("━" * 60)
print("  [1/3] BINARY CLASSIFIER — Will it collide?")
print("━" * 60)

y_binary = df["will_collide"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# scale_pos_weight handles class imbalance automatically
neg  = (y_train == 0).sum()
pos  = (y_train == 1).sum()
spw  = neg / pos
print(f"  Class balance — Safe: {neg} | Collision: {pos} | Weight: {spw:.1f}x\n")

model_binary = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = spw,   # handles imbalance
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
)
model_binary.fit(X_train, y_train, verbose=50)

y_pred  = model_binary.predict(X_test)
y_proba = model_binary.predict_proba(X_test)[:, 1]

print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["SAFE", "COLLISION"]))

auc = roc_auc_score(y_test, y_proba)
print(f"  ROC-AUC Score: {auc:.4f}  (1.0 = perfect, 0.5 = random)\n")

# ── MODEL 2: Risk level (Multi-class) ─────────────────────
print("━" * 60)
print("  [2/3] MULTI-CLASS CLASSIFIER — Risk level")
print("━" * 60)

le    = LabelEncoder()
y_risk = le.fit_transform(df["risk_level"])
print(f"  Classes: {list(le.classes_)}\n")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)

model_risk = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = "multi:softprob",
    num_class        = len(le.classes_),
    eval_metric      = "mlogloss",
    random_state     = 42,
)
model_risk.fit(X_train2, y_train2, verbose=50)

y_pred2 = model_risk.predict(X_test2)
print("\n  Classification Report:")
print(classification_report(y_test2, y_pred2,
      target_names=le.classes_))

# ── MODEL 3: Miss distance regression ─────────────────────
print("━" * 60)
print("  [3/3] REGRESSOR — Predicted miss distance (km)")
print("━" * 60)

y_dist = np.log1p(df["miss_distance_km"])  # log transform for stability

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X, y_dist, test_size=0.2, random_state=42
)

model_dist = XGBRegressor(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "rmse",
    random_state     = 42,
)
model_dist.fit(X_train3, y_train3, verbose=50)

y_pred3  = model_dist.predict(X_test3)
y_actual = np.expm1(y_test3)   # reverse log transform
y_pred_km = np.expm1(y_pred3)

mae = np.mean(np.abs(y_actual - y_pred_km))
print(f"\n  Mean Absolute Error: {mae:.2f} km")
print(f"  (On average, miss distance predictions are off by {mae:.2f} km)\n")

# ── FEATURE IMPORTANCE ────────────────────────────────────
print("━" * 60)
print("  FEATURE IMPORTANCE (what the model cares about most)")
print("━" * 60)
importance = pd.Series(
    model_binary.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

for feat, score in importance.head(8).items():
    bar = "█" * int(score * 200)
    print(f"  {feat:<28} {score:.4f}  {bar}")

# ── SAVE MODELS ────────────────────────────────────────────
print("\n  Saving models...")
with open("model_binary.pkl",  "wb") as f: pickle.dump(model_binary, f)
with open("model_risk.pkl",    "wb") as f: pickle.dump(model_risk, f)
with open("model_dist.pkl",    "wb") as f: pickle.dump(model_dist, f)
with open("label_encoder.pkl", "wb") as f: pickle.dump(le, f)

print("  ✅ model_binary.pkl  — collision yes/no")
print("  ✅ model_risk.pkl    — risk level")
print("  ✅ model_dist.pkl    — miss distance")
print("  ✅ label_encoder.pkl — class names")

print(f"\n{'=' * 60}")
print("  TRAINING COMPLETE")
print(f"  ROC-AUC: {auc:.4f}")
print(f"  Miss dist MAE: {mae:.2f} km")
print(f"{'=' * 60}")
print("\n  Next: python3 ml_predictor.py")