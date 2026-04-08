import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from orbit_simulator import parse_tle_file, get_position
from generate_training_data import (extract_expert_features,
                                    compute_pc_monte_carlo,
                                    get_position_covariance)

# ── LOAD TRAINED MODELS (SAFE) ───────────────────────────
print("Loading trained models...")

model_loaded = True

try:
    with open("model_binary.pkl", "rb") as f:
        model_binary = pickle.load(f)
    with open("model_risk.pkl", "rb") as f:
        model_risk = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    print("✅ Models loaded\n")

except Exception as e:
    print("⚠️ Models NOT found — using fallback mode\n")
    model_loaded = False

FEATURES = [
    "miss_distance_km", "miss_x_km", "miss_y_km", "miss_z_km",
    "alt1_km", "alt2_km", "alt_diff_km", "mean_altitude_km",
    "speed1_kms", "speed2_kms", "relative_speed_kms", "approach_velocity_kms",
    "combined_cov_trace", "mahalanobis_distance",
    "both_debris", "one_debris"
]

def predict_conjunction(pos1, pos2, name1, name2):

    is_deb1 = "DEB" in name1.upper()
    is_deb2 = "DEB" in name2.upper()

    features = extract_expert_features(pos1, pos2, is_deb1, is_deb2)
    pc = compute_pc_monte_carlo(pos1, pos2, is_deb1, is_deb2, n_trials=200)

    # 🔥 IF MODELS AVAILABLE → USE ML
    if model_loaded:
        try:
            X = pd.DataFrame([features])[FEATURES]
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).clip(-1e9, 1e9)

            will_collide  = model_binary.predict(X)[0]
            collide_proba = model_binary.predict_proba(X)[0][1]

            risk_encoded = model_risk.predict(X)[0]
            risk_label   = le.inverse_transform([risk_encoded])[0]
            risk_probas  = model_risk.predict_proba(X)[0]
            risk_conf    = max(risk_probas) * 100

        except:
            # fallback if model fails
            will_collide, collide_proba, risk_label, risk_conf = False, 0.3, "MEDIUM", 60

    # 🔥 FALLBACK MODE (NO MODELS)
    else:
        dist = features["miss_distance_km"]

        if dist < 50:
            risk_label = "CRITICAL"
            collide_proba = 0.9
        elif dist < 200:
            risk_label = "HIGH"
            collide_proba = 0.6
        elif dist < 500:
            risk_label = "MEDIUM"
            collide_proba = 0.3
        else:
            risk_label = "LOW"
            collide_proba = 0.1

        will_collide = collide_proba > 0.5
        risk_conf = collide_proba * 100

    return {
        "object1": name1,
        "object2": name2,
        "miss_dist_km": round(features["miss_distance_km"], 2),
        "will_collide": bool(will_collide),
        "collide_proba": round(collide_proba * 100, 2),
        "pc_monte_carlo": round(pc, 8),
        "risk_level": risk_label,
        "risk_confidence": round(risk_conf, 1),
        "alt1_km": round(pos1["altitude"], 1),
        "alt2_km": round(pos2["altitude"], 1),
        "mahal_dist": round(features["mahalanobis_distance"], 4),
    }

def run_ml_prediction(top_n=20):

    print("=" * 65)
    print("  ML-POWERED CONJUNCTION ASSESSMENT")
    print("=" * 65)

    all_objects = []
    all_objects += parse_tle_file("stations.tle")
    all_objects += parse_tle_file("debris.tle")

    now = datetime.now(timezone.utc)

    positions = {}
    for (name, line1, line2) in all_objects:
        pos = get_position(name, line1, line2, dt=now)
        if pos:
            positions[name] = pos

    close_pairs = []
    names = list(positions.keys())

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            p1, p2 = positions[n1], positions[n2]

            dx = p1["x"]-p2["x"]
            dy = p1["y"]-p2["y"]
            dz = p1["z"]-p2["z"]

            dist = (dx**2 + dy**2 + dz**2) ** 0.5

            if dist < 500:
                close_pairs.append((dist, n1, n2))

    close_pairs.sort()

    results = []
    for dist, n1, n2 in close_pairs[:top_n]:
        results.append(predict_conjunction(positions[n1], positions[n2], n1, n2))

    results.sort(key=lambda x: x["collide_proba"], reverse=True)

    print("\nTop Results:\n")
    for r in results[:5]:
        print(f"{r['risk_level']} | {r['object1']} × {r['object2']} | {r['collide_proba']}%")

    return results


if __name__ == "__main__":
    run_ml_prediction()