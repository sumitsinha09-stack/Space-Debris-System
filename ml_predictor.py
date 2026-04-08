import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from orbit_simulator import parse_tle_file, get_position
from generate_training_data import (extract_expert_features,
                                    compute_pc_monte_carlo,
                                    get_position_covariance)

# ── LOAD TRAINED MODELS ───────────────────────────────────
print("Loading trained models...")
with open("model_binary.pkl",  "rb") as f: model_binary  = pickle.load(f)
with open("model_risk.pkl",    "rb") as f: model_risk    = pickle.load(f)
with open("label_encoder.pkl", "rb") as f: le            = pickle.load(f)
print("✅ Models loaded\n")

FEATURES = [
    "miss_distance_km", "miss_x_km", "miss_y_km", "miss_z_km",
    "alt1_km", "alt2_km", "alt_diff_km", "mean_altitude_km",
    "speed1_kms", "speed2_kms", "relative_speed_kms", "approach_velocity_kms",
    "combined_cov_trace", "mahalanobis_distance",
    "both_debris", "one_debris"
]

def predict_conjunction(pos1, pos2, name1, name2):
    """
    Run all 3 predictions for a conjunction pair.
    Returns full risk assessment.
    """
    is_deb1 = "DEB" in name1.upper()
    is_deb2 = "DEB" in name2.upper()

    # Extract features
    features = extract_expert_features(pos1, pos2, is_deb1, is_deb2)

    # Monte Carlo Pc
    pc = compute_pc_monte_carlo(pos1, pos2, is_deb1, is_deb2, n_trials=500)

    # Build feature vector
    X = pd.DataFrame([features])[FEATURES]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).clip(-1e9, 1e9)

    # Model 1: Will it collide?
    will_collide  = model_binary.predict(X)[0]
    collide_proba = model_binary.predict_proba(X)[0][1]

    # Model 2: Risk level
    risk_encoded = model_risk.predict(X)[0]
    risk_label   = le.inverse_transform([risk_encoded])[0]
    risk_probas  = model_risk.predict_proba(X)[0]
    risk_conf    = max(risk_probas) * 100

    return {
        "object1":       name1,
        "object2":       name2,
        "miss_dist_km":  round(features["miss_distance_km"], 2),
        "will_collide":  bool(will_collide),
        "collide_proba": round(collide_proba * 100, 2),
        "pc_monte_carlo":round(pc, 8),
        "risk_level":    risk_label,
        "risk_confidence": round(risk_conf, 1),
        "alt1_km":       round(pos1["altitude"], 1),
        "alt2_km":       round(pos2["altitude"], 1),
        "mahal_dist":    round(features["mahalanobis_distance"], 4),
    }

def run_ml_prediction(top_n=20):
    """
    Scan all object pairs, run ML prediction on closest ones.
    Returns ranked list of highest-risk conjunctions.
    """
    print("=" * 65)
    print("  ML-POWERED CONJUNCTION ASSESSMENT")
    print("  XGBoost + Monte Carlo Pc (NASA CARA method)")
    print("=" * 65)

    # Load objects
    all_objects = []
    all_objects += parse_tle_file("stations.tle")
    all_objects += parse_tle_file("debris.tle")
    print(f"\n  Objects loaded : {len(all_objects)}")

    now = datetime.now(timezone.utc)
    print(f"  Epoch          : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Scanning pairs : checking {len(all_objects)**2 // 2:,} combinations\n")

    # Get all current positions
    print("  Computing current positions...")
    positions = {}
    for (name, line1, line2) in all_objects:
        pos = get_position(name, line1, line2, dt=now)
        if pos:
            positions[name] = pos

    print(f"  Valid positions: {len(positions)}\n")

    # Quick pre-filter — only check pairs within 500km
    # (no need to run ML on objects 10,000km apart)
    print("  Pre-filtering close pairs (< 500km)...")
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
    print(f"  Close pairs found: {len(close_pairs)}\n")

    if not close_pairs:
        print("  ✅ No close approaches detected at this moment.")
        print("  This is normal — try again in a few minutes as")
        print("  objects move into different positions.\n")
        return []

    # Run ML on close pairs
    print(f"  Running ML assessment on {min(len(close_pairs), top_n)} closest pairs...")
    print("─" * 65)

    results = []
    for dist, n1, n2 in close_pairs[:top_n]:
        result = predict_conjunction(positions[n1], positions[n2], n1, n2)
        results.append(result)

    # Sort by collision probability
    results.sort(key=lambda x: x["collide_proba"], reverse=True)

    # ── PRINT RESULTS ────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  TOP CONJUNCTION ASSESSMENTS")
    print(f"{'═' * 65}\n")

    icons = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🔵",
        "SAFE":     "🟢"
    }

    for i, r in enumerate(results[:10], 1):
        icon = icons.get(r["risk_level"], "⚪")
        print(f"  [{i:02d}] {icon} {r['risk_level']:<10} "
              f"Confidence: {r['risk_confidence']:.0f}%")
        print(f"       {r['object1'][:28]}")
        print(f"     × {r['object2'][:28]}")
        print(f"       Miss dist : {r['miss_dist_km']:>10.2f} km")
        print(f"       Collision : {r['collide_proba']:>9.4f}%  "
              f"(ML model)")
        print(f"       Pc (MC)   : {r['pc_monte_carlo']:>9.6f}   "
              f"(Monte Carlo)")
        print(f"       Altitudes : {r['alt1_km']:.0f} km / {r['alt2_km']:.0f} km")
        print(f"       Mahal dist: {r['mahal_dist']:.4f}")
        print()

    # ── SUMMARY ──────────────────────────────────────────
    by_risk = {}
    for r in results:
        by_risk[r["risk_level"]] = by_risk.get(r["risk_level"], 0) + 1

    print(f"{'─' * 65}")
    print(f"  SUMMARY — {len(results)} conjunctions assessed")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE"]:
        count = by_risk.get(level, 0)
        if count:
            icon = icons[level]
            print(f"  {icon} {level:<10} : {count}")

    print(f"{'═' * 65}\n")
    return results

run_ml_prediction()