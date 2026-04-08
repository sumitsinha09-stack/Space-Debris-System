import math
import random
import csv
import numpy as np
from datetime import datetime, timezone, timedelta
from orbit_simulator import parse_tle_file, get_position

# ── CONSTANTS ────────────────────────────────────────────
COLLISION_RADIUS_KM = 0.010
N_MONTE_CARLO       = 500  # reduced for speed, increase for production

# ── COVARIANCE MODEL ─────────────────────────────────────
def get_position_covariance(altitude_km, is_debris):
    if is_debris:
        sigma = 0.5 + (altitude_km / 1000.0)
    else:
        sigma = 0.1
    return np.diag([sigma**2, sigma**2, (sigma*1.5)**2])

# ── MONTE CARLO Pc ────────────────────────────────────────
def compute_pc_monte_carlo(pos1, pos2, is_debris1, is_debris2, n_trials=N_MONTE_CARLO):
    r1   = np.array([pos1["x"], pos1["y"], pos1["z"]])
    r2   = np.array([pos2["x"], pos2["y"], pos2["z"]])
    cov1 = get_position_covariance(pos1["altitude"], is_debris1)
    cov2 = get_position_covariance(pos2["altitude"], is_debris2)
    collisions = 0
    for _ in range(n_trials):
        noise1 = np.random.multivariate_normal([0, 0, 0], cov1)
        noise2 = np.random.multivariate_normal([0, 0, 0], cov2)
        p1 = r1 + noise1
        p2 = r2 + noise2
        if np.linalg.norm(p1 - p2) < COLLISION_RADIUS_KM:
            collisions += 1
    return collisions / n_trials

# ── LABEL FROM DISTANCE (ESA/NASA thresholds) ────────────
def label_from_distance(distance_km):
    if distance_km < 0.1:
        return 1, "CRITICAL"
    elif distance_km < 1.0:
        return 1, "HIGH"
    elif distance_km < 5.0:
        return 1, "MEDIUM"
    elif distance_km < 50.0:
        return 0, "LOW"
    else:
        return 0, "SAFE"

# ── FEATURE EXTRACTION ────────────────────────────────────
def extract_expert_features(pos1, pos2, is_debris1, is_debris2):
    r1 = np.array([pos1["x"], pos1["y"], pos1["z"]])
    r2 = np.array([pos2["x"], pos2["y"], pos2["z"]])

    miss_vector = r2 - r1
    miss_dist   = np.linalg.norm(miss_vector)
    alt_diff    = abs(pos1["altitude"] - pos2["altitude"])
    mean_alt    = (pos1["altitude"] + pos2["altitude"]) / 2
    rel_speed   = abs(pos1["speed"] - pos2["speed"])

    cov1 = get_position_covariance(pos1["altitude"], is_debris1)
    cov2 = get_position_covariance(pos2["altitude"], is_debris2)
    combined_cov_trace = np.trace(cov1) + np.trace(cov2)

    combined_cov = cov1 + cov2
    try:
        inv_cov    = np.linalg.inv(combined_cov)
        mahal_dist = math.sqrt(float(miss_vector @ inv_cov @ miss_vector))
    except:
        mahal_dist = miss_dist

    return {
        "miss_distance_km":      round(miss_dist, 4),
        "miss_x_km":             round(abs(miss_vector[0]), 4),
        "miss_y_km":             round(abs(miss_vector[1]), 4),
        "miss_z_km":             round(abs(miss_vector[2]), 4),
        "alt1_km":               round(pos1["altitude"], 2),
        "alt2_km":               round(pos2["altitude"], 2),
        "alt_diff_km":           round(alt_diff, 4),
        "mean_altitude_km":      round(mean_alt, 2),
        "speed1_kms":            round(pos1["speed"], 6),
        "speed2_kms":            round(pos2["speed"], 6),
        "relative_speed_kms":    round(rel_speed, 6),
        "approach_velocity_kms": round(pos1["speed"] + pos2["speed"], 6),
        "combined_cov_trace":    round(combined_cov_trace, 6),
        "mahalanobis_distance":  round(mahal_dist, 6),
        "both_debris":           int(is_debris1 and is_debris2),
        "one_debris":            int(is_debris1 or is_debris2),
    }

# ── MAIN GENERATOR ────────────────────────────────────────
def generate_data(n_samples=3000):
    print("=" * 60)
    print("  EXPERT ML TRAINING DATA GENERATOR")
    print("  Labels: ESA/NASA distance thresholds")
    print("=" * 60)

    all_objects = []
    all_objects += parse_tle_file("stations.tle")
    all_objects += parse_tle_file("debris.tle")
    print(f"\n  Loaded {len(all_objects)} real orbital objects")

    rows   = []
    errors = 0
    now    = datetime.now(timezone.utc)

    n_real      = n_samples // 2
    n_synthetic = n_samples - n_real

    print(f"  Real random pairs     : {n_real}")
    print(f"  Synthetic close pairs : {n_synthetic}\n")

    # ── PART 1: Real random pairs ─────────────────────────
    print("  [1/2] Generating real conjunction pairs...")
    for i in range(n_real):
        obj1, obj2 = random.sample(all_objects, 2)
        future     = now + timedelta(minutes=random.randint(0, 1440))
        pos1 = get_position(obj1[0], obj1[1], obj1[2], dt=future)
        pos2 = get_position(obj2[0], obj2[1], obj2[2], dt=future)

        if not pos1 or not pos2:
            errors += 1
            continue

        is_deb1  = "DEB" in obj1[0].upper()
        is_deb2  = "DEB" in obj2[0].upper()
        features = extract_expert_features(pos1, pos2, is_deb1, is_deb2)
        pc       = compute_pc_monte_carlo(pos1, pos2, is_deb1, is_deb2)
        will_collide, risk_level = label_from_distance(features["miss_distance_km"])

        rows.append({**features,
                     "pc_value":     pc,
                     "will_collide": will_collide,
                     "risk_level":   risk_level})

        if (i+1) % 250 == 0:
            print(f"  ✅ {i+1}/{n_real} real pairs done")

    # ── PART 2: Synthetic close approaches ────────────────
    print("\n  [2/2] Generating synthetic close approaches...")

    distance_bands = [
        (0.001, 0.100, 214),   # CRITICAL
        (0.100, 1.000, 214),   # HIGH
        (1.000, 5.000, 214),   # MEDIUM
        (5.000, 50.00, 214),   # LOW
        (50.00, 500.0, 214),   # SAFE
        (500.0, 2000., 214),   # SAFE far
        (2000., 10000, 216),   # SAFE very far
    ]

    for band_min, band_max, count in distance_bands:
        for j in range(count):
            obj1   = random.choice(all_objects)
            future = now + timedelta(minutes=random.randint(0, 1440))
            pos1   = get_position(obj1[0], obj1[1], obj1[2], dt=future)

            if not pos1:
                errors += 1
                continue

            # Place object 2 at controlled distance
            target_dist = random.uniform(band_min, band_max)
            direction   = np.array([random.gauss(0,1),
                                    random.gauss(0,1),
                                    random.gauss(0,1)])
            direction  /= np.linalg.norm(direction)

            pos2 = {
                "x":        pos1["x"] + direction[0] * target_dist,
                "y":        pos1["y"] + direction[1] * target_dist,
                "z":        pos1["z"] + direction[2] * target_dist,
                "altitude": pos1["altitude"] + random.uniform(-5, 5),
                "speed":    pos1["speed"]    + random.uniform(-0.5, 0.5),
            }

            is_deb1  = "DEB" in obj1[0].upper()
            is_deb2  = random.choice([True, False])
            features = extract_expert_features(pos1, pos2, is_deb1, is_deb2)
            pc       = compute_pc_monte_carlo(pos1, pos2, is_deb1, is_deb2)
            will_collide, risk_level = label_from_distance(features["miss_distance_km"])

            rows.append({**features,
                         "pc_value":     pc,
                         "will_collide": will_collide,
                         "risk_level":   risk_level})

        print(f"  ✅ Band {band_min}-{band_max} km done")

    # Shuffle
    random.shuffle(rows)

    # Save
    with open("training_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    flagged = sum(r["will_collide"] for r in rows)
    by_risk = {}
    for r in rows:
        by_risk[r["risk_level"]] = by_risk.get(r["risk_level"], 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  ✅ {len(rows)} samples saved → training_data.csv")
    print(f"\n  Risk distribution:")
    for level, count in sorted(by_risk.items()):
        bar = "█" * (count // 15)
        print(f"  {level:<10} {count:>5}  {bar}")
    print(f"\n  Will collide : {flagged}")
    print(f"  Safe         : {len(rows) - flagged}")
    print(f"  Errors       : {errors}")
    print(f"{'=' * 60}")
    print("\n  Next: python3 train_models.py")

generate_data(n_samples=3000)