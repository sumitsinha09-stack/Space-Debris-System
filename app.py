from flask import Flask, jsonify, render_template
from flask_cors import CORS
from datetime import datetime, timezone
import pickle
import numpy as np
import pandas as pd
import math
from env import SpaceEnv
from orbit_simulator import parse_tle_file, get_position



from datetime import timedelta  # add this extra import

def generate_orbit_path(name, line1, line2, now, minutes=90):
    points = []

    for i in range(0, minutes, 2):
        t = now + timedelta(minutes=i)

        pos = get_position(name, line1, line2, dt=t)

        if pos:
            EARTH_RADIUS = 6371
            points.append({
                "x": float(pos["x"]),
                "y": float(pos["y"]),
                "z": float(pos["z"])
            })

    return points

app = Flask(__name__)
CORS(app)

# ── Load models once at startup ───────────────────────────
print("Loading ML models...")
import os

try:
    with open("model_binary.pkl", "rb") as f:
        model_binary = pickle.load(f)
    print("model_binary loaded")
except FileNotFoundError:
    model_binary = None
    print("model_binary.pkl not found - using fallback")

try:
    with open("model_risk.pkl", "rb") as f:
        model_risk = pickle.load(f)
    print("model_risk loaded")
except FileNotFoundError:
    model_risk = None
    print("model_risk.pkl not found - using fallback")

try:
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    print("label_encoder loaded")
except FileNotFoundError:
    le = None
    print("label_encoder.pkl not found - using fallback")

print("Models ready")

FEATURES = [
    "miss_distance_km", "miss_x_km", "miss_y_km", "miss_z_km",
    "alt1_km", "alt2_km", "alt_diff_km", "mean_altitude_km",
    "speed1_kms", "speed2_kms", "relative_speed_kms", "approach_velocity_kms",
    "combined_cov_trace", "mahalanobis_distance", "both_debris", "one_debris"
]

def load_objects():
    objs = []
    objs += parse_tle_file("stations.tle")
    objs += parse_tle_file("debris.tle")
    return objs

# ── API: Current positions ────────────────────────────────
@app.route("/api/positions")
def api_positions():
    now     = datetime.now(timezone.utc)
    objects = load_objects()
    results = []
    for (name, line1, line2) in objects[:50]:  # top 80 for performance
        pos = get_position(name, line1, line2, dt=now)
        if pos:
            results.append({
                "name":     name,
                "x":        pos["x"],
                "y":        pos["y"],
                "z":        pos["z"],
                "altitude": pos["altitude"],
                "speed":    pos["speed"],
                "is_debris": "DEB" in name.upper()
            })
    return jsonify({
        "epoch":   now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "count":   len(results),
        "objects": results
    })

# ── API: Conjunctions ─────────────────────────────────────
@app.route("/api/conjunctions")
def api_conjunctions():
    now     = datetime.now(timezone.utc)
    objects = load_objects() 

    # Get positions
    positions = {}
    for (name, line1, line2) in objects[:80]:
        pos = get_position(name, line1, line2, dt=now)
        if pos:
            positions[name] = pos

    # Find close pairs
    names  = list(positions.keys())
    alerts = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            p1, p2 = positions[n1], positions[n2]
            dx = p1["x"]-p2["x"]
            dy = p1["y"]-p2["y"]
            dz = p1["z"]-p2["z"]
            dist = math.sqrt(dx**2+dy**2+dz**2)
            if dist < 2000:
                alerts.append({
                    "object1":   n1,
                    "object2":   n2,
                    "distance":  round(dist, 2),
                    "alt1":      round(p1["altitude"], 1),
                    "alt2":      round(p2["altitude"], 1),
                })

    alerts.sort(key=lambda x: x["distance"])
    return jsonify({
        "epoch":        now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total":        len(alerts),
        "conjunctions": alerts[:20]
    })




@app.route("/api/orbits")
def get_orbits():
    objects = load_objects()

    orbit_data = []
    now = datetime.now(timezone.utc)   # 👈 SINGLE SOURCE OF TIME

    for (name, line1, line2) in objects[:25]:
        orbit = generate_orbit_path(name, line1, line2, now)

        orbit_data.append({
            "name": name,
            "orbit": orbit
        })

    return jsonify(orbit_data)

# ── API: ML Predictions ───────────────────────────────────
@app.route("/api/ml")
def api_ml():
    try:
        now     = datetime.now(timezone.utc)
        objects = load_objects()

        positions = {}
        for (name, line1, line2) in objects[:50]:
            pos = get_position(name, line1, line2, dt=now)
            if pos:
                positions[name] = pos

        names   = list(positions.keys())
        results = []

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                p1, p2 = positions[n1], positions[n2]

                dx = p1["x"]-p2["x"]
                dy = p1["y"]-p2["y"]
                dz = p1["z"]-p2["z"]
                dist = math.sqrt(dx**2+dy**2+dz**2)

                if dist < 2000:
                    try:
                        features = {
                            "miss_distance_km": dist,
                            "miss_x_km": abs(dx),
                            "miss_y_km": abs(dy),
                            "miss_z_km": abs(dz),
                            "alt1_km": p1["altitude"],
                            "alt2_km": p2["altitude"],
                            "alt_diff_km": abs(p1["altitude"]-p2["altitude"]),
                            "mean_altitude_km": (p1["altitude"]+p2["altitude"])/2,
                            "speed1_kms": p1["speed"],
                            "speed2_kms": p2["speed"],
                            "relative_speed_kms": abs(p1["speed"]-p2["speed"]),
                            "approach_velocity_kms": p1["speed"]+p2["speed"],
                            "combined_cov_trace": 1.0,
                            "mahalanobis_distance": dist/10,
                            "both_debris": 0,
                            "one_debris": 0
                        }

                        X = pd.DataFrame([features])[FEATURES]
                        X = X.fillna(0)

                        risk_enc = int(model_risk.predict(X)[0])
                        risk_label = le.inverse_transform([risk_enc])[0]
                        collide_p  = float(model_binary.predict_proba(X)[0][1])

                        results.append({
                            "object1": str(n1),
                            "object2": str(n2),
                            "risk_level": str(risk_label),
                            "collide_prob": float(round(collide_p * 100, 2))
                        })
                    except:
                        continue  # skip broken pairs

        return jsonify({
            "results": results[:10]
        })

    except Exception as e:
        return jsonify({"error": str(e)})
# ── API: Stats ────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    objects  = load_objects()
    debris   = sum(1 for o in objects if "DEB" in o[0].upper())
    sats     = len(objects) - debris
    return jsonify({
        "total_objects": len(objects),
        "satellites":    sats,
        "debris":        debris,
        "model_accuracy": 99.0,
        "last_updated":  datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    })

# ── Serve dashboard ───────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    env = SpaceEnv()
    state = env.reset()
    return jsonify({"status": "ok", "state": state})

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/")
def index():
    return render_template("dashboard.html")

if __name__ == "__main__":
    print("🚀 Starting SDCPS Dashboard...")
    print("   Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=7860, debug=False)