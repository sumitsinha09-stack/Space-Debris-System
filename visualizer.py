import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime, timezone, timedelta
from orbit_simulator import parse_tle_file, get_position

# ── Colors ──────────────────────────────────────────────
COLORS = {
    "ISS (ZARYA)":     "#00d4ff",
    "CSS (TIANHE)":    "#00ff88",
    "COSMOS 2251 DEB": "#ff7b00",
    "FENGYUN 1C DEB":  "#ff2255",
}
DEFAULT_SAT_COLOR = "#aaaaaa"

def draw_earth(ax):
    """Draw a blue wireframe Earth"""
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    R = 6371  # Earth radius km
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#0a2a4a", linewidth=0.3, alpha=0.6)

def get_orbit_trail(name, line1, line2, minutes=90):
    """Get the full orbit path (last 90 minutes of positions)"""
    now = datetime.now(timezone.utc)
    xs, ys, zs = [], [], []
    for i in range(0, minutes, 2):
        t = now - timedelta(minutes=minutes) + timedelta(minutes=i)
        pos = get_position(name, line1, line2, dt=t)
        if pos:
            xs.append(pos["x"])
            ys.append(pos["y"])
            zs.append(pos["z"])
    return xs, ys, zs

def run_visualizer():
    # Load objects
    all_objects = []
    all_objects += parse_tle_file("stations.tle")
    all_objects += parse_tle_file("debris.tle")

    print(f"Loaded {len(all_objects)} objects — building visualization...\n")

    # ── Figure setup ────────────────────────────────────
    fig = plt.figure(figsize=(12, 9), facecolor="#020b18")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#020b18")

    ax.set_title("SPACE DEBRIS COLLISION PREDICTION SYSTEM",
                 color="#00d4ff", fontsize=11, pad=15, fontfamily="monospace")

    # Axis styling
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#0e2a4a")
    ax.tick_params(colors="#1a3a5a", labelsize=6)
    ax.set_xlabel("X (km)", color="#1a3a5a", fontsize=7)
    ax.set_ylabel("Y (km)", color="#1a3a5a", fontsize=7)
    ax.set_zlabel("Z (km)", color="#1a3a5a", fontsize=7)
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-10000, 10000)
    ax.set_zlim(-10000, 10000)

    # Draw Earth
    draw_earth(ax)

    # ── Draw orbit trails ────────────────────────────────
    print("Calculating orbit trails...")
    for (name, line1, line2) in all_objects:
        xs, ys, zs = get_orbit_trail(name, line1, line2)
        color = COLORS.get(name, DEFAULT_SAT_COLOR)
        ax.plot(xs, ys, zs, color=color, linewidth=0.6, alpha=0.4)
        print(f"  ✅ {name}")

    # ── Draw current positions ───────────────────────────
    now = datetime.now(timezone.utc)
    sat_dots = []
    labels   = []

    for (name, line1, line2) in all_objects:
        pos = get_position(name, line1, line2, dt=now)
        if not pos:
            continue
        color = COLORS.get(name, DEFAULT_SAT_COLOR)
        is_debris = "DEB" in name

        dot = ax.scatter(
            pos["x"], pos["y"], pos["z"],
            color=color,
            s=60 if not is_debris else 40,
            marker="o" if not is_debris else "x",
            zorder=5,
            depthshade=False
        )
        label = ax.text(
            pos["x"], pos["y"], pos["z"] + 300,
            name.split("(")[0].strip(),
            color=color, fontsize=6, fontfamily="monospace"
        )
        sat_dots.append((dot, pos))
        labels.append(label)

    # ── Legend ───────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#00d4ff", markersize=7, label="Active Satellite"),
        Line2D([0],[0], marker="x", color="#ff7b00", markersize=7, label="Debris (Warning)"),
        Line2D([0],[0], marker="x", color="#ff2255", markersize=7, label="Debris (High Risk)"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              facecolor="#040f1f", edgecolor="#0e2a4a",
              labelcolor="white", fontsize=7)

    # ── Info text ────────────────────────────────────────
    fig.text(0.02, 0.02,
             f"Objects: {len(all_objects)}  |  Time: {now.strftime('%Y-%m-%d %H:%M UTC')}  |  Orbits: 90-min trail",
             color="#4a6a8a", fontsize=7, fontfamily="monospace")

    print("\n✅ Visualization ready! Close the window to exit.")
    plt.tight_layout()
    plt.show()

run_visualizer()