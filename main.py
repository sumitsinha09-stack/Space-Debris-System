import os
import json
from datetime import datetime, timezone
from orbit_simulator import parse_tle_file, get_position
from collision_detector import run_prediction

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║     SPACE DEBRIS COLLISION PREDICTION SYSTEM             ║
║     Built with SGP4 · Space-Track.org · Python           ║
╚══════════════════════════════════════════════════════════╝
    """)

def save_report(alerts):
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_conjunctions": len(alerts),
        "critical": len([a for a in alerts if "CRITICAL" in a["level"]]),
        "warning":  len([a for a in alerts if "WARNING"  in a["level"]]),
        "monitor":  len([a for a in alerts if "MONITOR"  in a["level"]]),
        "events":   alerts
    }
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved → {filename}")
    return filename

def print_summary(alerts):
    critical = [a for a in alerts if "CRITICAL" in a["level"]]
    warning  = [a for a in alerts if "WARNING"  in a["level"]]
    monitor  = [a for a in alerts if "MONITOR"  in a["level"]]

    print("\n╔══════════════════════════════════════╗")
    print("║          PREDICTION SUMMARY          ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  🔴 Critical  : {str(len(critical)).ljust(22)}║")
    print(f"║  🟡 Warning   : {str(len(warning)).ljust(22)}║")
    print(f"║  🔵 Monitor   : {str(len(monitor)).ljust(22)}║")
    print(f"║  📊 Total     : {str(len(alerts)).ljust(22)}║")
    print("╚══════════════════════════════════════╝")

    if critical:
        print("\n🚨 CRITICAL EVENTS:")
        for a in critical[:3]:
            print(f"   {a['object1']} × {a['object2']} → {a['distance']} km at {a['time']}")

def ask_user(question):
    ans = input(f"\n{question} (y/n): ").strip().lower()
    return ans == "y"

def run():
    print_banner()

    # ── STEP 1: Load Data ────────────────────────────────
    print("━" * 55)
    print("[ 1/4 ] LOADING SATELLITE DATA...")
    print("━" * 55)

    with open("stations.tle") as f:
        sat_count = f.read().count("1 ")
    with open("debris.tle") as f:
        deb_count = f.read().count("1 ")

    print(f"  🛰️  Satellites : {sat_count}")
    print(f"  ☄️   Debris     : {deb_count}")
    print(f"  📊 Total       : {sat_count + deb_count}")
    print(f"  ✅ Using Space-Track.org live data\n")

    # ── STEP 2: Current Positions ────────────────────────
    print("━" * 55)
    print("[ 2/4 ] CURRENT ORBITAL POSITIONS (sample)")
    print("━" * 55)

    all_objects = []
    all_objects += parse_tle_file("stations.tle")
    all_objects += parse_tle_file("debris.tle")

    now = datetime.now(timezone.utc)
    print(f"  Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Show first 8 objects as sample
    for (name, line1, line2) in all_objects[:8]:
        pos = get_position(name, line1, line2)
        if pos:
            tag = "🛰️ " if "DEB" not in name else "☄️ "
            print(f"  {tag} {pos['name']:<25} Alt: {pos['altitude']:>7.1f} km   Speed: {pos['speed']} km/s")

    print(f"\n  ... and {len(all_objects) - 8} more objects being tracked\n")

    # ── STEP 3: Collision Prediction ─────────────────────
    print("━" * 55)
    print("[ 3/4 ] RUNNING COLLISION PREDICTION (24h window)")
    print("━" * 55)
    print("  ⏳ This may take 2-3 minutes for 1800+ objects...\n")
    alerts = run_prediction(hours=24, step_minutes=30)

    # ── STEP 4: Report ────────────────────────────────────
    print("━" * 55)
    print("[ 4/4 ] GENERATING REPORT")
    print("━" * 55)
    print_summary(alerts)
    save_report(alerts)

    # ── STEP 5: Visualizer ────────────────────────────────
    if ask_user("\n🌍 Launch 3D visualizer?"):
        from visualizer import run_visualizer
        run_visualizer()

    print("\n✅ System run complete. Stay safe up there! 🛸\n")

run()