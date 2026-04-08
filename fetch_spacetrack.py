import requests
import json

# ── Your space-track.org credentials ─────────────────────
USERNAME = "krishnashu06.india@gmail.com"   # ← change this
PASSWORD = "Getlost12345678"             # ← change this

BASE_URL = "https://www.space-track.org"

def login(session):
    """Login to space-track.org"""
    print("🔐 Logging in to Space-Track.org...")
    resp = session.post(
        f"{BASE_URL}/ajaxauth/login",
        data={"identity": USERNAME, "password": PASSWORD}
    )
    if resp.status_code == 200:
        print("✅ Login successful!\n")
        return True
    else:
        print(f"❌ Login failed: {resp.status_code}")
        return False

def fetch_debris_near_iss(session):
    """
    Fetch debris objects between 300-500km altitude
    (same shell as ISS — highest collision risk zone)
    """
    print("📡 Fetching debris catalog (300-500km altitude)...")

    # Query: debris objects, LEO altitude range, limit 500
    url = (
        f"{BASE_URL}/basicspacedata/query/class/gp/"
        f"MEAN_MOTION/>11/ECCENTRICITY/<0.01/"
        f"OBJECT_TYPE/DEBRIS/"
        f"orderby/NORAD_CAT_ID/limit/500/format/json"
    )

    resp = session.get(url)

    if resp.status_code != 200:
        print(f"❌ Fetch failed: {resp.status_code}")
        return []

    data = resp.json()
    print(f"✅ Got {len(data)} debris objects!\n")
    return data

def fetch_active_satellites(session):
    """Fetch active satellites in LEO"""
    print("🛰️  Fetching active satellites...")

    url = (
        f"{BASE_URL}/basicspacedata/query/class/gp/"
        f"MEAN_MOTION/>11/ECCENTRICITY/<0.01/"
        f"OBJECT_TYPE/PAYLOAD/"
        f"orderby/NORAD_CAT_ID/limit/100/format/json"
    )

    resp = session.get(url)
    data = resp.json()
    print(f"✅ Got {len(data)} active satellites!\n")
    return data

def save_as_tle(objects, filename):
    """Convert JSON response to TLE format and save"""
    with open(filename, "w") as f:
        for obj in objects:
            # space-track returns TLE lines directly
            name  = obj.get("OBJECT_NAME", "UNKNOWN")
            line1 = obj.get("TLE_LINE1", "")
            line2 = obj.get("TLE_LINE2", "")

            if line1 and line2:
                f.write(f"{name}\n{line1}\n{line2}\n")

    print(f"💾 Saved → {filename}")

def run():
    print("=" * 55)
    print("  SPACE-TRACK.ORG LIVE DATA FETCHER")
    print("=" * 55 + "\n")

    with requests.Session() as session:
        # Login
        if not login(session):
            print("Check your username/password and try again.")
            return

        # Fetch debris
        debris = fetch_debris_near_iss(session)
        if debris:
            save_as_tle(debris, "debris.tle")

        # Fetch satellites
        satellites = fetch_active_satellites(session)
        if satellites:
            save_as_tle(satellites, "stations.tle")

        # Summary
        total = len(debris) + len(satellites)
        print(f"\n{'=' * 55}")
        print(f"  ✅ DONE — {total} real objects loaded!")
        print(f"  🛰️  Satellites : {len(satellites)}")
        print(f"  ☄️   Debris     : {len(debris)}")
        print(f"  📁 Files saved : stations.tle, debris.tle")
        print(f"{'=' * 55}\n")
        print("Now run: python3 main.py")

run()