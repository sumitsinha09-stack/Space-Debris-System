from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
import math
import random


# ───────── GLOBAL FUNCTIONS (FOR app.py) ───────── #

def parse_tle_file(filename):
    satellites = []

    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for i in range(0, len(lines) - 2, 3):
        name  = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        satellites.append((name, line1, line2))

    return satellites


def get_position(name, line1, line2, dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)

    sat = Satrec.twoline2rv(line1, line2)

    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second)

    error, position, velocity = sat.sgp4(jd, fr)

    if error != 0:
        return None

    x, y, z = position
    vx, vy, vz = velocity

    distance = math.sqrt(x**2 + y**2 + z**2)
    altitude = distance - 6371

    return {
        "name": name,
        "x": round(x, 2),
        "y": round(y, 2),
        "z": round(z, 2),
        "altitude": round(altitude, 2),
        "speed": round(math.sqrt(vx**2 + vy**2 + vz**2), 4)
    }


# ───────── RL SIMULATOR ───────── #

class OrbitSimulator:

    def __init__(self):
        self.satellites = []
        self.debris = []
        self.current_time = None
        self.agent_sat = None   # 🔥 important
        self.load_data()

    def load_data(self):
        self.satellites = parse_tle_file("stations.tle")
        self.debris = parse_tle_file("debris.tle")

    def get_position(self, name, line1, line2, dt):
        sat = Satrec.twoline2rv(line1, line2)

        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second)

        error, position, velocity = sat.sgp4(jd, fr)

        if error != 0:
            return None

        x, y, z = position

        return {
            "x": x,
            "y": y,
            "z": z
        }

    # 🔥 INITIAL STATE
    def initialize(self):
        self.current_time = datetime.now(timezone.utc)

        # FIX: keep SAME satellite for RL
        self.agent_sat = random.choice(self.satellites)

        sat_pos = self.get_position(*self.agent_sat, self.current_time)

        # Handle None safely
        if sat_pos is None:
            sat_pos = {"x": 0, "y": 0, "z": 0}

        # Pick debris
        debris_sample = random.sample(self.debris, min(5, len(self.debris)))
        debris_positions = []

        for d in debris_sample:
            pos = self.get_position(*d, self.current_time)
            if pos:
                debris_positions.append(pos)

        return {
            "satellite": sat_pos,
            "debris": debris_positions
        }

    # 🔥 STEP UPDATE
       
    def update(self, action):

        # FIX: correct time progression
        self.current_time += timedelta(seconds=10)

        # FIX: same satellite (agent)
        sat_pos = self.get_position(*self.agent_sat, self.current_time)

        # Handle None safely
        if sat_pos is None:
            sat_pos = {"x": 0, "y": 0, "z": 0}

        # Get debris positions
        debris_positions = []
        for d in self.debris[:5]:
            pos = self.get_position(*d, self.current_time)
            if pos:
                debris_positions.append(pos)

        return {
            "satellite": sat_pos,
            "debris": debris_positions
        }