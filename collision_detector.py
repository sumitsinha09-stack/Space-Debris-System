import math

# Distance thresholds
CRITICAL_DISTANCE_KM  = 1
WARNING_DISTANCE_KM   = 10
MONITOR_DISTANCE_KM   = 50


def calculate_distance(obj1, obj2):
    dx = obj1["x"] - obj2["x"]
    dy = obj1["y"] - obj2["y"]
    dz = obj1["z"] - obj2["z"]
    return math.sqrt(dx**2 + dy**2 + dz**2)


# 🔥 MAIN FUNCTION FOR RL
def detect_collision(satellite, debris_list):
    """
    Used inside env.step()
    Returns:
        collision (bool)
        min_distance (float)
        risk_level (str)
    """

    if satellite is None or not debris_list:
        return False, float("inf"), "SAFE"

    min_dist = float("inf")
    risk_level = "SAFE"

    for debris in debris_list:
        dist = calculate_distance(satellite, debris)

        if dist < min_dist:
            min_dist = dist

        # Determine risk
        if dist < CRITICAL_DISTANCE_KM:
            return True, dist, "CRITICAL"

        elif dist < WARNING_DISTANCE_KM:
            risk_level = "WARNING"

        elif dist < MONITOR_DISTANCE_KM:
            if risk_level != "WARNING":
                risk_level = "MONITOR"

    return False, min_dist, risk_level