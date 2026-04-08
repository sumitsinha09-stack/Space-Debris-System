import random
from datetime import datetime, timezone

class SpaceEnv:
    def __init__(self):
        self.state_data = None

    def reset(self):
        self.state_data = {
            "satellites": random.randint(80, 120),
            "debris": random.randint(300, 600),
            "conjunctions": random.randint(0, 100),
            "timestamp": str(datetime.now(timezone.utc))
        }
        return self.state_data

    def step(self, action):
        reward = 0.0
        done = False

        if action == "monitor":
            reward = 0.2

        elif action == "avoid_collision":
            if self.state_data["conjunctions"] > 20:
                reward = 1.0
            else:
                reward = 0.5

        else:
            reward = 0.0

        return self.state_data, reward, done, {}

    def state(self):
        return self.state_data