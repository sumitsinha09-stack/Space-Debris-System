from env import SpaceDebrisEnv
import random

# Create environment
env = SpaceDebrisEnv(level="easy")

# Reset environment
state = env.reset()

print("🚀 Initial State:")
print(state)
print("\n--- Running Simulation ---\n")

# Run random actions
for step in range(30):
    action = random.randint(0, 4)

    state, reward, done, _ = env.step(action)

    print(f"Step {step+1}")
    print("Action:", action)
    print("State:", state)
    print("Reward:", reward)
    print()

    if done:
        print("❌ Episode ended (collision or fuel)")
        break