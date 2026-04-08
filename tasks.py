def task_easy(env):
    env.reset()
    _, reward, _, _ = env.step("monitor")
    return reward


def task_medium(env):
    env.reset()
    _, reward, _, _ = env.step("avoid_collision")
    return reward


def task_hard(env):
    env.reset()

    total_reward = 0
    for _ in range(5):
        _, reward, _, _ = env.step("avoid_collision")
        total_reward += reward

    return total_reward / 5