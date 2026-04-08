from env import SpaceEnv
from tasks import task_easy, task_medium, task_hard

def run():
    env = SpaceEnv()

    print("Running baseline inference...\n")

    scores = []
    scores.append(task_easy(env))
    scores.append(task_medium(env))
    scores.append(task_hard(env))

    print("Task Scores:", scores)
    print("Average Score:", sum(scores) / len(scores))


if __name__ == "__main__":
    run()