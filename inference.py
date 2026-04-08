import os
import asyncio
from openai import OpenAI
from env import SpaceEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "space-debris")
BENCHMARK = "space_debris_system"
MAX_STEPS = 8

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action(client, step, state):
    prompt = f"""
You are controlling a space debris management system.
Current state:
- Satellites: {state['satellites']}
- Debris: {state['debris']}
- Conjunctions (collision risks): {state['conjunctions']}
- Timestamp: {state['timestamp']}

Choose exactly one action: monitor or avoid_collision
Reply with only the action name, nothing else.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a space debris management AI. Reply with only: monitor or avoid_collision"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.3,
        )
        action = completion.choices[0].message.content.strip().lower()
        if action not in ["monitor", "avoid_collision"]:
            action = "avoid_collision"
        return action
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return "avoid_collision"

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SpaceEnv()

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, step, state)
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()