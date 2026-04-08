from flask import Flask, jsonify
from env import SpaceEnv
from tasks import task_easy, task_medium, task_hard

app = Flask(__name__)
env = SpaceEnv()

@app.post("/reset")
def reset():
    env.reset()
    return jsonify({"status": "ok"})

@app.post("/step")
def step():
    scores = []
    scores.append(task_easy(env))
    scores.append(task_medium(env))
    scores.append(task_hard(env))
    return jsonify({
        "scores": scores,
        "average": sum(scores) / len(scores)
    })

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)