🚀 SDCPS — Space Debris Collision Prevention System

🌍 Overview

SDCPS is a real-world simulation system that tracks satellites and space debris in Earth orbit and predicts potential collisions using Machine Learning and Reinforcement Learning.

This project mimics a mission control dashboard used in space agencies like NASA and ESA.

⸻

🎯 Problem Statement

Space debris is increasing rapidly and poses a serious threat to active satellites.
This project aims to:
	•	Track satellites & debris in real-time
	•	Detect close approaches (conjunctions)
	•	Predict collision risks
	•	Enable intelligent avoidance using RL

⸻

⚙️ Tech Stack
	•	Python (Flask Backend)
	•	SGP4 (Orbital Mechanics)
	•	Machine Learning (XGBoost)
	•	Reinforcement Learning (Custom Environment)
	•	HTML, CSS, JavaScript (Frontend Visualization)

⸻

🧠 System Components

1. Orbital Simulation
	•	Uses real TLE data
	•	Calculates satellite positions in 3D space

2. Collision Detection
	•	Computes distance between objects
	•	Flags risky proximity

3. Machine Learning
	•	Predicts:
	•	Collision probability
	•	Risk level (LOW, MEDIUM, HIGH, CRITICAL)

4. Reinforcement Learning Environment
	•	Agent = Satellite
	•	Goal = Avoid debris
	•	Actions:
	•	Move up/down/left/right
	•	Reward system:
	•	Safe → +0.5
	•	Risk → penalty
	•	Collision → negative reward

5. Live Dashboard
	•	Displays Earth with satellites
	•	Shows:
	•	Tracked objects
	•	Conjunctions
	•	ML risk predictions

⸻

🔁 OpenEnv API
	•	reset() → Initialize environment
	•	step(action) → Apply movement
	•	state() → Get current state

⸻

🎮 Difficulty Levels
	•	EASY → Few debris
	•	MEDIUM → Moderate traffic
	•	HARD → Dense debris + high risk
