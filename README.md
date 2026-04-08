---
title: Space Debris System
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🚀 Space Debris Collision Prediction System

This project simulates space debris tracking and predicts potential collisions using:

- Orbital mechanics (TLE data)
- Monte Carlo simulation
- Machine Learning (with fallback support)

## 🔧 Features

- Real-time orbit tracking
- Conjunction detection (< 500 km)
- Collision probability estimation
- Risk classification (CRITICAL → SAFE)
- Interactive dashboard

## 🧠 Tech Stack

- Python (Flask)
- SGP4 orbital propagation
- NumPy, Pandas
- Docker deployment
- HuggingFace Spaces

## 🌐 How it Works

1. Load satellite + debris TLE data  
2. Compute positions  
3. Detect close approaches  
4. Run ML / fallback prediction  
5. Display results on dashboard  

## ⚠️ Note

If ML models are unavailable, system uses fallback logic for risk estimation.

## 👨‍💻 Author

Krish