# AI Learns CartPole 🎮 — DQN Reinforcement Learning Project

This project trains an AI agent to play the classic **CartPole** game using **Deep Q-Networks (DQN)** with **PyTorch** and **Stable-Baselines3**.

The goal is to show an **end-to-end RL workflow**:
environment → training → evaluation → visualization (TensorBoard),
in a way that’s clean enough for a portfolio / resume project.

---

## ✨ Features

- ✅ Uses **Gymnasium** `CartPole-v1` environment  
- ✅ Implements **DQN** with Stable-Baselines3  
- ✅ Logs training metrics to **TensorBoard**  
- ✅ Includes scripts for:
  - random-agent sanity check  
  - training the agent  
  - evaluating a trained agent (visual demo)  
- ✅ Clean `.gitignore` (no virtualenv / logs in repo)

---

## 🧱 Tech Stack

- **Language:** Python 3.10+
- **RL Framework:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **DL Backend:** PyTorch
- **Environment:** [Gymnasium](https://gymnasium.farama.org/)
- **Visualization:** TensorBoard

---

## 📂 Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── test_env.py              # Random agent sanity check
├── train_cartpole_dqn.py    # Train DQN agent on CartPole
├── evaluate_cartpole_dqn.py # Load and run trained agent
├── models/                  # (local) saved models - ignored by git
└── tb_logs/                 # (local) TensorBoard logs - ignored by git
