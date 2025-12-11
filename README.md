
# AI Learns to Play a Game — CartPole (RL Starter Project)

This is a starter codebase for a 3-person team project where an AI learns to play the the CartPole game using Reinforcement Learning (RL).

## Tech Stack

- Python 3.10+
- gymnasium for the CartPole environment
- stable-baselines3 for RL algorithms (DQN)
- PyTorch (as backend)
- TensorBoard for training visualization

## Folder Structure

```text
rl_cartpole_starter/
├── README.md
├── requirements.txt
├── test_env.py
├── train_cartpole_dqn.py
├── evaluate_cartpole_dqn.py
└── models/
```

- `test_env.py`: sanity check — run a random agent in CartPole.
- `train_cartpole_dqn.py`: trains a DQN agent on CartPole and logs to TensorBoard.
- `evaluate_cartpole_dqn.py`: loads a trained model and lets you watch it play.
- `models/`: directory where trained models will be saved.

## Installation

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Test the environment

```bash
python test_env.py
```

You should see a CartPole window where the pole moves randomly and eventually falls.

### 2. Train the DQN agent

```bash
python train_cartpole_dqn.py
```

This will:
- Train a DQN agent on CartPole.
- Save the model to `models/cartpole_dqn.zip`.
- Log training to `./tb_logs/` for TensorBoard.

To view TensorBoard:

```bash
tensorboard --logdir tb_logs
```

### 3. Watch the trained agent

After training finishes:

```bash
python evaluate_cartpole_dqn.py
```

You will see the trained agent trying to balance the pole.

## Suggested Team Roles

- **Member A (RL Engineer)**: focus on `train_cartpole_dqn.py`, try different algorithms (DQN, PPO), tweak network architecture.
- **Member B (Statistics / Analysis)**: analyze rewards, training curves (via TensorBoard), propose hyperparameter and reward adjustments.
- **Member C (Environment / Demo)**: extend environment, adjust observation/reward, and create a nice demo or recording of the agent.
# ai-cartpole-rl
