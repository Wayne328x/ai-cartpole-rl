
import os
import time

import gymnasium as gym
from stable_baselines3 import DQN


def main():
    model_path = os.path.join("models", "cartpole_dqn.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Train the model first using train_cartpole_dqn.py"
        )

    # Create environment with rendering
    env = gym.make("CartPole-v1", render_mode="human")

    # Load trained model
    model = DQN.load(model_path)

    # Run a few episodes
    for episode in range(5):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Use the trained model to predict the action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Slow down rendering slightly so it's easier to watch
            time.sleep(0.02)

        print(f"Episode {episode + 1} reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    main()
