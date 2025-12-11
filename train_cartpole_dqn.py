
import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


def main():
    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Create CartPole environment
    env = gym.make("CartPole-v1")
    env = Monitor(env)

    # Configure TensorBoard logging
    log_dir = "./tb_logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Define the DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1_000,
        train_freq=4,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Attach custom logger
    model.set_logger(new_logger)

    # Train the agent
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the trained model
    model_path = os.path.join("models", "cartpole_dqn")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
