import time
import gymnasium as gym


def main():
    # Create the CartPole-v1 environment with human rendering
    env = gym.make("CartPole-v1", render_mode="human")

    num_episodes = 5  # run 5 episodes before quitting

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

        print(f"Starting episode {episode + 1}")

        while not done:
            # Take a random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Slow down so you can actually watch it
            time.sleep(0.02)

        print(f"Episode {episode + 1} finished with reward: {episode_reward}")

    print("Finished all episodes. Press Enter to close...")
    input()

    env.close()


if __name__ == "__main__":
    main()