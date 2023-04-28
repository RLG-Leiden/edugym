import gymnasium as gym
import edugym
import argparse

def run_env(args):
    env = gym.make(f"edugym/{args.env}", render_mode=args.render)
    observation, info = env.reset(seed=42)
    for i in range(2):
        print(f"Step: {i}")
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=42)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test edugym")
    parser.add_argument("--env", type=str, default="Roadrunner-v0")
    parser.add_argument("--render", type=str, default="terminal")
    args = parser.parse_args()
    run_env(args)

