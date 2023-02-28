import gymnasium as gym
import edugym
env = gym.make("edugym/Roadrunner-v0", render_mode="terminal")

observation, info = env.reset(seed=42)
for i in range(1000):
    print(f"Step: {i}")
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)
env.close()