import gymnasium as gym
import edugym
env = gym.make("edugym/Roadrunner-v0")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    print(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()