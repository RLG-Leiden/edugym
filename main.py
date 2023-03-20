import gymnasium as gym
import edugym
envs = []
envs.append(gym.make("edugym/Study-v0", total_days=21, n_actions=5))
envs.append(gym.make("edugym/Roadrunner-v0", render_mode="terminal"))
env_to_test = 0

observation, info = envs[env_to_test].reset(seed=42)
for i in range(1000):
    print(f"Step: {i}")
    action = envs[env_to_test].action_space.sample()
    observation, reward, terminated, truncated, info = envs[env_to_test].step(action)

    if terminated or truncated:
        observation, info = envs[env_to_test].reset(seed=42)
envs[env_to_test].close()
