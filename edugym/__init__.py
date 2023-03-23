from gymnasium.envs.registration import register

register(
     id="edugym/Roadrunner-v0",
     entry_point="edugym.envs:RoadrunnerEnv",
     max_episode_steps=300,
)

register(
     id="edugym/Study-v0",
     entry_point="edugym.envs:Study",
)

