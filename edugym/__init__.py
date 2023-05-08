from gymnasium.envs.registration import register

register(
    id="edugym/Golf-v0",
    entry_point="edugym.envs:GolfEnv",
    max_episode_steps=300,
)

register(
    id="edugym/Roadrunner-v0",
    entry_point="edugym.envs:RoadrunnerEnv",
    max_episode_steps=300,
)

register(
     id="edugym/SuperMarket-v0",
     entry_point="edugym.envs:SupermarketEnv",
     max_episode_steps=300,
)
