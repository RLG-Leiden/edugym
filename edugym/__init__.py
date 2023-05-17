from gymnasium.envs.registration import register


register(
    id="edugym/Boulder-v0",
    entry_point="edugym.envs:BoulderEnv"
)
register(
    id="edugym/Catch-v0",
    entry_point="edugym.envs:Catch"
)

register(
    id="edugym/Golf-v0",
    entry_point="edugym.envs:GolfEnv"
)

register(
    id="edugym/MemoryCorridor-v0",
    entry_point="edugym.envs:MemoryCorridorEnv"
)

register(
    id="edugym/Roadrunner-v0",
    entry_point="edugym.envs:RoadrunnerEnv",
    max_episode_steps=300,
)

register(
    id="edugym/Study-v0",
    entry_point="edugym.envs:Study"
)

register(
    id="edugym/SuperMarket-v0",
    entry_point="edugym.envs:SupermarketEnv"
)
register(
    id="edugym/Tamagotchi-v0",
    entry_point="edugym.envs:TamagotchiEnv"
)

register(
    id="edugym/TrashBotDiscrete-v0",
    entry_point="edugym.envs:TrashBotDiscreteEnv"
)

register(
    id="edugym/TrashBotContinuous-v0",
    entry_point="edugym.envs:TrashBotContinuousEnv"
)

