import unittest
import gymnasium as gym
import catch
from boulder import BoulderEnv
from golf import GolfEnv
from memorycorridor import MemoryCorridorEnv
from roadrunner import RoadrunnerEnv
from supermarket import SupermarketEnv
from study import Study
from trashbot import TrashBotDiscreteEnv
from tamagotchi import TamagotchiEnv
import numpy as np

class TestSeeding(unittest.TestCase):
  def test_seeding_works_uniformly(self):
    envs_to_test = [
      (lambda: TamagotchiEnv(), True, True, 0),
      (lambda: TrashBotDiscreteEnv(), True, True, [1, 1]),
      (lambda: Study(), True, True, 0),
      (lambda: SupermarketEnv(), True, False, 0),
      (lambda: RoadrunnerEnv(), True, True, 0),
      (lambda: MemoryCorridorEnv(), False, True, 0),
      (lambda: GolfEnv(), True, True, 0),
      (lambda: BoulderEnv(), True, True, 0),
      (lambda: gym.make("Catch-v0"), False, True, 0),
    ]
    for env_creator, third_is_equal, ends_by_itself, action in envs_to_test:
      env = env_creator()
      env2 = env_creator()
      env3 = env_creator()
      env.reset(seed=42)
      env2.reset(seed=42)
      env3.reset(seed=41)
      done = False
      truncated = False
      steps = 0
      while (ends_by_itself and not done and not truncated) or (not ends_by_itself and steps < 100):
        s_next, r, done, truncated, _ = env.step(action)
        s_next2, r2, done2, _, _ = env2.step(action)
        assert(np.array_equal(s_next, s_next2))
        assert(r == r2)
        assert(done == done2)
        s_next3, r3, done3, _, _ = env3.step(action)
        if not third_is_equal:
          assert(not np.array_equal(s_next, s_next3))
        steps += 1

    pass

if __name__ == '__main__':
    unittest.main()