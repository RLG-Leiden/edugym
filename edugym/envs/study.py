from __future__ import annotations

from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np


class Study(gym.Env):
    def __init__(self, total_days: int = 18, n_actions: int = 5,  seed: int = 42):
        assert n_actions >= 3, "There must be at least the actions 'study', 'sleep', 'go_out'"  # or one if we only have
        # the study action
        assert total_days >= 9, "At least 4 lectures should be able to take place"

        self.knowledge = 0
        # self.energy = 0
        self.total_days = total_days
        # which of the days have a lecture
        self.lecture_days = [1 if i in [0, 2, 7, 9] else 0 for i in range(total_days)]
        # day counter
        self.current_day = 0
        # what actions have been done so far. Needed for rendering
        self.action_history = [None for _ in range(total_days)]

        self.action_space = Discrete(n_actions, seed=seed)
        self.observation_space = MultiDiscrete([5, total_days])  # [knowledge, current_day]
        # self.observation_space = MultiDiscrete([5, 5, total_days])  # if we also use energy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.knowledge = 0
        # self.energy = 0
        self.current_day = 0
        self.action_history = [None for _ in range(self.total_days)]

        return np.array([0, 0]), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        terminated = False
        # update action history
        self.action_history[self.current_day] = action
        # increase knowledge if you study on a lecture day
        if action == 0 and self.lecture_days[self.current_day] == 1:
            self.knowledge += 1

        # Optional part:
        # elif action == 1:  # Sleep
        #     self.energy += 1
        #
        # elif action == 2:  # go out
        #     self.energy -= 1

        # calculate reward
        reward = 0
        if self.current_day == self.total_days-1:  # Exam day!
            terminated = True
            # we must have attended at least 3 lectures to pass the exam
            if self.knowledge >= 3:  # and self.energy > 3
                reward = 1

        self.current_day += 1

        self.render()

        return np.array([self.knowledge, self.current_day]), reward, terminated, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        for i in range(self.total_days):  # print lecture schedule
            if self.lecture_days[i] == 1:
                print('L', end=' ')  # lecture
            else:
                print('-', end=' ')  # anything else
        print('E')  # Exam

        for i in range(self.total_days):
            if self.action_history[i] == 0:  # Studied S
                print('S', end=' ')
            # elif self.action_history[i] == 1:  # Sleep ZzZ...
            #     print('Z', end=' ')
            # elif self.action_history[i] == 2:  # out O
            #     print('O', end=' ')
            elif self.action_history[i] == None:  # no action taken yet
                print('-', end=' ')
            else:
                print('*', end=' ')  # any other action was taken
        print()

