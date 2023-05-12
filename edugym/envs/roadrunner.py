import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
MAX_SPEED = 10


class RoadrunnerEnv(gym.Env):
    metadata = {"render_modes": ["terminal", "pygame", "none"]}

    def __init__(self, render_mode="terminal", size=10, negative_reward_size=-100, max_episode_steps=100):
        self.size = size  # The size of the single dimension grid

        # Observations are the agent's location in the grid and its speed
        # We can represent this as a single integer by taking their Cartesian product
        self.observation_space = spaces.Discrete(self.size * (MAX_SPEED + 1))

        # We have 3 actions, corresponding to "speed up", "slow down", "idle"
        self.action_space = spaces.Discrete(3)


        # The following dictionary maps abstract actions from
        # `self.action_space` to the direction we will walk in if that action
        # is taken. I.e. 0 corresponds to "right", 1 to "up" etc.
        self._action_to_speed = {
            0: np.array([0, 1]),  # speed up
            1: np.array([0, -1]),  # slow down
            2: np.array([0, 0]),  # idle
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "pygame":
            self.init_pygame()
        self.negative_reward_size = negative_reward_size
        self.max_episode_steps = max_episode_steps
        self._step_counter = 0

    def init_pygame(self):
        """
        Initialize the pygame screen and font.
        """
        self.grid_size = 50
        self.pg_width = self.size * self.grid_size
        self.pg_height = self.grid_size * 3
        self.screen = pygame.display.set_mode((self.pg_width, self.pg_height))
        pygame.display.set_caption("RoadrunnerEnv")
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)

    def draw_grid(self):
        """
        Draw the game grid on then pygame screen.

        The grid is a 1-D line with the agent, target and wall locations marked.
        """
        for i in range(self.size):
            rect = pygame.Rect(i * self.grid_size, 0, self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, WHITE, rect, 2)
            if i == self._agent_location[0]:
                pygame.draw.rect(self.screen, RED, rect)
                text = self.font.render("A", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            elif i == self._target_location[0]:
                pygame.draw.rect(self.screen, WHITE, rect)
                text = self.font.render("T", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            elif i == self._wall_location[0]:
                pygame.draw.rect(self.screen, WHITE, rect)
                text = self.font.render("W", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            else:
                pygame.draw.rect(self.screen, WHITE, rect)

    def _get_obs(self):
        """
        Get the current state, represented as an integer (see `self.observation_space`)
        """
        return (self._agent_location[0] * MAX_SPEED) + self._agent_location[1]

    def _get_info(self):
        """
        Get additional information about the current state.
        """
        return {"target": self._target_location, "wall": self._wall_location, "steps": self._step_counter}

    def _render_frame(self):
        """
        Print the current state to the terminal or pygame screen.
        """
        if self.render_mode == "terminal":
            for i in range(self.size):
                if i == self._agent_location[0]:
                    print("=A=", end=" ")
                elif i == self._target_location[0]:
                    print("=T=", end=" ")
                elif i == self._wall_location[0]:
                    print("=W=", end=" ")
                else:
                    print("= =", end=" ")
            print()
        elif self.render_mode == "pygame":
            self.screen.fill(BLACK)
            self.draw_grid()
            pygame.display.update()
            self.clock.tick(15)
        elif self.render_mode == "none":
            pass
        else:
            raise NotImplementedError

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a random initial state.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0, 0], dtype=int)
        self._target_location = np.array([self.size - 2, 0], dtype=int)
        self._wall_location = np.array([self.size - 1, 0], dtype=int)
        self._step_counter = 0

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def _compute_intermediate_reward(self):
        """
        Compute the intermediate reward for the current state where nothing happened (agent moved).
        """
        return -1

    def step(self, action):
        """
        Step through the environment by taking `action` and returning the next state, reward, and
        whether the episode is done.
        """
        # Map the action (element of {0,1,2}) to agent location
        action = self._action_to_speed[action]
        # First update location with current speed
        new_x = self._agent_location[0] + self._agent_location[1]
        # then update speed with action
        new_dx = self._agent_location[1] + action[1]

        self._step_counter += 1

        # An episode is done iff the agent has reached the target OR the agent has reached the wall
        if new_x >= self._target_location[0] and new_x <= self._wall_location[0]:
            terminated = True
            reward = 1
        elif new_x >= self._wall_location[0]:
            terminated = True
            reward = self.negative_reward_size
        elif new_dx <= 0:
            terminated = True
            reward = self.negative_reward_size
        else:
            terminated = False
            reward = self._compute_intermediate_reward()

        if self._step_counter > self.max_episode_steps:
          truncated = True
        else:
          truncated = False

        # Update the agent's location
        if new_dx >= MAX_SPEED:
            new_dx = MAX_SPEED
        if new_x >= self.size - 1:
            new_x = self.size - 1
        self._agent_location = np.array([new_x, new_dx], dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, reward, terminated, truncated, info
