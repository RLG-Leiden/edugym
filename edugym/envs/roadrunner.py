import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


class RoadrunnerEnv(gym.Env):
    metadata = {"render_modes": ["terminal", "pygame"]}

    def __init__(self, render_mode=None, size=10, discrete=True):
        self.size = size  # The size of the single dimension grid

        # Observations are dictionaries with the agent's location along a 1-D axis and speed.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0, 0], dtype=np.float64),
                    high=np.array([self.size - 1, self.size - 1], dtype=np.float64),
                    dtype=np.float64,
                ),
            }
        )

        # We have 3 actions, corresponding to "speed up", "slow down", "idle"
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_speed = {
            0: np.array([0, 1]),  # speed up
            1: np.array([0, -1]),  # slow down
            2: np.array([0, 0]),  # idle
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "pygame":
            self.init_pygame()

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
        return {"agent": self._agent_location}

    def _get_info(self):
        return {"target": self._target_location, "wall": self._wall_location}

    def _render_frame(self):
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
            self.clock.tick(30)
        else:
            raise NotImplementedError

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0, 0], dtype=np.float64)
        self._target_location = np.array([self.size - 2, 0], dtype=np.float64)
        self._wall_location = np.array([self.size - 1, 0], dtype=np.float64)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "teminal":
            self._render_frame()

        return observation, info

    def _compute_intermediate_reward(self):
        return 0

    def step(self, action):
        # Map the action (element of {0,1,2}) to agent location
        action = self._action_to_speed[action]
        # First update location with current speed
        new_x = self._agent_location[0] + self._agent_location[1]
        # then update speed with action
        new_dx = self._agent_location[1] + action[1]

        # Update the agent's location
        self._agent_location = np.array([new_x, new_dx], dtype=np.float64)

        # An episode is done iff the agent has reached the target OR the agent has reached the wall
        if (
            self._agent_location[0] >= self._target_location[0]
            and self._agent_location[0] <= self._wall_location[0]
        ):
            print("REACHED THE TARGET")
            terminated = True
            reward = 1
        elif self._agent_location[0] >= self._wall_location[0]:
            print("FELL OFF THE EDGE")
            terminated = True
            reward = -100
        elif self._agent_location[1] <= 0:
            print("SLOWED DOWN TOO MUCH")
            terminated = True
            reward = -100
        else:
            terminated = False
            reward = self._compute_intermediate_reward()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "terminal":
            self._render_frame()

        return observation, reward, terminated, False, info
