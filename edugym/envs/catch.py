# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Catch reinforcement learning environment."""

import gymnasium as gym
import numpy as np
import pygame
import sys
import os
def is_notebook():
    try:
        get_ipython
        return True
    except NameError:
        return False

_ACTIONS = (-1, 0, 1)  # Left, no-op, right.
ACTION_LEFT = -1
ACTION_NOOP = 0
ACTION_RIGHT = 1

OBSERVATION_TYPE_DEFAULT = 0
OBSERVATION_TYPE_MINIMAL = 1
OBSERVATION_TYPE_DEFAULT_WITH_COLORS = 2
OBSERVATION_TYPE_DEFAULT_ONEHOT = 3
OBSERVATION_TYPES = [
    OBSERVATION_TYPE_DEFAULT,
    OBSERVATION_TYPE_MINIMAL,
    OBSERVATION_TYPE_DEFAULT_WITH_COLORS,
    OBSERVATION_TYPE_DEFAULT_ONEHOT,
]
OBSERVATION_TYPE_3D = 4
OBSERVATION_TYPE_3D_GRID = 5

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
grey = (128, 128, 128)
red = (255, 0, 0)
yellow = (255, 255, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


class Catch(gym.Env):
    metadata = {
        "render_modes": ["inline", "terminal", "graphic", "notebook"],
        "render_fps": 50,
    }
    """A Catch environment built on the dm_env.Environment class.

  The agent must move a paddle to intercept falling balls. Falling balls only
  move downwards on the column they are in.

  The observation is an array shape (rows, columns), with binary values:
  zero if a space is empty; 1 if it contains the paddle or a ball.

  The actions are discrete, and by default there are three available:
  stay, move left, and move right.

  The episode terminates when the ball reaches the bottom of the screen.
  """

    def __init__(self, rows: int = 10, columns: int = 5, seed=None, observation_type=0, render_mode="inline"):
        """Initializes a new Catch environment.

        Args:
          rows: number of rows.
          columns: number of columns.
        """
        super().reset(seed=seed)
        self._rows = rows
        self._columns = columns
        self.width = columns
        self.height = rows
        self._board = np.zeros((rows, columns), dtype=np.float32)
        self._ball_x = None
        self._ball_y = None
        self._paddle_x = None
        self._paddle_y = None
        self._reset_next_step = True
        self._total_regret = 0.0
        self.observation_type = observation_type
        self.observation_space = self.observation_spec(observation_type)
        self.action_space = self.action_spec()
        # Rendering
        self.pygame_initialized = False
        self.render_mode = render_mode

    def reset(self, seed=None):
        """Returns the first `TimeStep` of a new episode."""
        super().reset(seed=seed)
        self._reset_next_step = False
        self._ball_x = self.np_random.integers(self._columns, size=1)[0]
        self._ball_y = 0
        self._paddle_x = self._columns // 2
        self._paddle_y = self._rows - 1
        self.observation_space = self.observation_spec(self.observation_type)
        self.action_space = self.action_spec()
        return self._observation()

    def step(self, action: int):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        # Move the paddle.
        dx = _ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

        # Drop the ball.
        self._ball_y += 1

        # Check for termination.
        if self._ball_y == self._paddle_y:
            reward = 1.0 if self._paddle_x == self._ball_x else -1.0
            self._reset_next_step = True
            self._total_regret += 1.0 - reward
            return self._observation(), reward, True, False, {}

        return self._observation(), 0.0, False, False, {}

    def observation_spec(self, observation_type):
        """Returns the observation spec."""
        if observation_type == OBSERVATION_TYPE_MINIMAL:
            highest_num = max(self._columns, self._rows)
            return gym.spaces.Box(
                low=-highest_num, high=highest_num, shape=(4,), dtype=np.int32
            )
        if observation_type == OBSERVATION_TYPE_DEFAULT_WITH_COLORS:
            return gym.spaces.Box(
                low=0.0,
                high=255.0,
                shape=(self._rows, self._columns, 3),
                dtype=np.float32,
            )
        if observation_type == OBSERVATION_TYPE_DEFAULT_ONEHOT:
            return gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._rows, self._columns, 2), dtype=np.int32
            )
        if observation_type == 4:
            return gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._rows, self._columns, self._rows),
                dtype=np.float32,
            )
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=self._board.shape, dtype=np.bool_
        )

    def action_spec(self):
        """Returns the action spec."""
        return gym.spaces.Discrete(len(_ACTIONS))

    def _observation(self):
        if self.observation_type == OBSERVATION_TYPE_MINIMAL:
            return np.array(
                [self._paddle_x, self._paddle_y, self._ball_x, self._ball_y]
            )
        if self.observation_type == OBSERVATION_TYPE_DEFAULT_WITH_COLORS:
            a = np.zeros((self._rows, self._columns, 3))
            a[self._ball_y, self._ball_x, 0] = 128.0
            a[self._ball_y, self._ball_x, 1] = 128.0
            a[self._ball_y, self._ball_x, 2] = 128.0
            a[self._paddle_y, self._paddle_x, 0] = 255.0
            a[self._paddle_y, self._paddle_x, 1] = 255.0
            a[self._paddle_y, self._paddle_x, 2] = 255
            return a
        if self.observation_type == OBSERVATION_TYPE_DEFAULT_ONEHOT:
            a = np.zeros((self._rows, self._columns, 2))
            a[self._ball_y, self._ball_x, 0] = 1.0
            a[self._paddle_y, self._paddle_x, 1] = 1.0
            return a
        self._board.fill(0.0)
        self._board[self._ball_y, self._ball_x] = 1.0
        self._board[self._paddle_y, self._paddle_x] = 1.0
        return self._board.copy()

    def render(self):
        mode = self.render_mode
        if mode == "graphic" or mode == "notebook":
            # Initialize pygame
            if not self.pygame_initialized:
                pygame.init()
                self.cell_size = 50
                screen_width, screen_height = (
                    self.width * self.cell_size,
                    (self.height) * self.cell_size,
                )
                self.screen = pygame.display.set_mode([screen_width, screen_height])
                pygame.display.set_caption("Catch Environment")
                self.pygame_initialized = True

            # Set background color
            self.screen.fill(black)
            grid_width = 5
            # Draw white bg
            pygame.draw.rect(self.screen, white, pygame.Rect(
                grid_width,
                grid_width,
                screen_width - 10,
                screen_height - 10,
            ))


            # Draw Ball
            pygame.draw.rect(self.screen, yellow, pygame.Rect(
                (self._ball_x * self.cell_size) + grid_width,
                (max(0, (self._ball_y)) * self.cell_size) + grid_width,
                self.cell_size - grid_width*2,
                self.cell_size - grid_width*2,
            ))
            # Draw Paddle
            pygame.draw.rect(self.screen, green, pygame.Rect(
                (self._paddle_x * self.cell_size) + grid_width,
                ((self._paddle_y) * self.cell_size + (self.cell_size/2)) + grid_width,
                self.cell_size  - grid_width*2,
                (self.cell_size / 2) - grid_width*2,
            ))
            
            # Flip the display
            pygame.display.flip()
            if is_notebook():
                from IPython.display import Image, display
                pygame.image.save(self.screen, 'frame.png')
                display(Image(filename='frame.png'))
            else:
                # Wait for a short time to slow down the rendering
                pygame.time.wait(25)
        else:
            render_grid = self._observation()
            print("\n")
            for y, row_data in enumerate(render_grid):
                row = ""
                for x, cell_entry in enumerate(row_data):
                    if cell_entry == 1:
                        if y == 9:
                            row += "_"
                        else:
                            row += "O"
                    else:
                        row += " "
                print(row)
            print("\n")


gym.register(
    id="Catch-v0",
    entry_point="catch:Catch",
    kwargs={},
)

gym.register(
    id="Catch-vectorized-v0",
    entry_point="catch:Catch",
    kwargs={"observation_type": OBSERVATION_TYPE_MINIMAL},
)

gym.register(
    id="Catch-color-v0",
    entry_point="catch:Catch",
    kwargs={"observation_type": OBSERVATION_TYPE_DEFAULT_WITH_COLORS},
)

gym.register(
    id="Catch-onehot-v0",
    entry_point="catch:Catch",
    kwargs={"observation_type": OBSERVATION_TYPE_DEFAULT_ONEHOT},
)

for size in range(2, 9):
    gym.register(
        id=f"Catch-{size}-v0",
        entry_point="catch:Catch",
        kwargs={"rows": 10 * size, "columns": 5 * size},
    )
    gym.register(
        id=f"Catch-{size}-vectorized-v0",
        entry_point="catch:Catch",
        kwargs={
            "observation_type": OBSERVATION_TYPE_MINIMAL,
            "rows": 10 * size,
            "columns": 5 * size,
        },
    )
    gym.register(
        id=f"Catch-{size}-color-v0",
        entry_point="catch:Catch",
        kwargs={
            "observation_type": OBSERVATION_TYPE_DEFAULT_WITH_COLORS,
            "rows": 10 * size,
            "columns": 5 * size,
        },
    )
    gym.register(
        id=f"Catch-{size}-onehot-v0",
        entry_point="catch:Catch",
        kwargs={
            "observation_type": OBSERVATION_TYPE_DEFAULT_ONEHOT,
            "rows": 10 * size,
            "columns": 5 * size,
        },
    )


for size in range(2, 21):
    gym.register(
        id=f"Catch-{size}x{size}-v0",
        entry_point="catch:Catch",
        kwargs={"rows": size, "columns": size},
    )
    gym.register(
        id=f"Catch-{size}x{size}-vectorized-v0",
        entry_point="catch:Catch",
        kwargs={
            "observation_type": OBSERVATION_TYPE_MINIMAL,
            "rows": size,
            "columns": size,
        },
    )
    gym.register(
        id=f"Catch-{size}x{size}-color-v0",
        entry_point="catch:Catch",
        kwargs={
            "observation_type": OBSERVATION_TYPE_DEFAULT_WITH_COLORS,
            "rows": size,
            "columns": size,
        },
    )
    gym.register(
        id=f"Catch-{size}x{size}-onehot-v0",
        entry_point="catch:Catch",
        kwargs={
            "observation_type": OBSERVATION_TYPE_DEFAULT_ONEHOT,
            "rows": size,
            "columns": size,
        },
    )