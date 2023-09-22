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


colab_rendering = "google.colab" in sys.modules
if colab_rendering or is_notebook():
    # set SDL to use the dummy NULL video driver,
    #   so it doesn't need a windowing system.
    from google.colab import output

    os.environ["SDL_VIDEODRIVER"] = "dummy"

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
light_blue = (154, 187, 255)
dark_grey = (110, 110, 110)
ligth_grey = (235, 235, 235)
dark_blue = (35, 110, 150)


class Catch(gym.Env):
    metadata = {
        "render_modes": ["terminal", "graphic", "none"],
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

    def __init__(
        self,
        rows: int = 10,
        columns: int = 5,
        seed=None,
        observation_type=0,
        render_mode="inline",
    ):
        """Initializes a new Catch environment.

        Args:
          rows: number of rows.
          columns: number of columns.
          seed: random seed for the RNG.
        """
        super().__init__()
        self.seed(seed)

        self._rows = rows
        self._columns = columns
        self.width = columns
        self.height = rows
        self._rng = np.random.RandomState(seed)
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
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.pygame_initialized = False
        self.render_mode = render_mode

    def seed(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._ball_x = self._rng.randint(self._columns)
        self._ball_y = 0
        self._paddle_x = self._columns // 2
        self._paddle_y = self._rows - 1
        self.observation_space = self.observation_spec(self.observation_type)
        self.action_space = self.action_spec()
        return self._observation()

    def step(self, action: int):
        """Updates the environment according to the action."""
        # if self._reset_next_step:
        #    return self.reset()

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
        if self.render_mode == "graphic":
            # Initialize pygame
            if not self.pygame_initialized:
                pygame.init()
                self.cell_size = 50
                self.screen_width, self.screen_height = (
                    self.width * self.cell_size,
                    (self.height) * self.cell_size,
                )
                self.screen = pygame.display.set_mode(
                    [self.screen_width, self.screen_height]
                )
                pygame.display.set_caption("Catch Environment")
                self.pygame_initialized = True

            # Set background color
            self.screen.fill(dark_blue)
            grid_width = 5
            # Draw white bg
            pygame.draw.rect(
                self.screen,
                ligth_grey,
                pygame.Rect(
                    grid_width,
                    grid_width,
                    self.screen_width - 10,
                    self.screen_height - 10,
                ),
            )

            # Draw Ball
            blue = (35, 110, 150)
            yellow = (243, 188, 87)
            red = (156, 39, 6)
            grey = (235, 235, 235)

            light_yellow = (243, 188, 87)
            ball_radius = (self.cell_size - grid_width * 4) // 2
            pygame.draw.circle(
                self.screen,
                light_yellow,
                (
                    (self._ball_x * self.cell_size) + grid_width * 2 + ball_radius,
                    (max(0, (self._ball_y)) * self.cell_size)
                    + grid_width * 3
                    + ball_radius,
                ),
                ball_radius,
            )
            # Draw Paddle
            # pygame.draw.rect(self.screen, green, pygame.Rect(
            #     (self._paddle_x * self.cell_size) + grid_width,
            #     ((self._paddle_y) * self.cell_size + (self.cell_size/2)) + grid_width,
            #     self.cell_size  - grid_width*2,
            #     (self.cell_size / 2) - grid_width*2,
            # ))
            paddle_color = (156, 39, 6)
            x_pos = (self._paddle_x * self.cell_size) + grid_width
            y_pos = (
                (self._paddle_y) * self.cell_size + (self.cell_size / 2)
            ) + grid_width
            width = self.cell_size - grid_width * 2
            height = (self.cell_size / 2) - grid_width * 2
            pygame.draw.polygon(
                self.screen,
                paddle_color,
                [
                    (x_pos, y_pos),
                    (x_pos + width / 10, y_pos),
                    (x_pos + width / 10, y_pos + (height - (height / 10))),
                    (x_pos + (width - width / 10), y_pos + (height - (height / 10))),
                    (x_pos + (width - width / 10), y_pos),
                    (x_pos + width, y_pos),
                    (x_pos + width, y_pos + height),
                    (x_pos, y_pos + height),
                    (x_pos, y_pos),
                ],
                5,
            )

            # Flip the display
            pygame.display.flip()
            if is_notebook():
                from IPython.display import Image, display

                output.clear()
                pygame.image.save(self.screen, "frame.png")
                display(Image(filename="frame.png"))
            else:
                # Wait for a short time to slow down the rendering
                pygame.time.wait(25)
        elif self.render_mode == "terminal":
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
        elif self.render_mode == "none":
            pass
        else:
            raise NotImplementedError


def test():
    render_mode = "graphic"  # 'terminal', 'none'
    # Initialize the environment
    from edugym.envs.interactive import play_env, play_env_terminal

    env = Catch(render_mode=render_mode)
    if render_mode == "graphic":
        play_env(
            env,
            "s=stay, a=left, d=right",
            {pygame.K_a: 0, pygame.K_s: 1, pygame.K_d: 2},
        )
    elif render_mode == "terminal":
        play_env_terminal(env, "s=stay, a=left, d=right", {"a": 0, "s": 1, "d": 2})


if __name__ == "__main__":
    test()
