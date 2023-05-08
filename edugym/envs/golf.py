import numpy as np

import gymnasium as gym
from gymnasium import spaces


class GolfEnv(gym.Env):
    metadata = {"render_modes": {"terminal"}}

    ascii_green = [
        ".....................",
        ".....................",
        ".....................",
        "......  <<|    ......",
        ".....     |     .....",
        "......         ......",
        ".....................",
        ".....................",
        ".....................",
    ]

    def __init__(self, render_mode=None, length=4, discrete=True):
        self.length = length  # The length of the 2D golf course
        self.discrete = discrete

        # Observations are dictionaries with the ball and green location.
        self.observation_space = spaces.Dict(
            {
                "ball": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([2, self.length + 1], dtype=np.float32),
                    dtype=np.float32,
                ),
                "green": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([2, self.length], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        if discrete:
            # We have 3 discrete actions, corresponding to the power of a swing: "soft", "medium", "hard"
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=np.float32(0), high=np.float32(3), dtype=np.float32
            )

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_speed = {
            0: np.array([1], dtype=np.float32),  # soft
            1: np.array([2], dtype=np.float32),  # medium
            2: np.array([3], dtype=np.float32),  # hard
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ascii_render = self.ascii_green + [self.ascii_green[-1]] * (length - 1) * 4

    def _get_obs(self):
        return {"ball": self._ball_location, "green": self._green_location}

    def _get_info(self):
        return {}

    def render(self):
        x_ball = round(self._ball_location[0] * 10)
        y_ball = round(self._ball_location[1] * 4)

        for i, line in enumerate(self.ascii_render, 1):
            if i == (len(self.ascii_render) - y_ball):
                print(line[:x_ball] + "o" + line[x_ball + 1 :])
            else:
                print(line)
        print()

    def reset(self, seed=0, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Get the degree of stochasiticity from options
        assert isinstance(options, dict) and "stochasticity" in options
        self.stochasticity = options["stochasticity"]

        # Choose the agent's location uniformly at random
        self._ball_location = np.array([1, 0], dtype=np.float32)
        self._green_location = np.array([1, self.length], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "terminal":
            self.render()

        return observation, info

    def _compute_intermediate_reward(self):
        return 0

    def _perpendicular(self, vec):
        per = np.zeros_like(vec)
        per[0] = -vec[1]
        per[1] = vec[0]
        return per

    def step(self, action):
        if self.discrete:
            # Map the action (element of {0,1,2}) to swing power
            action = self._action_to_speed[action]

        # Get unitvector of ball direction
        distance = self._green_location - self._ball_location
        direction = distance / np.linalg.norm(distance)

        # Sample random deflection of shot
        perpendicular = self._perpendicular(direction)
        std_dev = action**self.stochasticity
        deflection = self._np_random.normal(scale=std_dev).astype(np.float32)

        # Obtain the ball displacement
        shot = direction * action + perpendicular * deflection

        # Update the ball's location
        self._ball_location += shot

        # An episode is done iff the agent has reached the target OR the agent has reached the wall
        if np.linalg.norm(self._green_location - self._ball_location) <= 0.5:
            print("REACHED THE GREEN")
            terminated = True
            reward = 1
        elif not self.observation_space["ball"].contains(
            self._ball_location.astype(np.float32)
        ):
            print("BALL OFF COURSE")
            terminated = True
            reward = -100
        else:
            terminated = False
            reward = self._compute_intermediate_reward()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "terminal":
            self.render()

        return observation, reward, terminated, False, info
