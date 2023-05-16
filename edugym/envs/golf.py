import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete



class GolfEnv(gym.Env):
    metadata = {"render_modes": {"graphic"}}

    def __init__(
        self,
        render_mode="graphic",
        length=17,
        width=7,
        green_radius=1,
        max_swings=17,
        stochasticity=0.05,
    ):
        assert length % 2 == 1, "length of golf course must be an uneven number"
        assert width % 2 == 1, "width of golf course must be an uneven number"
        self.length = length  # The length of the 2D golf course (start to center green)
        self.width = width
        self.max_swings = max_swings
        self.stochasticity = stochasticity

        self.golf_course = GolfCourse(width, length, green_radius)
        self.ball = Ball()

        self.observation_space = MultiDiscrete(self.golf_course.bounds[2:])

        # We have 3 discrete actions, corresponding to the power of a swing: "putt", "chip", "drive"
        self.action_space = Discrete(3)
        self._action_to_distance = {
            0: np.array([1], dtype=np.float32),  # putt
            1: np.array([4], dtype=np.float32),  # chip
            2: np.array([10], dtype=np.float32),  # drive
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None

    def _get_obs(self):
        return self.ball.coordinates.astype(np.int32)

    def _get_info(self):
        return {}

    def render(self):
        if self.render_mode == "terminal":
            raise NotImplementedError("terminal render not imlemented yet")
        if self.render_mode == "graphic":
            view = None
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    self.golf_course.bounds[2:] * RenderConfig.render_scale
                )
                pygame.display.set_caption("Golf Environment")
            self.screen.fill(Colours.background)
            for obj in [self.golf_course, self.ball]:
                obj.render(self.screen)

            # Flip the display
            self.screen = pygame.transform.flip(self.screen, False, True)

            pygame.image.save(self.screen, "frame.png")
            return view

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Set location of the ball and green
        self.ball.move_to(np.array([self.width / 2] * 2, dtype=np.float32))
        self.swings = 0

        return self._get_obs()

    def _perpendicular(self, vec) -> np.ndarray:
        per = np.zeros_like(vec)
        per[0] = -vec[1]
        per[1] = vec[0]
        return per

    def step(self, action: np.ndarray):
        self.swings += 1

        # Map the action (element of {0,1,2}) to swing power
        action = self._action_to_distance[action]

        # Get unitvector of ball direction
        distance = self.golf_course.green_coordinates - self.ball.coordinates
        direction = distance / np.linalg.norm(distance)

        # Sample random deflection of shot
        perpendicular = self._perpendicular(direction)
        std_dev = self.stochasticity * (action**2)
        directional_deflection = self.np_random.normal(scale=std_dev).astype(
            np.float32
        )
        transverse_deflection = self.np_random.normal(scale=std_dev).astype(np.float32)

        # Obtain the ball displacement
        shot = (
            direction * (action + directional_deflection)
            + perpendicular * transverse_deflection
        )

        # Update the ball's location
        self.ball.move_to(self.ball.coordinates + shot)

        # An episode is done iff the agent has reached the target OR the agent has reached the wall
        if self.golf_course.on_green(self.ball.coordinates):
            # Ball reached the green
            done = True
            reward = max((self.max_swings - self.swings) / self.max_swings, -1)
        elif (
            not self.observation_space.contains(self._get_obs())
            or self.swings > self.max_swings
        ):
            # Ball off course
            done = True
            reward = -1
        else:
            done = False
            reward = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def close(self):
        if self.screen:
            pygame.quit()


@dataclass
class Colours:
    grass = (0, 154, 23)
    green = (89, 166, 8)
    ball = (255, 255, 255)
    flagpole = (255, 255, 255)
    flag = (255, 0, 0)
    background = (255, 255, 255)
    darkgrey = (64, 64, 64)


@dataclass
class RenderConfig:
    screen_size: tuple = (10, 30)
    render_scale: int = 30
    fps: int = 10
    frame_length: int = int(1 / fps * 1000)


class Ball:
    def __init__(self) -> None:
        pass

    def move_to(self, coordinates: np.ndarray) -> None:
        self.coordinates = coordinates

    def render(self, screen: pygame.Surface) -> None:
        colour = Colours.ball
        pygame.draw.circle(
            screen,
            colour,
            (self.coordinates * RenderConfig.render_scale).astype(np.int32),
            radius=1,
        )


class GolfCourse:
    def __init__(self, width: int, length: int, green_radius: int):
        self.bounds = np.array([0, 0, width, length + width], dtype=np.int32)
        self.green_coordinates = np.array(
            [width / 2, length + width / 2], dtype=np.float32
        )
        self.green_radius = green_radius

    def on_green(self, ball_coordinates: np.ndarray) -> bool:
        vector_to_green_center = ball_coordinates - self.green_coordinates
        return np.linalg.norm(vector_to_green_center) < self.green_radius

    def render(self, screen: pygame.Surface):
        scaled_bounds = self.bounds * RenderConfig.render_scale
        pygame.draw.rect(screen, Colours.grass, scaled_bounds)
        pygame.draw.circle(
            screen,
            Colours.green,
            (self.green_coordinates * RenderConfig.render_scale).astype(np.int32),
            radius=self.green_radius * RenderConfig.render_scale,
        )
        pygame.draw.line(
            screen,
            Colours.flagpole,
            (self.green_coordinates * RenderConfig.render_scale).astype(np.int32),
            (
                (self.green_coordinates + np.array([0, 0.5]))
                * RenderConfig.render_scale
            ).astype(np.int32),
        )
        pygame.draw.polygon(
            screen,
            Colours.flag,
            [
                (
                    (self.green_coordinates + np.array([0, 0.5]))
                    * RenderConfig.render_scale
                ).astype(np.int32),
                (
                    (self.green_coordinates + np.array([0.2, 0.4]))
                    * RenderConfig.render_scale
                ).astype(np.int32),
                (
                    (self.green_coordinates + np.array([0, 0.3]))
                    * RenderConfig.render_scale
                ).astype(np.int32),
            ],
        )
