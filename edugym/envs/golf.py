import sys
from dataclasses import dataclass
from typing import Union

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
        max_hits=17,
        stochasticity_level=0.05,
    ):
        assert length % 2 == 1, "length of golf course must be an uneven number"
        assert width % 2 == 1, "width of golf course must be an uneven number"
        self.length = length  # The length of the 2D golf course (start to center green)
        self.width = width
        self.max_hits = max_hits
        self.stochasticity_level = stochasticity_level

        self.golf_course = GolfCourse(width, length, green_radius)
        self.ball = Ball()

        self.observation_space = MultiDiscrete(self.golf_course.bounds[2:])

        # We have 3 discrete actions, corresponding to the power of a swing: "putt", "chip", "drive"
        self.action_space = Discrete(3)
        self._action_to_distance = {
            0: np.array([1]),  # putt
            1: np.array([4]),  # chip
            2: np.array([10]),  # drive
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None

    def _get_obs(self):
        return self.ball.coordinates.astype(int)

    def _get_info(self):
        return {}

    def render(self):
        if self.render_mode == "terminal":
            raise NotImplementedError("terminal render not imlemented yet")
        if self.render_mode == "graphic":
            view = None
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(self.golf_course.render_size)
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
        self.ball.move_to(np.array([self.width / 2] * 2))
        self.hits = 0

        return self._get_obs()

    def _perpendicular(self, vec) -> np.ndarray:
        per = np.zeros_like(vec)
        per[0] = -vec[1]
        per[1] = vec[0]
        return per

    def step(self, action: np.ndarray):
        self.hits += 1

        # Map the action (element of {0,1,2}) to swing power
        action = self._action_to_distance[action]

        # Get unitvector of ball direction
        distance = self.golf_course.green_coordinates - self.ball.coordinates
        direction = distance / np.linalg.norm(distance)

        # Sample random deflection of shot
        perpendicular = self._perpendicular(direction)
        std_dev = self.stochasticity_level * action
        directional_deflection = self.np_random.normal(scale=std_dev)
        transverse_deflection = self.np_random.normal(scale=std_dev)

        # Obtain the ball displacement
        aimed_coordinates = self.ball.coordinates + direction * action
        shot = (
            direction * (action + directional_deflection)
            + perpendicular * transverse_deflection
        )
        reached_coordinates = self.ball.coordinates + shot

        # Update the ball's location
        self.ball.move_to(reached_coordinates, aimed_coordinates, std_dev)

        # An episode is done iff the ball is on the green, off course, or when the
        # maximum number of hits are reached.
        if self.golf_course.on_green(self.ball.coordinates):
            # Ball reached the green
            done = True
            reward = max((self.max_hits - self.hits) / self.max_hits, -1)
        elif (
            not self.observation_space.contains(self._get_obs())
            or self.hits > self.max_hits
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
    tee = (0, 128, 19)
    ball = (255, 255, 255)
    flagpole = (255, 255, 255)
    flag = (255, 0, 0)
    background = (255, 255, 255)
    darkgrey = (64, 64, 64)
    water = (35, 137, 218)
    golf_cart = (255, 255, 255)
    tires = (0, 0, 0)


class RenderObject:
    render_scale = 60
    fps = 10
    frame_length = int(1 / fps * 1000)
    edge_size = 1
    render_offset = np.array([edge_size] * 2)

    def scale_to_int(self, value: Union[np.ndarray, int, float]):
        if isinstance(value, int) or isinstance(value, float):
            return int(value * self.render_scale)
        elif isinstance(value, np.ndarray):
            return (value * self.render_scale).astype(int)
        else:
            raise ValueError("Wrong input type")


class Ball(RenderObject):
    def __init__(self):
        self.std_dev = None
        self.aimed_coordinates = None

    def move_to(
        self,
        coordinates: np.ndarray,
        aimed_coordinates: np.ndarray = None,
        std_dev: float = None,
    ):
        self.coordinates = coordinates
        self.aimed_coordinates = aimed_coordinates
        self.std_dev = std_dev

    def render(self, screen: pygame.Surface) -> None:
        if self.aimed_coordinates is not None:
            pygame.draw.circle(
                screen,
                Colours.darkgrey,
                self.scale_to_int(self.aimed_coordinates + self.render_offset),
                radius=self.scale_to_int(0.04) or 1,
            )
            pygame.draw.circle(
                screen,
                Colours.darkgrey,
                self.scale_to_int(self.aimed_coordinates + self.render_offset),
                radius=int(self.std_dev * 2 * self.render_scale),
                width=self.scale_to_int(0.04) or 1,
            )

        pygame.draw.circle(
            screen,
            Colours.ball,
            self.scale_to_int(self.coordinates + self.render_offset),
            radius=self.scale_to_int(0.1) or 1,
        )


class GolfCourse(RenderObject):
    def __init__(self, width: int, length: int, green_radius: int):
        self.width = width
        self.length = length
        self.green_radius = green_radius

        self.bounds = np.array([0, 0, width, length + width])
        self.green_coordinates = np.array([width / 2, length + width / 2])

    def on_green(self, ball_coordinates: np.ndarray) -> bool:
        vector_to_green_center = ball_coordinates - self.green_coordinates
        return np.linalg.norm(vector_to_green_center) < self.green_radius

    def render(self, screen: pygame.Surface):
        # Draw the water
        water_rect = np.append([0, 0], self.render_size)
        pygame.draw.rect(screen, Colours.water, water_rect)

        # Draw the course
        course_rect = np.append(self.render_offset, self.bounds[2:])
        pygame.draw.rect(screen, Colours.grass, self.scale_to_int(course_rect))

        # Draw the tee
        tee_rect = np.append(self.render_offset + (self.width / 2) - 0.2, [0.4, 0.4])
        pygame.draw.rect(screen, Colours.tee, self.scale_to_int(tee_rect))

        # Draw the green and flag
        render_green_coordinates = self.green_coordinates + self.render_offset
        pygame.draw.circle(
            screen,
            Colours.green,
            self.scale_to_int(render_green_coordinates),
            radius=self.scale_to_int(self.green_radius) or 1,
        )
        pygame.draw.line(
            screen,
            Colours.flagpole,
            self.scale_to_int(render_green_coordinates),
            self.scale_to_int(render_green_coordinates + np.array([0, 1])),
            width=self.scale_to_int(0.1) or 1,
        )
        pygame.draw.polygon(
            screen,
            Colours.flag,
            [
                self.scale_to_int(render_green_coordinates + np.array([0, 1])),
                self.scale_to_int(render_green_coordinates + np.array([0.4, 0.8])),
                self.scale_to_int(render_green_coordinates + np.array([0, 0.6])),
            ],
        )

        # Draw the golf cart body
        cart_rect = np.append(self.render_offset + np.array([1, 1]), [1, 0.3])
        pygame.draw.rect(screen, Colours.golf_cart, self.scale_to_int(cart_rect))
        cart_rect = np.append(self.render_offset + np.array([1.1, 1.7]), [0.7, 0.1])
        pygame.draw.rect(screen, Colours.golf_cart, self.scale_to_int(cart_rect))
        pygame.draw.line(
            screen,
            Colours.golf_cart,
            self.scale_to_int(self.render_offset + np.array([1, 1.3])),
            self.scale_to_int(self.render_offset + np.array([1.1, 1.7])),
            width=self.scale_to_int(0.05) or 1,
        )
        pygame.draw.line(
            screen,
            Colours.golf_cart,
            self.scale_to_int(self.render_offset + np.array([1.9, 1])),
            self.scale_to_int(self.render_offset + np.array([1.7, 1.7])),
            width=self.scale_to_int(0.05) or 1,
        )

        # Draw the golf cart wheels
        pygame.draw.circle(
            screen,
            Colours.tires,
            self.scale_to_int(self.render_offset + np.array([1.2, 1])),
            self.scale_to_int(0.15) or 1,
        )
        pygame.draw.circle(
            screen,
            Colours.tires,
            self.scale_to_int(self.render_offset + np.array([1.9, 1])),
            self.scale_to_int(0.15) or 1,
        )
        pygame.draw.circle(
            screen,
            Colours.golf_cart,
            self.scale_to_int(self.render_offset + np.array([1.2, 1])),
            self.scale_to_int(0.06) or 1,
        )
        pygame.draw.circle(
            screen,
            Colours.golf_cart,
            self.scale_to_int(self.render_offset + np.array([1.9, 1])),
            self.scale_to_int(0.06) or 1,
        )

    @property
    def render_size(self):
        render_bounds = self.bounds[2:] + np.array([self.edge_size * 2] * 2)
        return render_bounds * self.render_scale
