import gymnasium as gym
import pygame
import abc
import numpy as np
import sys
import io, base64
import random
from dataclasses import dataclass
from gym.spaces.box import Box as sBox
from gym.spaces.multi_discrete import MultiDiscrete

colab_rendering = "google.colab" in sys.modules
if colab_rendering:
    import cv2
    from google.colab.patches import cv2_imshow
    from google.colab import output
    import os

    os.environ["SDL_VIDEODRIVER"] = "dummy"


class TrashBot(gym.Env):

    metadata = {"render_modes": ["terminal", "graphic", "none"]}

    def __init__(
        self,
        container_width=100,
        render_mode="graphic",
        action_scaling=1,
        action_range=(-1, 1),
        mode="angle",
        seed=None,
    ):
        super().__init__()
        self.seed(seed)

        self.config = config = TrashBotConfig()
        self.container_width = container_width
        # Objects
        self.floor = Box(**config.floor)
        self.water = Box(**config.water, is_water=True)
        self.base = Box(**config.base)
        self.crates = [Crate(coords=[65, 300], color="rg", margin=0)]
        self.bucket = Crate(width=container_width, **config.bucket)
        self.arm = Arm(
            angles=[-1, -1, np.pi / 2],
            segment_l=(70, 70, 40),
            base_pos=[300 / 2, 200],
            mode=mode,
        )

        # Spaces
        self.observation_space = sBox(low=-1, high=1, shape=(9,), dtype=np.float32)
        self.action_space = None  # Defined in subclasses
        self.action_scaling = action_scaling
        self.action_range = action_range
        self.mode = mode
        # Rendering
        self.bg = make_bg(pygame.Surface(config.screen_size))
        self.logo = "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABDklEQVR4nIXUwU7CQBDG8V8RjBKMRMGDiTHoyYtP4Bv4sD6CZ48ePRjPBvQiKokEPKytW7pbvjTNdnb+05nZaQtrWxQcisRON8sc8c44skw3XYr0mw/ZzQSNQnTSHjkSP4za4aRewSnFXzlRzaG8Z64b2DRyiNSFIb3ScMmUOZM6mVKXiAwKfa6YY97ScMdjw7ZTf8yQ4I4THhgxYsm69br4X3dhxS0z26dtXN6nIe1KI8Ys+OImTwYNSIznGVi2ktinH8azcYCkTih26/MU0m4pNfUxwSd7AZ5lJqFXNmKWjhA1bFEuVnyDYWm5aqSwotawj/r2RFYDXjbgDc0z9qjAzM+gUtXhc+45qG3+AvN4O//tkpDbAAAAAElFTkSuQmCC"
        self.logo = pygame.image.load(io.BytesIO(base64.b64decode(self.logo)))
        self.screen = None
        self.steps = 0
        self.render_mode = render_mode

    def seed(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)

    def reset(self):
        self.bg = make_bg(pygame.Surface(self.config.screen_size))
        self.crates = [Crate(coords=[65, 300], color="rg", margin=0)]
        self.bucket = Crate(width=self.container_width, **self.config.bucket)
        self.arm = Arm(
            angles=[-1, -1, np.pi / 2],
            segment_l=(70, 70, 40),
            base_pos=[300 / 2, 200],
            mode=self.mode,
        )
        self.steps = 0
        return self._get_obs()

    @abc.abstractmethod
    def _process_action(self):
        """
        Adapter for continuous and discrete action spaces
        """
        pass

    def step(self, action):
        r, done = 0, False

        action = self._process_action(action)
        self.arm.change_torques(action)
        self.arm.update()
        self.bucket.update()

        col = collision_detection
        for crate in self.crates:
            ## Collision checks
            crashed = False
            leeway = self.bucket.width / 2 - crate.width / 2
            crate.update()
            # Picking up
            proper_contact = droppable(self.arm.magnet, crate)[0]
            if crate.movable & (self.arm.holding is None) & col(self.arm.magnet, crate):
                if proper_contact:
                    self.arm.attach(crate)
                    r += 1
                else:
                    done = True

            # Unloading trash
            is_droppable, offset = droppable(crate, self.bucket)
            if is_droppable & crate.movable:
                self.arm.detach()
                crate.movable = False
                r += 2 + 2 * (leeway - offset) / leeway
            elif col(crate, self.bucket) & (not is_droppable) & crate.movable:
                done = True

            # Termination conditions
            if (
                any([col(crate, x) for x in [self.base, self.floor]])
                or any([col(self.arm.magnet, x) for x in [self.base, self.bucket]])
                or done
            ):
                done = True
                r -= 2
                break

        # End if all crates are in the container
        if all([not crate.movable for crate in self.crates]):
            done = True

        # Exceeding max steps
        self.steps += 1
        if self.steps >= 150:
            done = True
            r -= 1
        return self._get_obs(), r, done, False, {}

    def render(self, save=False):
        if self.render_mode == "graphic":
            view, img_bgr = None, None
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(self.config.screen_size)

            self.screen.blit(self.bg, (0, 0))
            for obj in [self.water, self.arm, *self.crates, self.bucket, self.base]:
                obj.render(self.screen)
            x, y = self.config.bucket["coords"]
            self.screen.blit(self.logo, (x - 12, y - 30))
            pygame.draw.rect(self.screen, (0, 0, 0), [x - 14, y - 32, 24, 24], width=2)
            pygame.display.flip()

            if save:
                pygame.image.save(self.screen, "frame.png")

            if colab_rendering:
                output.clear()
                view = pygame.surfarray.array3d(self.screen)
                view = view.transpose([1, 0, 2])
                img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                cv2_imshow(img_bgr)
                pygame.time.wait(TrashBotConfig.frame_length)
                return view
        elif self.render_mode == "terminal":
            raise NotImplementedError("terminal rendering not implemented yet")
        elif self.render_mode == "none":
            pass
        else:
            raise NotImplementedError

    def _get_obs(self):
        segment1, segment2 = self.arm.segments[:2]
        size = self.config.screen_size[1]
        return np.concatenate(
            [
                np.array([segment1.torque, segment2.torque]),  # Torques
                np.array(segment1.end) / size,  # End of first segment, x and y
                np.array(segment2.end) / size,  # End of second segment, x and y
                np.array(self.arm.magnet.origin) / size,  # Origin of magnet, x and y
                [self.arm.holding is not None],
            ]
        )  # Holding a crate (bool)


class TrashBotDiscreteEnv(TrashBot):
    def __init__(
        self, container_width=90, n_bins=5, action_scaling=2, mode="angle", **kwargs
    ):
        super().__init__(container_width, action_scaling=action_scaling, mode=mode)
        self.mode = mode
        self.action_mapping = np.linspace(*self.action_range, n_bins + 2)[1:-1]
        self.action_space = MultiDiscrete([n_bins, n_bins])

    def _process_action(self, action):
        ## Linear interpolation from action_range to n_bins
        bins = self.action_mapping
        return np.array([bins[action[0]], bins[action[1]]]) * self.action_scaling


class TrashBotContinuousEnv(TrashBot):
    def __init__(self, container_width=90, action_scaling=2, mode="angle", **kwargs):
        super().__init__(container_width, action_scaling=action_scaling, mode=mode)
        low, high = self.action_range
        self.mode = mode
        self.action_space = sBox(
            low=low * action_scaling, high=high * action_scaling, shape=(2,)
        )

    def _process_action(self, action):
        return np.array(action) * self.action_scaling


###############################
## Helper objects and functions
@dataclass
class TrashBotConfig:
    # Rendering
    screen_size: tuple = (300, 400)
    x, y = screen_size
    fps: int = 10
    frame_length: int = int(1 / fps * 1000)
    water_level = 300

    # Objects
    floor = dict(coords=(x / 2, 390), width=x, height=20, color="k")
    water = dict(coords=(x / 2, 350), width=x, height=100, color="b")
    base = dict(coords=(x / 2, water_level), width=50, height=y / 2, color="grey")
    arm = dict(segment_l=(80, 80, 40), base_pos=(x / 2, 200))

    bucket = dict(
        coords=[240, 290],
        height=80,
        color="g",
        movable=False,
        margin=0,
        is_container=True,
    )
    colors = dict(
        w=(255, 255, 255),
        k=(0, 0, 0),
        r=(200, 100, 100),
        g=(100, 255, 100),
        b=(100, 100, 200),
        grey=(128, 128, 128),
        darkgrey=(64, 64, 64),
        rg=(205, 133, 63),
        lg=(75, 122, 71),
        t=(232, 220, 202),
    )


class Segment:
    def __init__(
        self,
        length: int,
        origin: tuple,
        angle: float = 0,
        torque: float = 0,
        color="darkgrey",
    ):
        self.length = length
        self.origin = origin
        self.end = [origin[0] + length, origin[1]]
        self.angle = angle
        self.torque = torque
        self.color = color

    def render(self, screen: pygame.Surface):
        mapping = TrashBotConfig.colors
        col = np.array(mapping[self.color]) - [20, 20, 20]
        pygame.draw.line(screen, mapping[self.color], self.origin, self.end, width=16)
        pygame.draw.circle(screen, col, self.origin, 11)
        pygame.draw.circle(screen, mapping["grey"], self.origin, 4)


class Arm:
    def __init__(self, segment_l: list, base_pos: tuple, angles: list, mode="angle"):
        self.segments = []
        for length, angle in zip(segment_l, angles):
            segment = Segment(length, base_pos, angle)
            base_pos = segment.end
            self.segments += [segment]
        self.magnet = Box(self.segments[-1].end, 30, 30, color="grey")
        self.holding = None
        self.mode = mode

    def update(self):
        previous_segment = self.segments[0]
        for i, segment in enumerate(self.segments):
            if i > 0:
                segment.origin = previous_segment.end
            if 0 < i < len(self.segments) - 1:
                segment.angle += previous_segment.torque
            x, y, theta = segment.origin[0], segment.origin[1], segment.angle
            segment.angle += segment.torque
            segment.end = (
                x + segment.length * np.cos(theta),
                y + segment.length * np.sin(theta),
            )
            previous_segment = segment
        self.magnet.move_to(*self.segments[-1].end)

    def attach(self, obj):
        obj.attached_to = self.magnet
        self.holding = obj

    def detach(self):
        self.holding.attached_to = None
        self.holding = None

    def change_torques(self, torques: list = [0, 0]):
        for torque, segment in zip(torques, self.segments[: len(torques)]):
            if self.mode == "torque":
                segment.torque += torque
            elif self.mode == "angle":
                segment.torque = torque

    def render(self, screen: pygame.Surface):
        for component in self.segments + [self.magnet]:
            component.render(screen)


class Box:
    def __init__(
        self,
        coords: tuple,
        width: int,
        height: int,
        color="k",
        margin=0,
        is_water=False,
    ):
        self.origin = coords
        self.width = width
        self.height = height
        self.dx, self.dy = 0, 0
        self.color = color
        self.margin = margin
        self.is_water = is_water
        self.is_container = False

    @property
    def bounds(self) -> tuple:
        (x, y), h, w = self.origin, self.height, self.width
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def move_to(self, x: float, y: float):
        x_, y_ = self.origin
        self.dx, self.dy = x_ - x, y_ - y
        self.origin = [x, y]

    def render(self, screen: pygame.Surface):
        mapping = TrashBotConfig.colors
        bottom = self.bounds[3]
        x, y = self.bounds[0] - self.margin, self.bounds[1]
        h, w = self.height, self.width + 2 * self.margin
        color = mapping[self.color]

        offset = 2
        x_, y_ = x - offset, y + offset
        x__, y__ = x_ - offset, y_ + offset
        color = np.clip(np.array(color), 0, 255)
        if self.is_container:
            pygame.draw.polygon(
                screen,
                mapping["darkgrey"],
                [
                    (x__, y__),
                    (x__ + w, y__),
                    (x__ + w - 10, y__ + h),
                    (x__ + 10, y__ + h),
                ],
                width=4,
            )
            pygame.draw.polygon(
                screen,
                mapping["darkgrey"],
                [(x_, y_), (x_ + w, y_), (x_ + w - 10, y_ + h), (x_ + 10, y_ + h)],
                width=4,
            )
            pygame.draw.polygon(
                screen,
                color,
                [(x, y), (x + w, y), (x + w - 10, y + h), (x + 10, y + h)],
            )
        else:
            pygame.draw.rect(screen, mapping["darkgrey"], [x__, y__, w, h], width=4)
            pygame.draw.rect(screen, mapping["darkgrey"], [x_, y_, w, h], width=4)
            pygame.draw.rect(screen, color, [x, y, w, h])
        # different shading underwater
        if bottom > TrashBotConfig.water_level and not self.is_water:
            difference = abs(bottom - TrashBotConfig.water_level)
            d = difference if difference < h else h
            uy = TrashBotConfig.water_level if difference < h else y
            color = np.clip(np.array(color) - [50, 50, -50], 0, 255)
            if self.is_container:
                pygame.draw.polygon(
                    screen,
                    color,
                    [(x, uy), (x + w, uy), (x + w - 10, uy + d), (x + 10, uy + d)],
                )
            else:
                pygame.draw.rect(screen, color, [x, uy, w, d])
        if self.is_container:
            pygame.draw.polygon(
                screen,
                mapping["darkgrey"],
                [(x - 2, y), (x + w + 2, y), (x + w - 8, y + h), (x + 8, y + h)],
                width=5,
            )
        else:
            pygame.draw.rect(screen, mapping["darkgrey"], [x, y, w, h], width=4)


class Crate(Box):
    def __init__(
        self,
        coords: tuple,
        width: int = 50,
        height: int = 50,
        color="k",
        movable=True,
        margin=0,
        is_container=False,
    ):
        super().__init__(coords, width, height, color, margin=margin)
        self.attached_to = None
        self.movable = movable
        self.v = 0
        self.is_container = is_container

    def update(self) -> None:
        # If attached to something, move with it (using the object's dx, dy)
        if self.attached_to is not None:
            self.origin[0] -= self.attached_to.dx
            self.origin[1] -= self.attached_to.dy
        else:
            ## Falling conditions
            y, water = self.origin[1], 310
            if y < water:
                self.v += 0.5
                self.origin[1] = y + self.v
            else:
                self.v = -0.5
                self.origin[1] = water

    def attach_to(self, obj: Box):
        self.attached_to = obj


def collision_detection(obj1: Box, obj2: Box) -> bool:
    # Check if two boxes are colliding
    # left, top, right, bottom
    x1, y1, x2, y2 = obj1.bounds
    x3, y3, x4, y4 = obj2.bounds
    return x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3


def droppable(crate: Box, container: Box, height=10) -> bool:
    # A check specifically for dropping crates
    x1, y1, x2, y2 = crate.bounds
    x3, y3, x4, y4 = container.bounds
    center_offset = abs(crate.origin[0] - container.origin[0])
    return x1 > x3 and x2 < x4 and (-height < (y2 - y3) < 4 * height), center_offset


def generate_cloud(surface, x, y, size):
    # Generate random colors for the cloud
    colors = [(255, 255, 255), (200, 200, 200), (230, 230, 230)]

    # Draw random-sized circles and ellipses to form the cloud shape
    for i in range(size):
        color = random.choice(colors)
        radius = random.randint(5, 20)
        offset_x = random.randint(-radius, radius)
        offset_y = random.randint(-radius, radius)

        # Draw circles
        pygame.draw.circle(surface, color, (x + offset_x, y + offset_y), radius)
        # Draw ellipses
        ellipse_width = random.randint(2 * radius, 4 * radius)
        ellipse_height = random.randint(radius // 2, 2 * radius)
        ellipse_offset_x = random.randint(-radius, radius)
        ellipse_offset_y = random.randint(-radius, radius)

        ellipse_rect = pygame.Rect(
            x + ellipse_offset_x, y + ellipse_offset_y, ellipse_width, ellipse_height
        )
        pygame.draw.ellipse(surface, color, ellipse_rect)


def make_bg(surface):
    surface.fill((135, 206, 250))
    generate_cloud(surface, 50, 50, 10)
    generate_cloud(surface, 200, 50, 10)
    generate_cloud(surface, 300, 50, 10)
    return surface


def test():
    from edugym.envs.interactive import play_env, play_env_terminal

    # Initialize the environment
    render_mode = "graphic"  # 'terminal'
    env = TrashBotDiscreteEnv(render_mode=render_mode, n_bins=5, action_scaling=0.1)

    if render_mode == "graphic":
        play_env(
            env,
            "top-left=w, bottom-left=a, middle=s, bottom-right='d', top-right=e",
            {
                pygame.K_w: [1, 1],
                pygame.K_a: [1, -1],
                pygame.K_s: [0, 0],
                pygame.K_d: [-1, 1],
                pygame.K_e: [-1, -1],
            },
        )
    elif render_mode == "terminal":
        play_env_terminal(
            env,
            "top-left=w, bottom-left=a, middle=s, bottom-right='d', top-right=e",
            {"w": [1, 1], "a": [1, -1], "s": [0, 0], "d": [-1, 1], "e": [-1, -1],},
        )


if __name__ == "__main__":
    test()
