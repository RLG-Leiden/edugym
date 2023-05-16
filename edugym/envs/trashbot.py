import gymnasium
import pygame
import abc
import numpy as np
import sys
from dataclasses import dataclass
from gymnasium.spaces.box import Box as sBox
from gymnasium.spaces.multi_discrete import MultiDiscrete

class TrashBot(gymnasium.Env):
    def __init__(self, container_width=80, render_mode='human',
                 action_scaling=1, action_range=(-1, 1), mode='angle'):
        super().__init__()

        self.config = config = TrashBotConfig()
        self.container_width = container_width
        # Objects
        self.floor = Box(**config.floor)
        self.water = Box(**config.water)
        self.base  = Box(**config.base)
        self.crates = [Crate(coords=[65, 300], color='r', margin=0)]
        self.bucket = Crate(width=container_width, **config.bucket)
        self.arm = Arm(angles=[-1, -1, np.pi/2], segment_l=(70, 70, 40), base_pos=[300/2, 200], mode=mode)

        # Spaces
        self.observation_space = sBox(low=-1, high=1, shape=(9, ), dtype=np.float32)
        self.action_space = None # Defined in subclasses
        self.action_scaling = action_scaling
        self.action_range = action_range
        self.mode = mode
        # Rendering
        self.screen = None
        self.steps = 0

    def reset(self, seed=None):
        self.crates = [Crate(coords=[65, 300], color='r', margin=0)]
        self.bucket = Crate(width=self.container_width, **self.config.bucket)
        self.arm = Arm(angles=[-1, -1, np.pi/2], segment_l=(70, 70, 40), base_pos=[300/2, 200], mode=self.mode)
        self.steps = 0
        return self._get_obs()
    
    @abc.abstractmethod
    def _process_action(self):
        """
        Adapter for continuous and discrete action spaces
        """
        pass

    def step(self, action):
        r, done, truncated = 0, False, False

        action = self._process_action(action)
        self.arm.change_torques(action)
        self.arm.update()
        self.bucket.update()

        col = collision_detection
        for crate in self.crates:
            ## Collision checks
            crashed = False
            leeway = self.bucket.width/2 - crate.width/2
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
                r += 2 + 2 * (leeway-offset)/leeway
            elif col(crate, self.bucket) & (not is_droppable) & crate.movable:
                done = True

            # Termination conditions
            if any([col(crate, x) for x in [self.base, self.floor]]) or \
               any([col(self.arm.magnet, x) for x in [self.base, self.bucket]]) \
               or done:
                    done = True
                    r -= 2
                    break

        # End if all crates are in the container
        if all([not crate.movable for crate in self.crates]):
          done = True 

        # Exceeding max steps
        self.steps += 1
        if self.steps >= 150:
            truncated = True
            r -= 1
        return self._get_obs(), r, done, truncated, {}

    def render(self, mode='human'):
        view, img_bgr = None, None
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.config.screen_size)
        self.screen.fill(self.config.colors['w'])
        for obj in [self.water, self.bucket, self.arm, *self.crates, self.base]:
            obj.render(self.screen)
        pygame.display.flip()

        if mode == 'human' and colab_rendering:
            cv2_imshow(img_bgr)
            pygame.time.wait(TrashBotConfig.frame_length)
        elif mode == 'save':
            pygame.image.save(self.screen, 'frame.png')
        return view

    def _get_obs(self):
        segment1, segment2 = self.arm.segments[:2]
        size = self.config.screen_size[1]
        return np.concatenate([
            np.array([segment1.torque, segment2.torque]), # Torques
            np.array(segment1.end)/size, # End of first segment, x and y
            np.array(segment2.end)/size, # End of second segment, x and y
            np.array(self.arm.magnet.origin)/size, # Origin of magnet, x and y
            [self.arm.holding is not None]])       # Holding a crate (bool)


class TrashBotDiscreteEnv(TrashBot):
    def __init__(self, container_width=70, n_bins=5, action_scaling=2, mode='angle', **kwargs):
        super().__init__(container_width, action_scaling=action_scaling, mode=mode)
        self.mode = mode
        self.action_mapping = np.linspace(*self.action_range, n_bins+2)[1:-1]
        self.action_space = MultiDiscrete([n_bins, n_bins])
      
    def _process_action(self, action):
        ## Linear interpolation from action_range to n_bins
        bins = self.action_mapping
        return np.array([bins[action[0]], bins[action[1]]]) * self.action_scaling


class TrashBotContinuousEnv(TrashBot):
    def __init__(self, container_width=70, action_scaling=2, mode='angle', **kwargs):
        super().__init__(container_width, action_scaling=action_scaling, mode=mode)
        low, high = self.action_range 
        self.mode = mode
        self.action_space = sBox(low=low*action_scaling, high=high*action_scaling, shape=(2, ))
    
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
    frame_length: int = int(1/fps*1000)
    water_level = 300

    # Objects
    floor = dict(coords=(x/2, 390), width=x, height=20,  color='k')
    water = dict(coords=(x/2, 350), width=x, height=100, color='b')
    base  = dict(coords=(x/2, water_level), width=50,  height=y/2, color='grey')
    arm   = dict(segment_l=(80, 80, 40), base_pos=(x/2, 200))
    
    bucket = dict(coords=[240, 290], height=60, color='g', movable=False, margin=0)
    colors = dict(
        w=(255, 255, 255),
        k=(0, 0, 0),
        r=(255, 0, 0),
        g=(0, 255, 0),
        b=(0, 0, 255),
        grey=(128, 128, 128),
        darkgrey=(64, 64, 64))


class Segment:
    def __init__(self, length: int, origin: tuple, angle: float=0, 
                 torque: float=0, color='darkgrey'):
        self.length = length
        self.origin = origin
        self.end = [origin[0] + length, origin[1]]
        self.angle = angle
        self.torque = torque
        self.color = color 
    
    def render(self, screen: pygame.Surface):
        mapping = TrashBotConfig.colors
        pygame.draw.line(screen, mapping[self.color], self.origin, self.end, width=15)
        pygame.draw.circle(screen, mapping[self.color], self.origin, 10)
        pygame.draw.circle(screen, mapping['grey'], self.origin, 3)


class Arm:
    def __init__(self, segment_l: list, base_pos: tuple, angles: list, mode='angle'):
        self.segments = []
        for length, angle in zip(segment_l, angles):
            segment = Segment(length, base_pos, angle)
            base_pos = segment.end
            self.segments += [segment]
        self.magnet  = Box(self.segments[-1].end, 15, 15, color='grey')
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
            segment.end = (x + segment.length * np.cos(theta),
                           y + segment.length * np.sin(theta))
            previous_segment = segment
        self.magnet.move_to(*self.segments[-1].end)

    def attach(self, obj):
        obj.attached_to = self.magnet
        self.holding = obj
    
    def detach(self):
        self.holding.attached_to = None
        self.holding = None
    
    def change_torques(self, torques: list=[0, 0]):
        for torque, segment in zip(torques, self.segments[:len(torques)]):
            if self.mode == 'torque':
                segment.torque += torque
            elif self.mode == 'angle':
                segment.torque = torque

    def render(self, screen: pygame.Surface):
        for component in self.segments + [self.magnet]:
            component.render(screen)


class Box:
    def __init__(self, coords: tuple, width: int, height: int, color='k', margin=0):
        self.origin = coords
        self.width = width
        self.height = height
        self.dx, self.dy = 0, 0
        self.color = color
        self.margin = margin

    @property
    def bounds(self) -> tuple:
        (x, y), h, w = self.origin, self.height, self.width
        return (x - w/2, y - h/2, 
                x + w/2, y + h/2)
    
    def move_to(self, x: float, y: float):
        x_, y_ = self.origin
        self.dx, self.dy = x_ - x, y_ - y
        self.origin = [x, y]

    def render(self, screen: pygame.Surface):
        mapping = TrashBotConfig.colors
        x, y = self.bounds[0] - self.margin, self.bounds[1]
        h, w = self.height, self.width + 2*self.margin
        pygame.draw.rect(screen, mapping[self.color], [x, y, w, h])
        pygame.draw.rect(screen, mapping['darkgrey'], [x, y, w, h], width=2)

        
class Crate(Box):
    def __init__(self, coords: tuple, width: int=45, height: int=45, 
                 color='k', movable=True, margin=0):
        super().__init__(coords, width, height, color, margin=margin)
        self.attached_to = None
        self.movable = movable
        self.v = 0

    def update(self) -> None:
        # If attached to something, move with it (using the object's dx, dy)
        if self.attached_to is not None:
            self.origin[0] -= self.attached_to.dx
            self.origin[1] -= self.attached_to.dy
        else:
            ## Falling conditions
            y, water = self.origin[1], 320
            if y < water:
                self.v += .5
                self.origin[1] = y + self.v
            else:
                self.v = -.5
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
