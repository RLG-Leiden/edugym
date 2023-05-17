import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import math
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
#RED = (255, 0, 0)
RED = (156, 39, 6)
SKY_BLUE = (135, 206, 235)
STONE_GRAY = (136, 140, 141)
GRASS_GREEN = (86, 125, 70)
ROAD_SIGN_GREEN = (1,115,92)
TREE_GREEN = (34, 139, 34)
#YELLOW = (255,255,0)
YELLOW = (243, 188, 87)

MAX_SPEED = 10


class RoadrunnerEnv(gym.Env):
    metadata = {"render_modes": ["terminal", "graphic", "none"]}

    def __init__(self, render_mode="terminal", size=10, discrete=True, negative_reward_size=-100, max_episode_steps=100):
        super().__init__()
        self.size = size  # The size of the single dimension grid

        # Observations are dictionaries with the agent's location along a 1-D axis and speed.
        self.observation_space = spaces.Discrete(self.size * (MAX_SPEED + 1))

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
        if self.render_mode == "graphic":
            self.init_pygame()
        self.negative_reward_size = negative_reward_size
        self.max_episode_steps = max_episode_steps
        self._step_counter = 0

    def init_pygame(self):
        """
        Initialize the pygame screen and font.
        """
        self.grid_size = 100
        self.pg_width = self.size * self.grid_size
        self.pg_height = self.grid_size * 3
        self.screen = pygame.display.set_mode((self.pg_width, self.pg_height))
        pygame.display.set_caption("RoadrunnerEnv")
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", int(self.grid_size/2))

    def draw_agent(self, position, size):
        SIZE_SCALER = 6
        pygame.draw.ellipse(self.screen, YELLOW, (position[0], position[1], size, size))
        pygame.draw.ellipse(self.screen, BLACK, (position[0]+(2*size/SIZE_SCALER), position[1]+size/SIZE_SCALER, size/SIZE_SCALER, size/SIZE_SCALER)) # eye
        pygame.draw.ellipse(self.screen, BLACK, (position[0]+(size)-(2*size/SIZE_SCALER), position[1]+(size/SIZE_SCALER), size/SIZE_SCALER, size/SIZE_SCALER)) # eye

    def draw_sign(self, position, size):
        sign = pygame.Rect(position[0]+(size/10), position[1], size*0.8, size*0.5)
        pygame.draw.rect(self.screen, ROAD_SIGN_GREEN, sign)
        pygame.draw.line(self.screen, BLACK, (position[0]+(size/2), position[1]+size), (position[0]+(size/2), position[1]+(size/2)), 5) # signpost
        sign_font = pygame.font.SysFont("Arial", int(size/4))
        text = sign_font.render("danger", True, WHITE) 
        text_rect = text.get_rect(center=sign.center)
        self.screen.blit(text, text_rect)

    def draw_tree(self, position, height, angle):
        # Define the tree parameters
        scale_factor = 0.7
        if height < 5:
            return

        # Calculate the endpoint of the branch
        endpoint_x = position[0] + height * math.sin(angle)
        endpoint_y = position[1] - height * math.cos(angle)
        endpoint = (int(endpoint_x), int(endpoint_y))

        # Draw the branch
        pygame.draw.line(self.screen, TREE_GREEN, position, endpoint, 5)

        # Draw the left branch recursively
        self.draw_tree(endpoint, height * scale_factor, angle - math.pi / 6)

        # Draw the right branch recursively
        self.draw_tree(endpoint, height * scale_factor, angle + math.pi / 6)


    def draw_grid(self):
        """
        Draw the game grid on then pygame screen.

        The grid is a 1-D line with the agent, target and wall locations marked.
        """
        for i in range(self.size):
            # Draw area above and below 1-D line
            bg_rect_up = pygame.Rect(i * self.grid_size, 0, self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, SKY_BLUE, bg_rect_up)
            
            bg_rect_down = pygame.Rect(i * self.grid_size, 2*self.grid_size, self.grid_size, self.grid_size)
            grass = pygame.Rect(i * self.grid_size, 2*self.grid_size, self.grid_size, self.grid_size/5)
            if i != self.size-1:
                pygame.draw.rect(self.screen, STONE_GRAY, bg_rect_down)
                pygame.draw.rect(self.screen, GRASS_GREEN, grass)
            else:
                pygame.draw.rect(self.screen, SKY_BLUE, bg_rect_down)

            rect = pygame.Rect(i * self.grid_size, self.grid_size, self.grid_size, self.grid_size)
            if i == self._agent_location[0]:
                pygame.draw.rect(self.screen, SKY_BLUE, rect)
                text = self.font.render(f"{self._agent_location[1]}", True, BLACK)
                agent = self.draw_agent((i*self.grid_size, self.grid_size), self.grid_size)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            elif i == self._target_location[0]:
                pygame.draw.rect(self.screen, SKY_BLUE, rect)
                self.draw_sign((i*self.grid_size, self.grid_size), self.grid_size)
            elif i == self._wall_location[0]:
                pygame.draw.rect(self.screen, SKY_BLUE, rect)
                text = self.font.render("W", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            else:
                pygame.draw.rect(self.screen, SKY_BLUE, rect)
            
            if i % 5 == 0 and i < (self.size-2): 
                self.draw_tree((i*self.grid_size, 2*self.grid_size), 0.5*self.grid_size, 0)
            
            
    def _get_obs(self):
        return (self._agent_location[0] * MAX_SPEED) + self._agent_location[1]

    def _get_info(self):
        return {"target": self._target_location, "wall": self._wall_location, "steps": self._step_counter}

    def _render_frame(self):
        if self.render_mode == "terminal":
            for i in range(self.size):
                if i == self._agent_location[0]:
                    print("[A]", end=" ")
                elif i == self._target_location[0]:
                    print("[T]", end=" ")
                elif i == self._wall_location[0]:
                    print("[W]", end=" ")
                else:
                    print("[ ]", end=" ")
            print()
        elif self.render_mode == "graphic":
            self.screen.fill(WHITE)
            self.draw_grid()
            pygame.display.update()

            self.clock.tick(15)
        elif self.render_mode == "none":
            pass
        else:
            raise NotImplementedError

    def reset(self, seed=None, options=None):
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

    def render(self):
      self._render_frame()

    def _compute_intermediate_reward(self):
        return -1

    def step(self, action):
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
            terminated = False
            reward = self.negative_reward_size
        elif new_dx < 0:
            new_dx = 0
            terminated = False
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
def test():
    render_mode = "graphic"  # 'inline'
    # Initialize the environment
    from edugym.envs.interactive import play_env
    env = RoadrunnerEnv(render_mode=render_mode)
    play_env(env, "w=speed up, s=slow down, d= do nothing / idle", {"w":0, "s": 1, "d": 2})

if __name__ == "__main__":
    test()