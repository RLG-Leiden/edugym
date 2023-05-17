import numpy as np
import math

import gymnasium as gym
from gymnasium import spaces

import pygame

# Define colors
#grey = (128, 128, 128)
brown = (150, 75, 0)
white = (255, 255, 255)
black = (0, 0, 0)
grey = (235,235,235)
red = (156,39,6)
yellow = (243, 188, 87)
green = (0, 255, 0)
blue = (35, 110, 150)
light_blue = (154, 187, 255)
dark_grey = (110,110,110)
ligth_grey = (235,235,235)
dark_blue = (35, 110, 150)

class BoulderEnv(gym.Env):
    metadata = {"render_modes": ["terminal", "graphic"]}

    def __init__(self, render_mode=None, height=10, n_grips=2, max_steps=100):
        '''
        |- |
        |- |   '-': grip
        | -|   '*': agent
        | -|
        _*__
        '''
        self.height = height
        self.n_grips = n_grips
        self.max_steps = max_steps
        self.steps_taken = 0
        

        # Observation is the current height of the agent.
        self.observation_space = spaces.Box(
                            low=np.array([0]),
                            high=np.array([self.height]),
                            dtype=int)

        # We have n_grips actions, every time the agent needs to grip the right grips
        self.action_space = spaces.Discrete(self.n_grips)
        
        self.pygame_initialized = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.reset()

    def _get_obs(self):
        return self._agent_location

    def render(self):
        wall = np.zeros((self.height+1, self.n_grips)) # create the wall
        for y in range(1, self.height+1):
            wall[y, self.grips[y-1]] = 1

        if self.render_mode == 'terminal':
            for y in range(self.height, 0, -1):
                print("|", end="")
                for x in wall[y]:
                    if x == 1:
                        if y == self._agent_location:
                            print('*', end="")
                        else:
                            print('-', end="")
                    elif x == 0:
                        print(' ', end="")
                print("|")
            if self._agent_location == 0:
                print("_" * int((self.n_grips/2)+1), end="")
                print("*", end="")
                print("_" * int((self.n_grips/2)), end="")
            else:
                print("_" * int((self.n_grips+2)), end="")
            print("")
            print("")

        elif self.render_mode == 'graphic':
            # Initialize pygame
            if not self.pygame_initialized:
                pygame.init()
                self.cell_size = 50
                screen_width, screen_height = (
                                            self.n_grips * self.cell_size,
                                            (self.height+1) * self.cell_size,
                                            )
                self.screen = pygame.display.set_mode([screen_width, screen_height])
                pygame.display.set_caption("Bouldering")
                self.pygame_initialized = True
            # Set background color
            self.screen.fill(brown)
            # Draw grid
            for y in range(self.height+1):
                for x in range(self.n_grips):
                    rect = pygame.Rect(
                        x * self.cell_size,
                        (self.height-y) * self.cell_size,
                        40,
                        20,
                    )
                    if y == 0 and self._agent_location == 0:
                        if x == 1:
                            pygame.draw.circle(self.screen, yellow, center=(x * self.cell_size + self.cell_size*0.5, (self.height-y) * self.cell_size + self.cell_size * 0.5), radius=self.cell_size * 0.2)
                            pygame.draw.circle(self.screen, black, center=(x * self.cell_size + self.cell_size*0.45, (self.height-y) * self.cell_size + self.cell_size * 0.4), radius=self.cell_size * 0.05)
                            pygame.draw.circle(self.screen, black, center=(x * self.cell_size + self.cell_size*0.6, (self.height-y) * self.cell_size + self.cell_size * 0.4), radius=self.cell_size * 0.05)
                            pygame.draw.arc(self.screen, black, pygame.Rect(x * self.cell_size + self.cell_size*0.4, (self.height-y) * self.cell_size + self.cell_size * 0.5, 10, 5), 3.54, 5.88, 2)
                        else:
                            pygame.draw.rect(self.screen, brown, rect)
                    else:
                        if wall[y][x] == 1: # grip
                            if y == self._agent_location:
                                pygame.draw.circle(self.screen, yellow, center=(x * self.cell_size + self.cell_size*0.5, (self.height-y) * self.cell_size + self.cell_size * 0.5), radius=self.cell_size * 0.2)
                                pygame.draw.circle(self.screen, black, center=(x * self.cell_size + self.cell_size*0.45, (self.height-y) * self.cell_size + self.cell_size * 0.4), radius=self.cell_size * 0.05)
                                pygame.draw.circle(self.screen, black, center=(x * self.cell_size + self.cell_size*0.6, (self.height-y) * self.cell_size + self.cell_size * 0.4), radius=self.cell_size * 0.05)
                                pygame.draw.arc(self.screen, black, pygame.Rect(x * self.cell_size + self.cell_size*0.4, (self.height-y) * self.cell_size + self.cell_size * 0.5, 10, 5), 3.54, 5.88, 2)
                                self.draw_cobblestone((x * self.cell_size, (self.height-y) * self.cell_size), (25, 20))
                            else:
                                #pygame.draw.rect(self.screen, grey, rect)
                                self.draw_cobblestone((x * self.cell_size, (self.height-y) * self.cell_size), (25, 20))
                        elif wall[y][x] == 0:
                            if y== 0:
                                pygame.draw.rect(self.screen, brown, rect)
                            else:
                                self.draw_cobblestone((x * self.cell_size, (self.height-y) * self.cell_size+10), (22, 20), color=black)
                                self.draw_crack((x * self.cell_size + self.cell_size*0.4, (self.height-y) * self.cell_size + self.cell_size * 0.5), 10, 0.6)
                                self.draw_crack((x * self.cell_size + self.cell_size*0.4, (self.height-y) * self.cell_size + self.cell_size * 0.5), 10, -0.6)

            # Flip the display
            pygame.display.flip()
            # Wait for a short time to slow down the rendering
            pygame.time.wait(25)


    def reset(self, seed=1, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        np.random.seed(seed)

        # Initialize positions of grips
        self._agent_location = np.array([0], dtype=int)
        self.grips = np.random.choice(self.n_grips, self.height)
        self.steps_taken = 0

        observation = self._get_obs()

        return observation

    def step(self, action):
        # if the action match the given grip 
        if action == self.grips[self._agent_location]:
            self._agent_location += 1
        else:
            self._agent_location = 0

        if self._agent_location == self.height:
            #print("REACHED THE TARGET")
            reward = 1
            terminated = True
            truncated = False
        else:
            reward = 0
            terminated = False
            truncated = False

        self.steps_taken += 1

        if self.steps_taken == self.max_steps:
            #print("MAX STEPS IS REACHED")
            terminated = False
            truncated = True

        observation = self._get_obs()

        return observation, reward, terminated, truncated, {}

    def draw_crack(self, position, height, angle):
        crack_color = brown
        # Define the tree parameters
        scale_factor = 0.7
        if height < 5:
            return
            
          # Calculate the endpoint of the branch
        endpoint_x = position[0] + height * math.sin(angle)
        endpoint_y = position[1] - height * math.cos(angle)
        endpoint = (int(endpoint_x), int(endpoint_y))
            
          # Draw the branch
        pygame.draw.line(self.screen, crack_color, position, endpoint, 5)
            
          # Draw the left branch recursively
        self.draw_crack(endpoint, height * scale_factor, angle - math.pi / 6)
            
          # Draw the right branch recursively
        self.draw_crack(endpoint, height * scale_factor, angle + math.pi / 6) 

    def draw_cobblestone(self, pos, size, color=(128, 128, 128)):
        stone_width = size[0] // 2
        stone_height = size[1] // 4
        for row in range(2):
            for col in range(4):
                stone_pos = (pos[0] + col * stone_width, pos[1] + row * stone_height)
                if (row + col) % 2 == 0:
                    pygame.draw.rect(self.screen, color, (stone_pos[0], stone_pos[1], stone_width, stone_height))
                else:
                    pygame.draw.circle(self.screen, color, (stone_pos[0] + stone_width // 2, stone_pos[1] + stone_height // 2), stone_width // 2)
def test():
    render_mode = "graphic"  # 'inline'
    # Initialize the environment
    from edugym.envs.interactive import play_env
    env = BoulderEnv(render_mode=render_mode)
    play_env(env, "a=left, d=right", {"a":0, "d": 1})

if __name__ == "__main__":
    test()