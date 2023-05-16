import numpy as np

import gymnasium as gym
from gymnasium import spaces

import pygame

# Define colors
white = (255, 255, 255)
black = (0, 0, 128)
grey = (128, 128, 128)
red = (255, 0, 0)
green = (128, 128, 0)

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
            self.screen.fill(white)
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
                        pygame.draw.rect(self.screen, grey, rect)
                    else:
                        if wall[y][x] == 1: # grip
                            if y == self._agent_location:
                                pygame.draw.circle(self.screen, black, center=(x * self.cell_size + self.cell_size*0.5, (self.height-y) * self.cell_size + self.cell_size * 0.5), radius=self.cell_size * 0.2)
                            else:
                                pygame.draw.rect(self.screen, green, rect)
                        elif wall[y][x] == 0:
                            pygame.draw.rect(self.screen, red, rect)

            # Flip the display
            pygame.display.flip()

            '''
            # convert image so it can be displayed in OpenCV
            if colab_rendering:
                output.clear()
                view = pygame.surfarray.array3d(self.screen)
                view = view.transpose([1, 0, 2])
                img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                cv2_imshow(img_bgr)
            '''

            # Wait for a short time to slow down the rendering
            pygame.time.wait(25)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize positions of grips
        self._agent_location = np.array([0], dtype=int)
        self.grips = self.np_random.choice(self.n_grips, self.height)
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
