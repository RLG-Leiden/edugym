#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Supermarket environment
Specifically designed for model-based reinforcement learning experiments

"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pygame
import sys

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
grey = (128, 128, 128)
red = (255, 0, 0)
yellow = (255, 255, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


class SupermarketEnv(gym.Env):
    metadata = {"render.modes": ["inline", "graphic"]}

    def __init__(self, step_timeout=0.0, noise=0.0, render_mode="graphic", use_single_dim=False):

        """
        Initialize the Supermarket environment.
        
        
        -- s ------
        |       i |       s = start
        |         |       i = item
        |------   |       e = exit
        |         |
        |  | i |  |
        |  |   |  |
        |i        |
        ------ e --
        
        
        Parameters:
            step_timeout (float): Timeout in seconds between calls to the step function. 

        """
        # Grid size
        self.width = 10
        self.height = 10

        # Observation space: (x,y) location and whether we picked up each of the three items
        self.state_dims = [self.width, self.height, 2, 2, 2]
        self.n_states = np.prod(self.state_dims)
        self.use_single_dim = use_single_dim
        if use_single_dim:
            self.observation_space = spaces.Discrete(self.n_states)
        else:
            self.observation_space = spaces.MultiDiscrete(self.state_dims)

        # Action space: up, down, left, right
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)

        # Build grid
        self._build_grid()

        # Add shopping items
        self.item1_location = [1, 8]
        self.item2_location = [7, 2]
        self.item3_location = [5, 5]

        # Reward parameters
        self.item_reward = 25
        self.final_reward = 50
        self.step_penalty = -1

        # Timeout of step function
        self.step_timeout = step_timeout
        self.last_call_to_step = None

        # Rendering
        self.pygame_initialized = False
        self.render_mode = render_mode

        # Initialize the state
        self.noise = noise
        self._build_model(noise)

    def _build_grid(self):
        # Create the grid: 0 = empty, 1 = wall
        self.grid = np.zeros((self.height, self.width))

        # Fix the outside walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Open the start and exit
        self.start_location = [0, 1]
        self.exit_location = [9, 8]
        self.grid[0, 1] = 0  # start
        self.grid[9, 8] = 0  # exit

        # Add walls
        self.grid[3, :-4] = 1  # horizontal wall
        self.grid[5:8, 3] = 1  # vertical wall 1
        self.grid[5:8, 6] = 1  # vertical wall 2

    def _build_model(self, noise=0.0):

        # Empty transition, reward and termination models
        self.p_sas = np.zeros(
            [self.n_states, self.n_actions, self.n_states]
        )  # p(s'|s,a)
        self.r_sas = np.zeros(
            [self.n_states, self.n_actions, self.n_states]
        )  # r(s,a,s')
        self.r_sas_noise = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.term_s = np.zeros(self.n_states)  # terminal(s)

        # Loop over all state-action pairs to fill the
        for s in range(self.n_states):
            for a in range(self.n_actions):

                s_vector = self.state_to_vector(s)
                (
                    y,
                    x,
                    item1,
                    item2,
                    item3,
                ) = s_vector  # Extract the current position and items collected from the state

                # First check whether we are at a terminal state
                if self.grid[y, x] == 1 or (
                    y == self.exit_location[0] and x == self.exit_location[1]
                ):
                    self.term_s[
                        s
                    ] = True  # make the state terminal when it is unreachable (stand on a wall) or the exit door.
                    self.p_sas[s, a, s] = 1.0  # make every action a self loop...
                    self.r_sas[s, a,] = np.zeros(
                        self.n_states
                    )  # ... with all rewards at zero
                else:
                    # Set all the default rewards
                    self.term_s[
                        s
                    ] = False  # make the state terminal when it is unreachable (stand on a wall) or the exit door.

                    self.r_sas[s, a,] += self.step_penalty
                    if noise > 0.0:
                        self.r_sas_noise[s, a,] = np.random.normal(
                            size=self.n_states, scale=noise
                        )

                    # Move the agent based on the action
                    y_old = y
                    x_old = x

                    if a == 0:  # Up
                        y = max(0, y - 1)
                    elif a == 1:  # Down
                        y = min(self.height - 1, y + 1)
                    elif a == 2:  # Left
                        x = max(0, x - 1)
                    elif a == 3:  # Right
                        x = min(self.width - 1, x + 1)

                    # Check whether we landed on a wall
                    if self.grid[y, x] == 1:
                        y, x = y_old, x_old  # step back to old location

                    extra_reward = 0

                    # Check whether we reached a special position
                    if (
                        y == self.item1_location[0]
                        and x == self.item1_location[1]
                        and item1 == 0
                    ):
                        item1 = 1
                        extra_reward += self.item_reward
                    elif (
                        y == self.item2_location[0]
                        and x == self.item2_location[1]
                        and item2 == 0
                    ):
                        item2 = 1
                        extra_reward += self.item_reward
                    elif (
                        y == self.item3_location[0]
                        and x == self.item3_location[1]
                        and item3 == 0
                    ):
                        item3 = 1
                        extra_reward += self.item_reward
                    elif (
                        y == self.exit_location[0] and x == self.exit_location[1]
                    ):  # Reached the exit
                        extra_reward += self.final_reward

                    # Update the state
                    s_prime_vector = np.array([y, x, item1, item2, item3])
                    s_prime = self.vector_to_state(s_prime_vector)

                    self.p_sas[s, a, s_prime] = 1.0  # update transition table
                    self.r_sas[s, a, s_prime] += extra_reward

    def reset(self, seed=None):
        """
        Reset the supermarket environment to its initial state.

        Returns:
            state (np.ndarray): The current state of the environment.
        """
        super().reset(seed=seed)

        # Create the state
        vector_state = np.array(
            [0, 1, 0, 0, 0]
        )  # start in (0,1) with none of the three items collected
        self.done = False
        self.state = self.vector_to_state(vector_state)
        obs = self.state if self.use_single_dim else np.array(self.state_to_vector(self.state))
        return obs
    def can_call_step(self):
        """
        Checks whether enough time has passed for a new call to step() (without actually calling step()).  

        Returns:
        step_can_be_called (bool): whether step() will execute (True) or not (False)

        """
        if (
            self.last_call_to_step is None
            or (time.time() - self.last_call_to_step) > self.step_timeout
        ):
            step_can_be_called = True
        else:
            step_can_be_called = False
        return step_can_be_called

    def time_till_next_possible_step(self):
        """
        Prints the minimal time left until we can call step() again. 
        """
        return max(self.step_timeout - (time.time() - self.last_call_to_step), 0)

    def vector_to_state(self, state):
        """
        This method takes a vectorized state and turns it into its unique state index.

        Parameters:
        state (np.array): The vectorized state.
        
        Returns:
        index (int): A unique identifier for the given state.
        """
        index = np.ravel_multi_index(state, dims=self.state_dims)
        return index

    def state_to_vector(self, index):
        """
        This method takes a state index and turns it into a vectorized state.

        Parameters:
        index (int): The unique identifier for the state.
        
        Returns:
        state (np.array): The vectorized state associated with the given index.
        """
        state = np.unravel_index(index, shape=self.state_dims)
        return state

    def descriptive_model(self, state, action, noise=True):
        p_sas = self.p_sas[
            state, action,
        ]
        r_sas = self.r_sas[
            state, action,
        ]
        if noise:
            r_sas = r_sas + self.r_sas_noise[state, action]
        done_s = self.term_s
        return np.copy(p_sas), np.copy(r_sas), np.copy(done_s)

    def generative_model(self, state, action, noise=True):
        next_state = np.random.choice(self.n_states, p=self.p_sas[state, action])
        reward = self.r_sas[state, action, next_state]
        if noise:
            reward = reward + self.r_sas_noise[state, action, next_state]
        done = self.term_s[next_state]
        return np.copy(next_state), np.copy(reward), np.copy(done)

    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Parameters:
            action (int): The action to take.

        Returns:
            state (np.ndarray): The current state of the environment.
            reward (float): The reward for the current step.
            done (bool): Whether the episode is over or not.
            info (dict): Any additional information about the current step.
        """

        if self.done:
            RuntimeError(
                "You call step() on a terminated environment. You need to call reset() first."
            )

        # Initialize time counter on first call
        if self.last_call_to_step is None:
            self.last_call_to_step = time.time()

        # Wait until we can call the step method
        while not self.can_call_step():
            time.sleep(0.01)

        # Update the step call moment
        self.last_call_to_step = time.time()

        # actually move
        next_state, reward, done = self.generative_model(
            self.state, action, noise=False
        )
        self.state = next_state
        self.done = done
        info = {}
        obs = next_state if self.use_single_dim else np.array(self.state_to_vector(next_state))
        return obs, reward, done, False, info

    def render(self):
        """
        Render the environment using Pygame.
        
        Parameters:
            mode (str): 'inline' for inline plotting, 'graphic' for pygame visualisation
        
        Returns:
            np.ndarray: A 3D array of the RGB values of the pixels in the window.
        """
        mode = self.render_mode
        assert mode in ["inline", "graphic"], print(
            "mode needs to be 'inline' or 'graphic'"
        )

        # Build the grid of the current situation
        # 0 = empty, 1 = wall, 2 = item1, 3 = item2, 4=item3, 5=exit, 6=agent
        render_grid = np.copy(self.grid)  # only contains the walls and exit
        y, x, item1, item2, item3 = self.state_to_vector(self.state)
        if not bool(item1):
            render_grid[self.item1_location[0], self.item1_location[1]] = 2
        if not bool(item2):
            render_grid[self.item2_location[0], self.item2_location[1]] = 3
        if not bool(item3):
            render_grid[self.item3_location[0], self.item3_location[1]] = 4
        render_grid[self.exit_location[0], self.exit_location[1]] = 5
        render_grid[y, x] = 6  # agent

        if mode == "inline":
            # Visualize the grid
            print("\n")
            for y, row_data in enumerate(render_grid):
                row = ""
                for x, cell_entry in enumerate(row_data):
                    if cell_entry == 0:
                        row += " "
                    elif cell_entry == 1:
                        row += "#"
                    elif cell_entry == 2 or cell_entry == 3 or cell_entry == 4:
                        row += "i"
                    elif cell_entry == 5:
                        row += "e"
                    elif cell_entry == 6:
                        row += "A"
                print(row)
            print("\n")

        elif mode == "graphic":

            # Initialize pygame
            if not self.pygame_initialized:
                pygame.init()
                self.cell_size = 50
                screen_width, screen_height = (
                    self.width * self.cell_size,
                    self.height * self.cell_size,
                )
                self.screen = pygame.display.set_mode([screen_width, screen_height])
                pygame.display.set_caption("Supermarket Environment")
                self.pygame_initialized = True

#            if colab_rendering:
#                output.clear()

            # Set background color
            self.screen.fill(white)

            # Draw grid
            for y in range(self.height):
                for x in range(self.width):
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    if render_grid[y][x] == 1:
                        # Brick
                        brick_color = [139, 79, 57]
                        pygame.draw.rect(self.screen, brick_color, rect)
                        pygame.draw.rect(self.screen, black, rect, 2)

                    elif render_grid[y][x] == 2:
                        # Apple
                        green_apple = (75, 122, 71)
                        pygame.draw.circle(self.screen, red, (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2), self.cell_size // 4)
                        pygame.draw.circle(self.screen, black, (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2), self.cell_size // 4,2)
                        pygame.draw.line(self.screen, green_apple, (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 3.75 ), (x * self.cell_size + self.cell_size // 1.3, y * self.cell_size + self.cell_size // 8), 5)

                    elif render_grid[y][x] == 3:
                        # Beer can
                        can_color = (200, 200, 40)
                        can_top = (150,150,150)

                        pygame.draw.ellipse(self.screen, can_color, (x * self.cell_size + 10 , y * self.cell_size + 40 , 30, 10))
                        pygame.draw.ellipse(self.screen, black, (x * self.cell_size + 10 , y * self.cell_size + 40 , 30, 10),2)
                        pygame.draw.rect(self.screen, can_color , (x * self.cell_size + 12 , y * self.cell_size + 10,28,35))
                        pygame.draw.ellipse(self.screen, can_top , (x * self.cell_size + 10 , y * self.cell_size + 5 , 30, 10))
                        pygame.draw.ellipse(self.screen, black , (x * self.cell_size + 10 , y * self.cell_size + 5 , 30, 10),2)
                        pygame.draw.line(self.screen, black, (x * self.cell_size + 10 , y * self.cell_size + 10 ), (x * self.cell_size + 10 , y * self.cell_size + 43 ), 2)
                        pygame.draw.line(self.screen, black, (x * self.cell_size + 39 , y * self.cell_size + 10 ), (x * self.cell_size + 39 , y * self.cell_size + 43 ), 2)
                        mouth_pos = ((x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size)

                    elif render_grid[y][x] == 4:
                        # Muffin
                        pygame.draw.circle(self.screen, yellow, (x * self.cell_size + 25, y * self.cell_size + 25,), 20)
                        pygame.draw.circle(self.screen, black, (x * self.cell_size + 25, y * self.cell_size + 25,), 20, 2)
                        pygame.draw.rect(self.screen, white, ( x * self.cell_size + 5, y * self.cell_size + 25, 40,20))
                        pygame.draw.line(self.screen, black, (x * self.cell_size + 5 , y * self.cell_size + 25 ), (x * self.cell_size + 45 , y * self.cell_size + 25 ), 2)
                        pygame.draw.rect(self.screen, (240, 240, 240), ( x * self.cell_size + 10, y * self.cell_size + 25, 30,20))
                        pygame.draw.rect(self.screen, black, ( x * self.cell_size + 10, y * self.cell_size + 25, 30,20), 2)

                    elif render_grid[y][x] == 5:
                        # Exit
                        pygame.draw.rect(self.screen, grey, rect)
                        pygame.draw.lines(
                            self.screen,
                            black,
                            True,
                            (
                                (x * self.cell_size + 25, y * self.cell_size),
                                (x * self.cell_size + 25, y * self.cell_size + self.cell_size),
                            ),
                            width=3,
                        )
                        pygame.draw.rect(self.screen, (75, 122, 71), (x * self.cell_size , y * self.cell_size ,50,20))
                        exit_text = pygame.font.Font(None, 25).render("Exit", True, (255, 255, 255))
                        self.screen.blit(exit_text, (x * self.cell_size + 10 , y * self.cell_size + 3))

                    elif render_grid[y][x] == 6:
                        pygame.draw.circle(
                            self.screen,
                            yellow,
                            ((x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size),
                            0.5 * (self.cell_size - 10),
                        )

                        pygame.draw.ellipse(self.screen, (0, 0, 0), ((x + 0.5) * self.cell_size - 10, (y + 0.5) * self.cell_size - 10, 5, 5),3)
                        pygame.draw.ellipse(self.screen, (0, 0, 0), ((x + 0.5) * self.cell_size + 6, (y + 0.5) * self.cell_size - 10, 5, 5),3)

                        mouth_pos = ((x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size)
                        mouth_radius = 10
                        pygame.draw.arc(self.screen, (0,0,0), pygame.Rect(mouth_pos[0]-mouth_radius, mouth_pos[1]-mouth_radius, 2*mouth_radius, 2*mouth_radius), 3.54, 5.88, 3)


            # Flip the display
            pygame.display.flip()
            # Wait for a short time to slow down the rendering
            pygame.time.wait(25)
    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        return


def test():

    render_mode = "graphic"  # 'inline'

    # Initialize the environment
    from edugym.envs.interactive import play_env
    env = SupermarketEnv(step_timeout=0.1, render_mode=render_mode)
    play_env(env, "w=up, s=down, a=left, d=right", {"w":0, "s": 1, "a": 2, "d": 3})

    env = SupermarketEnv(step_timeout=0.1, render_mode=render_mode, use_single_dim=True)
    # Difference between step(), descriptive_model() and generative_model()
    # Calling step
    print("Calling step(action) example \n")
    state = env.reset()
    for i in range(5):
        state_vector = env.state_to_vector(state)
        action = env.action_space.sample()  # sample a random action
        next_state, reward, done, truncated, info = env.step(
            action
        )  # info is only there for compatibility with default Gym environments
        next_state_vector = env.state_to_vector(next_state)
        print(
            f"Started in state: {state} (vector {state_vector}) and took action: {action}. Observed next state {next_state} (vector {next_state_vector}) with reward: {reward} and a done (termination) flag: {done}"
        )
        state = next_state  # set the state to the new observation for the next round

    print("\nCalling generative_model(state,action) example \n")
    # Calling model
    for i in range(5):
        state = (
            env.observation_space.sample()
        )  # sample random state, since model() can be called from any state
        state_vector = env.state_to_vector(state)
        action = env.action_space.sample()
        next_state, reward, done = env.generative_model(
            state, action
        )  # this needs both a state and action as input, as opposed to step()
        next_state_vector = env.state_to_vector(next_state)
        print(
            f"When in state : {state} (vector {state_vector})  and we take action: {action}, the next state would be {next_state} (vector {next_state_vector}) with reward: {reward} and a done (termination) flag: {done}"
        )

    print("\nCalling descriptive_model(state,action) example \n")
    # Calling model
    for i in range(1):
        state = (
            env.observation_space.sample()
        )  # sample random state, since model() can be called from any state
        state_vector = env.state_to_vector(state)
        action = env.action_space.sample()
        probs, rewards, done = env.descriptive_model(
            state, action
        )  # this needs both a state and action as input, as opposed to step()
        next_state_vector = env.state_to_vector(next_state)
        print(
            f"When in state : {state} (vector {state_vector}) and we take action: {action}, the probability of each possible next state is {probs} with reward: {rewards} and done (termination) flags: {done}"
        )


if __name__ == "__main__":
    test()
