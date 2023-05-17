from __future__ import annotations
import sys
from typing import SupportsFloat, Any, Dict

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import pygame

class Study(gym.Env):
    metadata = {"render_modes": ["terminal", "graphic", None]}

    def __init__(self, total_days: int = 10, n_actions: int = 5, lectures_days: int = 3, lectures_needed: int = 2, energy_needed: int = 1, seed: int = None, action_reward_noise_mean = 0.5, action_reward_noise_sigma = 0.05, render_mode = None):
        assert n_actions >= 3, "There must be at least the actions 'study', 'sleep', 'go_out'"
        assert total_days > lectures_days, "The number total days should at least the same or larger than the number of lectures "
        assert lectures_days >= lectures_needed, "The number lectures days should at least the same or larger than the number of lectures needed "
        assert total_days > lectures_needed + energy_needed, "The number total days should at least the same or larger than the number of (lectures + energy) needed "

        self.knowledge = 0
        self.energy = 0
        self.total_days = total_days
        self.lectures_needed = lectures_needed
        self.energy_needed = energy_needed


        # determine which of the days have a lecture, randomly initilized
        lectures = np.random.choice(total_days-1, lectures_days, replace=False)
        self.lecture_days = [1 if i in lectures else 0 for i in range(total_days)]
        # day counter
        self.current_day = 0
        # what actions have been done so far. Needed for rendering
        self.action_history = [None for _ in range(total_days)]

        self.action_space = Discrete(n_actions)
        self.observation_space = MultiDiscrete([5, 5, total_days]) 

        self.action_reward_noise_mean = action_reward_noise_mean
        self.action_reward_noise_sigma = action_reward_noise_sigma

        # set the means and noise levels for each action
        #                             sleep, go out, study 
        base_action_rewards = np.array([-0.5, 0.5, -0.5,])
        if self.action_space.n > 3: 
          self.action_rewards = np.append(base_action_rewards,np.random.uniform(-action_reward_noise_mean,action_reward_noise_mean,self.action_space.n-3))
        else: 
          self.action_rewards = base_action_rewards
        
        # Setup rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode        
        self.pygame_initialized = False
        self.cell_width = int(60 * 21 / total_days)
        self.cell_height = 60
        self.rendergrid = [self.lecture_days]
        self.rendergrid.append([0 for _ in range(len(self.lecture_days))])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.knowledge = 0
        self.energy = 0
        self.current_day = 0
        self.action_history = [None for _ in range(self.total_days)]

        return np.array([0,0,self.current_day]), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        # update action history
        self.action_history[self.current_day] = action

        # modify internal variables
        if action == 0: # sleep
            if self.energy < 4:
                self.energy += 1
        elif action == 1:  
            if self.energy > 0: # go out
                self.energy -= 1
        elif action == 2:  # study
            if self.lecture_days[self.current_day] == 1: 
              self.knowledge += 1

        # determine reward
        if action == 2 and self.current_day == self.total_days - 1 and self.knowledge >= self.lectures_needed and self.energy > self.energy_needed:
          reward = 10
        else: 
          reward = self.np_random.normal(self.action_rewards[action], self.action_reward_noise_sigma)

        # termination
        terminated = False
        if self.current_day == self.total_days - 1:
          terminated = True

        # Increase day counter
        self.current_day += 1

        return np.array([self.knowledge, self.energy, self.current_day]), reward, terminated, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "terminal":
            for i in range(self.total_days):  # print lecture schedule
                if self.lecture_days[i] == 1:
                    print('L', end=' ')  # lecture
                else:
                    print('-', end=' ')  # anything else
            print('E')  # Exam

            for i in range(self.total_days):
                if self.action_history[i] == 3:  # Studied S
                    print('S', end=' ')
                elif self.action_history[i] == 1:  # Sleep ZzZ...
                    print('Z', end=' ')
                elif self.action_history[i] == 2:  # out O
                    print('O', end=' ')
                elif self.action_history[i] == None:  # no action taken yet
                    print('-', end=' ')
                else:
                    print('*', end=' ')  # any other action was taken
            print()

        elif self.render_mode == 'graphic':
            # Initialize pygame
            if not self.pygame_initialized:
                pygame.init()
                screen_width = 6 * self.cell_width
                screen_height = (int(self.total_days/5.1)+1 ) * 2 * self.cell_height
                self.screen = pygame.display.set_mode([screen_width, screen_height])
                pygame.display.set_caption("Study Environment")
                self.font = pygame.font.SysFont(None, 24)
                self.img_lecture = self.font.render('Lecture', True, (100, 100, 100))
                self.img_other = self.font.render('Free', True, (100, 100, 100))
                self.img_exam = self.font.render('Exam', True, (100, 100, 100))
                self.imgs_days = [self.font.render(day, True, (0, 0, 0)) for day in ['Mo', 'Tu', 'We', 'Th', 'Fr']]

                self.pygame_initialized = True

            # Set background color
            self.screen.fill((255, 255, 255))

            day_text = self.font.render('Day', True, (0, 0, 0))
            self.screen.blit(day_text,
                  (self.cell_width / 2, self.cell_height / 4))
            activity_text = self.font.render('Activity', True, (0, 0, 0))
            self.screen.blit(activity_text,
                  (self.cell_width / 2, self.cell_height + self.cell_height / 4))
            pygame.draw.line(self.screen, (0,0,0), (self.cell_width,0), (self.cell_width,self.cell_height*2), 5)


            for row in range(len(self.rendergrid)):
                for col in range(len(self.rendergrid[0])):
                    render_row_counter = int(col/5) 
                    if row == 0:
                        self.screen.blit(self.imgs_days[col%5],
                                             ((col%5) *  self.cell_width  + 3* self.cell_width / 2,2* render_row_counter*self.cell_height + 3*self.cell_height / 8))
                        if self.rendergrid[row][col] == 1:
                            self.screen.blit(self.img_lecture,
                                             ((col%5) *  self.cell_width  + 3* self.cell_width / 2,2* render_row_counter*self.cell_height + 6*self.cell_height / 8))
                        elif col == len(self.rendergrid[0]) - 1:
                            self.screen.blit(self.img_exam,
                                             ((col%5) *  self.cell_width  + 3* self.cell_width / 2,2* render_row_counter*self.cell_height + 6*self.cell_height / 8))
                        else:
                            self.screen.blit(self.img_other,
                                             ((col%5) *  self.cell_width  + 3* self.cell_width / 2,2* render_row_counter*self.cell_height + 6*self.cell_height / 8))

                    elif row == 1:
                        if col < self.current_day:
                            if self.rendergrid[0][col] == 1 and self.action_history[col] == 2:  # studied at the right moment
                                curr_img = self.font.render('Study', True, (127, 176, 105))
                            elif self.rendergrid[0][col] != 1 and self.action_history[col] == 2:  # studied at the wrong moment
                                curr_img = self.font.render('Study', True, (230, 170, 104))
                            elif self.action_history[col] == 0:  # sleeping
                                curr_img = self.font.render('ZzZ', True, (127, 176, 105))
                            elif self.action_history[col] == 1:  # going out
                                curr_img = self.font.render('Go Out', True, (202, 60, 37))
                            else:
                                curr_img = self.font.render('Other', True, (230, 170, 104))
                            self.screen.blit(curr_img,
                                             ((col%5+1) * self.cell_width + self.cell_width / 2,
                                              2*render_row_counter*self.cell_height + self.cell_height / 4 + self.cell_height))
                        if col == self.current_day:
                          # Plot agent
                          agent_x = (col%5+1) * self.cell_width + self.cell_width / 2
                          agent_y = 2*render_row_counter*self.cell_height + self.cell_height / 4 + self.cell_height - 10
                          mouth_pos = (agent_x+18, agent_y + 18)
                          mouth_radius = 10
                          pygame.draw.ellipse(self.screen, (220, 220, 160), (agent_x, agent_y , 35, 35))
                          pygame.draw.ellipse(self.screen, (0, 0, 0), (agent_x + 10, agent_y +10 , 5, 5),3)
                          pygame.draw.ellipse(self.screen, (0, 0, 0), (agent_x + 20, agent_y +10 , 5, 5),3)
                          pygame.draw.arc(self.screen, (0, 0, 0),  pygame.Rect(mouth_pos[0]-mouth_radius, mouth_pos[1]-mouth_radius, 2*mouth_radius, 2*mouth_radius), 3.54, 5.88, 3)

            pygame.display.update()

            pygame.time.wait(25)
        elif self.render_mode == None:
          pass
        else:
            raise ValueError("render_mode {} not recognized".format(self.render_mode))

def test():
    render_mode = "graphic"  # 'inline'
    # Initialize the environment
    from edugym.envs.interactive import play_env
    env = Study(render_mode=render_mode)
    play_env(env, "s=study, z=sleep, g=go_out, o=other", {"s":2, "z": 0, "g": 1, "o":3})

if __name__ == "__main__":
    test()