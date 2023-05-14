import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box
import pygame


class MemoryCorridorEnv(gymnasium.Env):
    metadata = {"render_modes": ["terminal", "pygame"]}

    def __init__(self, render_mode=None, num_doors=3, verbose=0):
        self.num_doors = num_doors

        # Observation space is the index of the door that is currently the correct door
        # or num_doors if the correct door is hidden,
        self.observation_space = Discrete(self.num_doors + 1)

        # We have num_doors actions, corresponding to "open door 0", "open door 1", "open door 2" etc.
        self.action_space = Discrete(self.num_doors)

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.pygame_initialized = False

        self.verbose = verbose

    def _get_obs(self):
        # Show correct door if it is the last of the sequence, else all doors are hidden (num_doors)
        obs = self.num_doors

        if self._on_last_door:
            obs = self._final_correct_door

        return obs

    def _get_info(self):
        return {
            "successfull_opens": self._successfull_opens,
            "depth": self._depth,
            "max_depth": len(self._correct_door_path),
        }

    def _compute_reward(self):
        if self._terminated:
            return 0
        return 1

    def reset(self):
        # Generate a new set of self.num_doors doors and the sequence of the correct doors
        self._correct_door_path = [np.random.randint(self.num_doors)]
        self._depth = 1
        self._successfull_opens = 0
        self._terminated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        assert action in self.action_space, "Invalid action: %s" % action

        if action != self._correct_door_path[self._depth - 1]:
            self._terminated = True
            if self.verbose > 0:
                print("=== GAME OVER ===")

        elif self._on_last_door:
            # Last door of sequence, generate a new final door and start from beginning of sequence
            self._correct_door_path.append(np.random.randint(self.num_doors))
            self._depth = 1
            self._successfull_opens += 1
            if self.verbose > 0:
                print("=== DOOR OPENED - NEW SEQUENCE ===")

        else:
            self._depth += 1
            self._successfull_opens += 1
            if self.verbose > 0:
                print("=== DOOR OPENED ===")

        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._terminated
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "terminal":
            info = self._get_info()
            print()
            print("Depth: %d/%d" % (info["depth"], info["max_depth"]), "| Succesfull opens:", self._successfull_opens)
            door_w = 7
            door_h = 8
            for d in range(self.num_doors):
                door_type = "="
                if self._on_last_door and d == self._final_correct_door:
                    door_type = "#"
                print(door_type * door_w, end="   ")
            print()
            for h in range(door_h - 1):
                for d in range(self.num_doors):
                    door_type = "|"
                    if self._on_last_door and d == self._final_correct_door:
                        door_type = "#"
                    if h == door_h // 3:
                        print(door_type + " " * ((door_w - 2) // 2) + str(d) + " " * ((door_w - 2) // 2) + door_type, end="   ")
                    else:
                        print(door_type + " " * (door_w - 2) + door_type, end="   ")
                print()
            print()
        elif self.render_mode == "pygame":
            if not self.pygame_initialized:
                # Define door dimensions
                self.door_w = 100
                self.door_h = 200
                self.door_x = 100
                self.door_y = 100

                pygame.init()
                self.screen = pygame.display.set_mode(
                    (
                        self.num_doors * (self.door_w + self.door_x),
                        self.door_h + self.door_y + 50,
                    )
                )
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("Arial", 20)
                pygame.display.set_caption("MemoryCorridorEnv PyGame Visualization")
                self.pygame_initialized = True

            num_doors = self.num_doors
            obs = self._get_obs()
            info = self._get_info()

            # Define door colors
            white = (255, 255, 255)
            dark_brown = (139, 69, 19)
            correct_door_color = (255, 215, 0)

            # Clear the screen
            self.screen.fill((0, 0, 0))
            border_width = 4

            for d in range(num_doors):
                if obs == d:
                    door_color = correct_door_color
                else:
                    door_color = white
                # Draw the door rectangle with a light brown color
                door_rect = pygame.Rect(
                    self.door_x + d * (self.door_w + 50),
                    self.door_y,
                    self.door_w,
                    self.door_h - border_width,
                )
                pygame.draw.rect(self.screen, door_color, door_rect)

                # Draw the door border with a dark brown color
                border_rect = pygame.Rect(
                    self.door_x + d * (self.door_w + 50),
                    self.door_y,
                    self.door_w,
                    self.door_h,
                )
                pygame.draw.rect(self.screen, dark_brown, border_rect, border_width)

                # Draw the door knob
                knob_x = self.door_x + d * (self.door_w + 50) + self.door_w // 2 - (self.door_w // 3)
                knob_y = self.door_y + self.door_h // 2 - 10
                knob_rect = pygame.Rect(knob_x, knob_y, 20, 10)
                pygame.draw.ellipse(self.screen, (35, 35, 35), knob_rect)

                # Render the door number on the door
                font_size = 36
                font_obj = pygame.font.Font(None, font_size)
                door_number = str(d)
                text_surface = font_obj.render(door_number, True, (0, 0, 0))  # Black color for the text
                text_rect = text_surface.get_rect()
                text_rect.center = (self.door_x + d * (self.door_w + 50) + self.door_w // 2, self.door_y + self.door_h // 4)
                self.screen.blit(text_surface, text_rect)

            # Update the Pygame window
            pygame.display.flip()

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        return

    @property
    def _final_correct_door(self):
        return self._correct_door_path[-1]

    @property
    def _on_last_door(self):
        return self._depth == len(self._correct_door_path)


if __name__ == "__main__":
    env = MemoryCorridorEnv(render_mode="terminal")
    env.reset()

    # Play with user input
    while True:
        env.render()
        action = int(input("Action: "))
        env.step(action)
        if env._terminated:
            env.reset()
