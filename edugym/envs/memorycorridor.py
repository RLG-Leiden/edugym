import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box
import pygame


class MemoryCorridorEnv(gymnasium.Env):
    metadata = {"render_modes": ["terminal", "graphic"]}

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

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Generate a new set of self.num_doors doors and the sequence of the correct doors
        self._correct_door_path = [self.np_random.integers(self.num_doors, size=1)[0]]
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
            self._correct_door_path.append(self.np_random.integers(self.num_doors, size=1)[0])
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
        elif self.render_mode == "graphic":
            # Define door dimensions
            door_width = 200
            door_height = 400
            door_x = door_width // 2
            door_y = 100

            # Define the colors for the room
            wall_color = (127, 176, 105)
            floor_color = (230, 170, 104)
            ceiling_color = wall_color

            # define the colors for the door
            door_border_color = (29, 26, 5)
            door_color = (202, 60, 37)
            knob_color = (255, 255, 255)
            knob_shade_color = (200, 200, 200)
            window_color = (208, 233, 241)
            windows_color_border = door_border_color

            num_doors = self.num_doors
            obs = self._get_obs()
            info = self._get_info()

            if not self.pygame_initialized:
                pygame.init()
                self.screen_width = self.num_doors * (door_width + door_x) + door_x
                self.screen_height = door_height + door_y + (door_height // 4)
                self.screen = pygame.display.set_mode(
                    (
                        self.screen_width,
                        self.screen_height,
                    )
                )
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("Arial", 20)
                pygame.display.set_caption("MemoryCorridorEnv PyGame Visualization")
                self.pygame_initialized = True

            def draw_door(d, x):
                door_surface_rect = pygame.Rect(x, door_y, door_width, door_height)

                # Draw the door surface
                door_surface = pygame.Surface((door_width, door_height))
                door_surface.fill(door_border_color)

                # use the surface as the border and draw the door surface on it
                door_rect = pygame.Rect(2, 2, door_width - 4, door_height)
                pygame.draw.rect(door_surface, door_color, door_rect)

                # Draw the windows on the door incl border
                window_width = door_width - 20
                window_height = door_height // 3
                window_rect = pygame.Rect(10, 10, window_width, window_height)
                pygame.draw.rect(door_surface, windows_color_border, window_rect)
                window_rect = pygame.Rect(12, 12, window_width - 4, window_height - 4)
                pygame.draw.rect(door_surface, window_color, window_rect)

                if not (0 <= obs < num_doors):
                    # draw ? in the window when no information in the observation
                    font_size = window_height // 2
                    font = pygame.font.SysFont(None, font_size)
                    text_surface = font.render("?", True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(door_width // 2, window_height // 2))
                    door_surface.blit(text_surface, text_rect)

                # Draw the door knob
                knob_radius = door_width // 25
                knob_x = door_width // 4
                knob_y = door_height // 2

                pygame.draw.circle(door_surface, knob_color, (knob_x, knob_y), knob_radius)

                # Draw the shining shade on the door knob
                shade_radius = knob_radius - 2
                shade_rect = pygame.Rect(knob_x - shade_radius - 1, knob_y - shade_radius - 1, shade_radius * 2, shade_radius * 2)
                pygame.draw.ellipse(door_surface, knob_shade_color, shade_rect)

                # Render the number of the door
                font_size = door_height // 10
                font_color = (255, 255, 255)  # White color for the text
                font = pygame.freetype.SysFont(None, font_size)
                text_surface, _ = font.render(str(d), font_color)

                text_x = door_width // 2 - text_surface.get_width() // 2
                text_y = door_height // 2 - text_surface.get_height() // 2

                # Draw the text onto the door surface
                door_surface.blit(text_surface, (text_x, text_y))

                # Draw the door on the screen
                self.screen.blit(door_surface, door_surface_rect)

            def draw_light(d, x):
                light_color_on = (255, 255, 0)  # Yellow color when the light is on
                light_color_off = (128, 128, 128)  # Gray color when the light is off

                light_color = light_color_on if d == obs else light_color_off

                light_y = door_y // 3
                light_width = door_width // 4
                light_height = door_height // 6
                x = x - light_width // 2

                pygame.draw.ellipse(self.screen, (0, 0, 0), (x - 2, light_y - 2, light_height + 4, light_width + 4))  # border of the light
                pygame.draw.ellipse(self.screen, light_color, (x, light_y, light_height, light_width))

                # Draw a smaller white circle inside the light for a highlight effect
                if d == obs:
                    highlight_radius = light_height // 7
                    highlight_color = (255, 255, 255)  # White color for the highlight
                    pygame.draw.circle(self.screen, highlight_color, (x + light_width // 2, light_y + light_height // 2), highlight_radius)

                # draw grates in front of the light
                pygame.draw.line(self.screen, (0, 0, 0), (x, light_y + light_height // 3), (x + light_height, light_y + light_height // 3), 2)

            # Fill the screen surface with the room colors (wall, floor, ceiling)
            self.screen.fill(wall_color)
            pygame.draw.rect(self.screen, floor_color, (0, door_y + door_height, self.screen_width, self.screen_height))
            pygame.draw.rect(self.screen, ceiling_color, (0, 0, self.screen_width, door_y - 50))

            # draw a new surface and rotate it to create perspective
            new_surface = pygame.Surface((door_width, door_height), pygame.SRCALPHA)
            new_surface.fill(wall_color)
            new_surface = pygame.transform.rotate(new_surface, -45)
            self.screen.blit(new_surface, (0 - door_width, self.screen_height - door_height))
            new_surface = pygame.transform.rotate(new_surface, 90)
            self.screen.blit(new_surface, (self.screen_width - door_width, self.screen_height - door_height))

            for d in range(num_doors):  # doors and lights
                door_x_pos = door_x + d * (door_width + door_width // 2)
                light_x_pos = door_x_pos + door_width // 2
                draw_door(d, door_x_pos)
                draw_light(d, light_x_pos)

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


def test():
    render_mode = "graphic"  # 'inline'
    # Initialize the environment
    from edugym.envs.interactive import play_env, play_env_terminal
    env = MemoryCorridorEnv(render_mode=render_mode)
    play_env(env, "0=First Door, 1=Second Door, 2=Third Door", {pygame.K_0:0, pygame.K_1: 1, pygame.K_2: 2})
    # play_env_terminal(env, "1=First Door, 2=Second Door, 3=Third Door", {"1":0, "2": 1, "3": 2})

if __name__ == "__main__":
    test()