import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
import numpy as np
import random

colab_rendering = "google.colab" in sys.modules
if colab_rendering:
    import cv2
    from google.colab.patches import cv2_imshow
    from google.colab import output
    import os

    # set SDL to use the dummy NULL video driver, so it doesn't need a windowing system.
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class TamagotchiEnv(gym.Env):

    metadata = {"render_modes": ["terminal", "graphic", "none"]}

    def __init__(
        self,
        vocab=["play", "sleep", "feed", "clean"],
        max_msg_length=1,
        tau=1.0,
        steps_per_episode=100,
        render_mode=None,
        seed=None,
    ):
        """
        Tamagotchi environment.

        Args:
        vocab (dict, optional): The vocabulary used to generate messages. Defaults to ["play", "sleep", "feed", "clean"].
        max_msg_length (int, optional): The maximum length of the messages. Defaults to 1.
        tau (float, optional): The temperature parameter for the softmax function, this determines the informativeness of the utterances. Defaults to 1.0.
        steps_per_episode (int, optional): The maximum number of steps per episode. Defaults to 100.
        communicate (bool, optional): Whether the agents communicates about its internal variables. Defaults to True.
        render_mode (str, optional): The render mode. Defaults to None.
        """

        super(TamagotchiEnv, self).__init__()
        self.seed(seed)

        # Vocabulary
        self.vocab = vocab
        self.max_msg_length = max_msg_length

        # Max number of steps per episode
        self.steps_per_episode = steps_per_episode
        self.step_count = 0

        # Joy, Energy, Food, Hygiene
        self.internal_vars = np.array([100, 100, 100, 100])
        self.happiness = 100
        self.hp = 100
        self.weights = np.ones(len(self.internal_vars))
        self.required_action = self.vocab.index("play")
        self.tau = tau

        # Observation space: the words used to make utterances of a speficied length
        self.state_dims = np.concatenate(
            [[101], [len(self.vocab)] * (max_msg_length)], axis=None
        )

        self.n_states = np.prod(self.state_dims)
        self.observation_space = spaces.MultiDiscrete(self.state_dims)

        # Action space: 4 actions
        self.action_space = spaces.Discrete(4)

        self.tamagotchi_msg = self.generate_message()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.pygame_initialized = False

    def seed(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
        seed (int, optional): The seed for the random number generator. Defaults to None.
        options (dict, optional): The options for the environment. Defaults to None.

        Returns:
        state (np.array): The initial state of the environment.

        """

        super().reset(seed=seed)

        self.step_count = 0
        self.internal_vars = np.array([100, 100, 100, 100])
        self.happiness = 100
        self.hp = 100
        self.required_action = self.get_required_action()
        self.tamagotchi_msg = self.generate_message()

        return self.vector_to_state(self.get_observation())

    def get_required_action(self):
        """
        Calculates the required action based on the internal variables.
        The required action is the variable with the highest weight.

        Returns:
        The index of the required action.
        """

        # get the highest weights, in case of multiple highest weights, select one at random
        highest_weights = np.argwhere(self.weights == np.amax(self.weights)).flatten()

        if len(highest_weights) > 1:
            # multiple variables require attention, return a random index from the highest weights
            return random.choice(highest_weights)
        else:
            # return the index of the highest weight
            return highest_weights[0]

    def generate_message(self):
        """
        Generates a message from the Tamagotchi, based on the weights it has for its internal states.
        The message is a list of indeces that correspond to the subject and the verbs of the message.
        Tokens are selected by sampling from a temperature softmax distribution over the weights.
        Higher temperatures result in more uniform distributions, thus more noise in the messages.

        Returns:
        Array of indeces that correspond to tokens of the message.
        """

        # calculate softmax over the weights with a temperature variable
        softmax_action = np.exp(self.weights / self.tau) / np.sum(
            np.exp(self.weights / self.tau)
        )

        verbs = np.random.choice(
            self.vocab[: len(self.weights)],
            p=softmax_action,
            size=self.max_msg_length,
            replace=False,
        )

        verbs = [self.vocab.index(verb) for verb in verbs]

        return np.array(verbs)

    def get_observation(self):
        """
        Returns the current state of the environment as a vector including the hp and message.
        """

        return np.concatenate([[self.hp], self.tamagotchi_msg])

    def get_info(self):
        return {}

    def vector_to_state(self, state):
        """
        This method takes a vectorized state and turns it into its unique state index.

        Args:
        state (np.array): The vectorized state.

        Returns:
        index (int): A unique identifier for the given state.
        """

        index = np.ravel_multi_index(state, dims=self.state_dims)
        return index

    def calculate_happiness(self):
        """
        Calculate the happiness value based on the internal variables

        High values of the internal variables are good and have low weights
        Low values of the internal variables are bad and have high weights

        This way, low values have a stronger effect on the happiness score.
        Argmax on the weights is thus the variable that requires the most attention.

        Returns:
        The sum of the weighted internal variables.
        """

        self.weights = np.exp(-1 * self.internal_vars / 100)

        return np.sum((self.internal_vars - 50) * self.weights)

    def update_hp(self):
        """
        Update the hp value based on the happiness value

        Returns:
        The new hp value, capped to the range [0, 100]
        """
        # Calculate the hp change based on the happiness value
        hp_change = self.happiness / 10

        # return the hp value and cap it to the range [0, 100]
        return int(max(0, min(100, self.hp + hp_change)))

    def step(self, action):
        """
        Make step in environment. When the required action is chosen,
        the internal variables corresponding to that action is updated positively.
        When the wrong action is chosen, all internal variables are updated
        negatively (but the action is still performed).

        Args:
        action (int): The index of the chosen action.

        Returns:
        observation (np.array): The vectorized state of the environment, this is the message uttered.
        reward (float): The reward for the chosen action, this is the happiness value.
        done (bool): Whether the episode is done, this is when the hp == 0 or when 100 steps are taken.
        info (dict): Additional information about the environment.
        """

        assert self.action_space.contains(action)

        # Update the internal variables based on the chosen action - Joy, Energy, Food, Hygiene
        if action == 0:  # play
            self.internal_vars += np.array([30, -5, -5, -5])

        elif action == 1:  # sleep
            self.internal_vars += np.array([-5, 30, -5, -5])

        elif action == 2:  # feed
            self.internal_vars += np.array([-5, -5, 30, -5])

        elif action == 3:  # clean
            self.internal_vars += np.array([-5, -5, -5, 30])

        if action != self.required_action:
            # If the wrong action is chosen, the internal variables are updated negatively
            self.internal_vars += np.array([-10, -10, -10, -10])

        # Cap the internal variables to [0, 100] range
        self.internal_vars = np.clip(self.internal_vars, 0, 100)

        # Update the happiness and hp values. This also determined the reward value.
        self.happiness = self.calculate_happiness()
        self.hp = self.update_hp()

        # Update the required action after the internal variables have been updated
        self.required_action = self.get_required_action()

        # Generate a new Tamagotchi message
        self.tamagotchi_msg = self.generate_message()

        observation_vector = self.get_observation()
        self.step_count += 1

        # Return the new observation, reward and done flag
        state = self.vector_to_state(observation_vector)
        info = self.get_info()
        reward = self.happiness
        done = (self.hp == 0) or (self.step_count == self.steps_per_episode)

        return state, reward, done, False, info

    def render(self):
        if self.render_mode == "terminal":
            print(
                f"HP: {self.hp}, Happiness: {self.happiness}, Internal vars: {self.internal_vars}, Required action: {self.required_action}, Tamagotchi message: {self.tamagotchi_msg}"
            )
            print("-" * 100)
        elif self.render_mode == "graphic":
            if not self.pygame_initialized:
                pygame.init()
                self.screen = pygame.display.set_mode((700, 800))
                self.font = pygame.font.SysFont("Raleway", 30)
                self.font2 = pygame.font.SysFont("Raleway", 30, italic=True)
                # self.font = pygame.font.Font(None, 30)
                self.clock = pygame.time.Clock()

                # Define button positions
                self.play_button = pygame.Rect(150, 470, 100, 70)
                self.sleep_button = pygame.Rect(460, 470, 100, 70)
                self.feed_button = pygame.Rect(240, 550, 100, 70)
                self.clean_button = pygame.Rect(370, 550, 100, 70)
                pygame.display.set_caption("Tamagotchi")
                self.pygame_initialized = True

            # Fill screen with white
            self.screen.fill((255, 255, 255))

            # Draw Tamagotchi background
            tamagotchi_colour = (80, 140, 80)
            black = (40, 40, 40)
            pygame.draw.ellipse(self.screen, tamagotchi_colour, (325, 70, 50, 50), 15)
            pygame.draw.ellipse(self.screen, black, (325, 70, 50, 50), 5)
            pygame.draw.ellipse(self.screen, black, (340, 85, 20, 20), 5)

            pygame.draw.ellipse(self.screen, tamagotchi_colour, (200, 100, 300, 300))
            pygame.draw.ellipse(self.screen, black, (200, 100, 300, 300), 5)
            pygame.draw.ellipse(self.screen, tamagotchi_colour, (100, 200, 500, 500))
            pygame.draw.ellipse(self.screen, black, (100, 200, 500, 500), 5)

            vertices = [(220, 180), (122, 350), (400, 350), (400, 180)]
            pygame.draw.polygon(self.screen, tamagotchi_colour, vertices)
            pygame.draw.line(self.screen, black, (220, 180), (122, 350), 5)

            vertices = [(480, 180), (577, 350), (300, 350), (300, 180)]
            pygame.draw.polygon(self.screen, tamagotchi_colour, vertices)
            pygame.draw.line(self.screen, black, (480, 180), (577, 350), 5)

            # Side pieces
            side_colour = (240, 220, 160)
            vertices = [(200, 340), (200, 360), (122, 360), (131, 340)]
            pygame.draw.polygon(self.screen, side_colour, vertices)

            vertices = [(500, 340), (500, 360), (577, 360), (567, 340)]
            pygame.draw.polygon(self.screen, side_colour, vertices)

            # Draw screen
            pygame.draw.rect(self.screen, (255, 255, 255), (200, 250, 300, 200))
            pygame.draw.rect(self.screen, black, (200, 250, 300, 200), 2)

            # Draw smiley
            smiley_colour = (243, 188, 87)
            pygame.draw.ellipse(self.screen, smiley_colour, (275, 275, 150, 150))
            # Eyes
            pygame.draw.ellipse(self.screen, (0, 0, 0), (315, 325, 15, 15), 10)
            pygame.draw.ellipse(self.screen, (0, 0, 0), (365, 325, 15, 15), 10)
            # Mouth
            mouth_color = (0, 0, 0)
            if self.hp > 75:
                mouth_pos = (350, 360)
                mouth_radius = 40
                pygame.draw.arc(
                    self.screen,
                    mouth_color,
                    pygame.Rect(
                        mouth_pos[0] - mouth_radius,
                        mouth_pos[1] - mouth_radius,
                        2 * mouth_radius,
                        2 * mouth_radius,
                    ),
                    3.54,
                    5.88,
                    5,
                )
            elif self.hp > 50:
                pygame.draw.line(self.screen, mouth_color, (320, 380), (380, 380), 5)
            else:
                mouth_pos = (350, 400)
                mouth_radius = 40
                pygame.draw.arc(
                    self.screen,
                    mouth_color,
                    pygame.Rect(
                        mouth_pos[0] - mouth_radius,
                        mouth_pos[1] - mouth_radius,
                        2 * mouth_radius,
                        2 * mouth_radius,
                    ),
                    0.4,
                    2.74,
                    5,
                )

            # HP text
            hp_text = self.font.render(f"HP: {self.hp}", True, (0, 0, 0))
            self.screen.blit(hp_text, (210, 260))

            # Utterance
            utterance_color = (50, 50, 50)
            utterance = [self.vocab[i] for i in self.tamagotchi_msg]
            utterance = '"' + " ".join(utterance) + '"'
            utterance_text = self.font2.render(f"{utterance}", True, utterance_color)
            self.screen.blit(utterance_text, (415, 410))

            # Draw buttons
            grey = (235, 235, 235)
            pygame.draw.ellipse(self.screen, grey, self.play_button)
            # pygame.draw.ellipse(self.screen, black, self.play_button, 3)

            pygame.draw.ellipse(self.screen, grey, self.sleep_button)
            # pygame.draw.ellipse(self.screen, black, self.sleep_button, 3)

            pygame.draw.ellipse(self.screen, grey, self.feed_button)
            # pygame.draw.ellipse(self.screen, black, self.feed_button, 3)

            pygame.draw.ellipse(self.screen, grey, self.clean_button)
            # pygame.draw.ellipse(self.screen, black, self.clean_button, 3)

            # Add text to buttons
            play_text = self.font.render("Play", True, (0, 0, 0))
            self.screen.blit(
                play_text, (self.play_button.x + 25, self.play_button.y + 25)
            )

            sleep_text = self.font.render("Sleep", True, (0, 0, 0))
            self.screen.blit(
                sleep_text, (self.sleep_button.x + 20, self.sleep_button.y + 25)
            )

            feed_text = self.font.render("Feed", True, (0, 0, 0))
            self.screen.blit(
                feed_text, (self.feed_button.x + 25, self.feed_button.y + 25)
            )

            clean_text = self.font.render("Clean", True, (0, 0, 0))
            self.screen.blit(
                clean_text, (self.clean_button.x + 20, self.clean_button.y + 25)
            )

            # Update display
            pygame.display.update()

            # Flip the display
            pygame.display.flip()

            # convert image so it can be displayed in OpenCV
            if colab_rendering:
                output.clear()
                view = pygame.surfarray.array3d(self.screen)
                view = view.transpose([1, 0, 2])
                img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                cv2_imshow(img_bgr)

            # Wait for a short time to slow down the rendering
            pygame.time.wait(25)

        elif self.render_mode == "none":
            pass
        else:
            raise NotImplementedError(
                "render_mode {} not implemented".format(self.render_mode)
            )

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        return


def test():
    from edugym.envs.interactive import play_env, play_env_terminal

    # Initialize the environment
    render_mode = "graphic"  # 'terminal', 'none'
    env = TamagotchiEnv(render_mode=render_mode)

    if render_mode == "graphic":
        play_env(
            env,
            "play=p, sleep=s, clean=c, feed=f",
            {pygame.K_p: 0, pygame.K_s: 1, pygame.K_c: 2, pygame.K_f: 3},
        )
    elif render_mode == "terminal":
        play_env_terminal(
            env, "play=p, sleep=s, clean=c, feed=f", {"p": 0, "s": 1, "c": 2, "f": 3}
        )


if __name__ == "__main__":
    test()
