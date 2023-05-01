from gymnasium import spaces

import gymnasium as gym

import numpy as np
import random

class TamagotchiEnv(gym.Env):
    
    metadata = {"render_modes": ["terminal"]}

    def __init__(self, vocab = ["play", "sleep", "feed", "clean", "nothing"], max_msg_length=1, tau=1.0, steps_per_episode=100, render_mode=None):
        """
        Tamagotchi environment.

        Args:
        vocab (dict, optional): The vocabulary used to generate messages. Defaults to {"subject":["I", "You"], "verbs": ["play", "sleep", "feed", "clean", "nothing"]}.
        max_msg_length (int, optional): The maximum length of the messages. Defaults to 2.
        tau (float, optional): The temperature parameter for the softmax function. Defaults to 1.0.
        steps_per_episode (int, optional): The maximum number of steps per episode. Defaults to 100.
        communicate (bool, optional): Whether the agents communicates about its internal variables. Defaults to True.
        render_mode (str, optional): The render mode. Defaults to None.
        """
        
        super(TamagotchiEnv, self).__init__()
        
        # Vocabulary
        self.vocab = vocab
        self.max_msg_length = max_msg_length
        
        # Observation space: the words used to make utterances of a speficied length
        self.state_dims = np.concatenate([[len(vocab)]*(max_msg_length)], axis=None)
        
        self.n_states = np.prod(self.state_dims)
        self.observation_space = spaces.MultiDiscrete(self.state_dims)
        
        # Action space: 5 actions
        self.action_space = spaces.Discrete(5)

        # Max number of steps per episode
        self.steps_per_episode = steps_per_episode
        self.step_count = 0

        # Joy, Energy, Food, Hygiene
        self.internal_vars = np.array([100, 100, 100, 100])
        self.happiness = 50
        self.hp = 50
        self.weights = np.ones(len(self.internal_vars))
        self.required_action = self.vocab.index("nothing")
        self.tau = tau

        self.tamagotchi_msg = self.generate_message()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    

    def reset(self, seed=None, options=None):
        # print('Resetting..')
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

        When the Tamagotchi is happy, no action is required. 

        Returns:
        The index of the required action.
        """
        
        if self.happiness >= 85:
            return self.vocab.index("nothing")
        else:
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
        
        #calculate softmax over the weights with a temperature variable
        softmax_action = np.exp(self.weights / self.tau) / np.sum(np.exp(self.weights / self.tau))

        verbs = np.random.choice(self.vocab[:len(self.weights)], p=softmax_action, size=self.max_msg_length, replace=True)

        # If the Tamagotchi is happy, it will not ask for anything
        if self.required_action == self.vocab.index("nothing"):
            # create a placeholder message of the required length
            placeholder = list(np.random.choice(self.vocab[:len(self.weights)], size=self.max_msg_length, replace=True))
            # replace the first token with the required action
            placeholder[0] = "nothing"
        
            verbs = placeholder
        
        verbs = [self.vocab.index(verb) for verb in verbs]

        return np.array(verbs)
    

    def get_observation(self):
        """
        Returns the current state of the environment as a vector including the hp and message.
        """
    
        return self.tamagotchi_msg
        

    def get_info(self):
        return {}


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


    def calculate_happiness(self):
        """
        Calculate the happiness value based on the internal variables

        High values of the internal variables are good and have low weights
        Low values of the internal variables are bad and have high weights

        This way, low values have a stronger effect on the happiness score. 
        Argmax on the weights is thus the variable that requires the most attention.

        Returns:
        The sum of the weighted internal variables, shifted by 50 to be in the range [-50, 50]    
        """
        
        self.weights = np.exp(-1 * self.internal_vars / 100)
        # self.weights = -1 * self.internal_vars / 100
        
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
        return max(0, min(100, self.hp + hp_change))


    def step(self, action):
        """
        Make step in environment. When the required action is chosen, the internal variables corresponding to that action is updated positively.
        When the wrong action is chosen, all internal variables are updated negatively (but the action is still performed).

        Parameters:
        action (int): The index of the chosen action.

        Returns:
        observation (np.array): The vectorized state of the environment, this is the message uttered.
        reward (float): The reward for the chosen action, this is the happiness value.
        done (bool): Whether the episode is done, this is when the hp == 0 or when 100 steps are taken.
        info (dict): Additional information about the environment.
        """

        assert self.action_space.contains(action)

        # Update the internal variables based on the chosen action - Joy, Energy, Food, Hygiene
        if action == 0: # play
            self.internal_vars += np.array([50, -3, -3, -3])

        elif action == 1: # sleep
            self.internal_vars += np.array([-3, 50, -3, -3])
            
        elif action == 2: # feed
            self.internal_vars += np.array([-3, -3, 50, -3])

        elif action == 3: # clean
            self.internal_vars += np.array([-3, -3, -3, 50])
            
        elif action == 4: # no-op
            self.internal_vars += np.array([-5, -5, -5, -5])

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
        
        return state, reward, done, info


    def render(self, mode="terminal"):
        print(f"HP: {self.hp}, Happiness: {self.happiness}, Internal vars: {self.internal_vars}, Required action: {self.required_action}, Tamagotchi message: {self.tamagotchi_msg}")
        print("-" * 100)