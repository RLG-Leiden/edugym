import numpy as np
import random
from typing import List, Tuple, Any
from gymnasium import Env


class QLearningAgent:
    def __init__(self, env: Env, update_depth: int = 1, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.0001):
        """
        Initialization of the n-step Q-learning agent.

        :param env: The RL environment.
        :param update_depth: Number of steps to include in the n-step update.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Initial epsilon
        :param epsilon_decay: Change of epsilon per time step.
        """
        assert update_depth > 0, "At least one step is needed to update the Q values"

        self.env = env
        self.update_depth = update_depth
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((env.observation_space.nvec[0], env.observation_space.nvec[1]+1, env.action_space.n))

    def choose_action(self, state: List[int]) -> int:
        """
        Pick an epsilon-greedy action
        :param state: current state
        :return: action index
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state[0], state[1]]))

    def update(self, rollout: List[Tuple[Any, int, Any, float]]):
        """
        Update the q-table based on the current rollout.

        :param rollout: A list of transitions and resulting rewards in the environment
        """
        # extract last state and last reward
        last_state, last_action, _, last_reward = rollout[-1]

        # Compute the n-step return for each step in the rollout
        # initialize the returns
        returns = [(last_state, last_action, last_reward)]
        current_return = last_reward
        # compute the returns for each state iteratively starting from the last return in the rollout.
        for i in range(len(rollout) - 2, -1, -1):
            state, action, _, reward = rollout[i]
            current_return = reward + self.gamma * current_return
            returns.append((state, action, current_return))

        # Update the Q-values for each transition in the rollout
        for state, action, n_step_return in returns:
            q_target = n_step_return
            q_current = self.q_table[state[0], state[1], action]
            error = q_target - q_current
            self.q_table[state[0], state[1], action] += self.alpha * error

        # update epsilon based on the decay
        if self.epsilon > 0.0:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = 0

    def train(self, episodes: int = 50000):
        """
        Train the n-step Q-learning agent.

        :param episodes: Number of episodes to train for
        """
        for episode in range(episodes):
            state = self.env.reset()[0]
            terminated = False

            rollout = []
            while not terminated:
                for i in range(self.update_depth):
                    if not terminated:
                        action = self.choose_action(state)
                        next_state, reward, terminated, _, _ = self.env.step(action)
                        rollout.append((state, action, next_state, reward))
                        state = next_state
                    if terminated:
                        break

                self.update(rollout)

            if (episode + 1) % 100 == 0:
                print(f"Episode: {episode + 1}, Epsilon: {self.epsilon}")
