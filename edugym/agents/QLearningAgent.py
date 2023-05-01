#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Q-learning Agent

"""

import numpy as np
import plotly.graph_objects as go

from edugym.agents.Agent import Agent
from edugym.envs.supermarket import SupermarketEnv


class QLearningAgent(Agent):
    def __init__(self, n_states, n_actions, gamma=1.0, learning_rate=0.1):
        """
        This method initializes a Q-learning agent. 

        Parameters
        n_states (int): The number of possible states.
        n_actions (int): The number of possible actions.
        gamma (float, optional): The discount factor used in the Q-learning algorithm. The default value is 1.0.
        learning_rate (float, optional): The learning rate (alpha) used in the Q-learning algorithm. The default value is 0.1.
        """
        super(QLearningAgent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize Q(s,a) table
        self.Q_sa = np.zeros([self.n_states, self.n_actions])

    def select_action(self, state, epsilon=0.0):
        """
        This method takes an action in the given state, using an epsilon-greedy policy. It returns the index of the selected action.

        Parameters
        state (np.array): The current state of the agent.
        epsilon (float, optional): The value of epsilon for the epsilon-greedy policy. The default value is 0.0.
        
        Returns
        action (int): The index of the selected action.
        """
        if np.random.random() < epsilon:
            action = np.random.randint(
                self.n_actions
            )  # Explore: choose a random action
        else:
            action = np.argmax(
                self.Q_sa[state]
            )  # Exploit: choose the action with the highest Q-value
        return action

    def update(self, state, action, next_state, reward):
        """
        This method updates the Q-values and model based on the observed state, action, next state, reward, and done flag.

        Parameters
        state (np.array): The current state of the agent.
        action (int): The index of the selected action.
        next_state (np.array): The next state of the agent.
        reward (float): The reward received for taking the action in the current state.
        update_model (bool, optional): A flag indicating whether or not to update the internal model of the agent. The default value is False.
        
        Returns
        None
        """

        # Estimate new Q target
        new_Qsa_estimate = reward + self.gamma * np.max(self.Q_sa[next_state])

        # Update the tabular estimate
        td_error = new_Qsa_estimate - self.Q_sa[state, action]
        self.Q_sa[state, action] += self.learning_rate * td_error

    def train(self, env, eval_env, epsilon, n_timesteps=10000, eval_interval=500):
        """
        This method trains the agent using the provided environment.

        Parameters
        env (gym.Env): The training environment. 
        eval_env (gym.Env): The evaluation environment. 
        epsilon (float): The value of epsilon for the epsilon-greedy policy.
        n_timesteps (int, optional): The number of timesteps to train for. The default value is 10000.
        eval_interval (int, optional): The interval at which to perform evaluations. The default value is 500.
        """

        timesteps = []
        mean_eval_returns = []

        reward_sums = []
        s, info = env.reset()
        r_sum = 0
        steps = 0
        step_sums = []
        for t in range(n_timesteps):
            # take action in environment
            a = self.select_action(s, epsilon)
            s_next, r, done, truncated, _ = env.step(a)
            r_sum += r
            steps += 1

            # update Q table
            self.update(s, a, s_next, r)

            # Evaluate
            if (t % eval_interval) == 0:
                mean_eval_return = self.evaluate(eval_env, epsilon=0.0)
                mean_eval_returns.append(mean_eval_return)

            # Set next state
            if done or truncated:
                reward_sums.append(r_sum)
                step_sums.append(steps)
                steps = 0
                r_sum = 0
                timesteps.append(t)
                s, info = env.reset()
            else:
                s = s_next

        return timesteps, mean_eval_returns, reward_sums, step_sums

    def evaluate(self, eval_env, epsilon, n_episodes=1, max_horizon=100):
        """
        Evaluates the performance of the agent by running a number of episodes in the environment and computing the average return. The method returns the mean return.
        
        eval_env (gym.Env): The environment to evaluate the agent in.
        epsilon (float): The probability of selecting a random action during evaluation.
        n_episodes (int, optional): The number of episodes to run. Defaults to 50.
        max_horizon (int, optional): The maximum number of steps per episode. Defaults to 100.
        """

        returns = []  # list to store the reward per episode
        for i in range(n_episodes):  # run 50 evaluation episodes
            s, info = eval_env.reset()
            R_ep = 0
            for t in range(max_horizon):
                a = self.select_action(s, epsilon)
                s_prime, r, done, truncated, _ = eval_env.step(a)
                R_ep += r
                if done or truncated:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)

        return mean_return


def test():
    """ Basic Q-learning experiment """

    learning_rate = 0.1
    gamma = 1.0
    epsilon = 0.1

    n_timesteps = 10001
    n_repetitions = 5

    results = []
    for rep in range(n_repetitions):
        env = SupermarketEnv()
        eval_env = SupermarketEnv(step_timeout=0.0)
        Agent = QLearningAgent(
            env.observation_space.n, env.action_space.n, gamma, learning_rate
        )
        time_steps, returns = Agent.train(env, eval_env, epsilon, n_timesteps)
        results.append(returns)
        print("Completed repetition {}".format(rep))
    average_learning_curve = np.mean(np.array(results), axis=0)

    # Generate figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=average_learning_curve, name="Q-learning"))

    # Customize layout
    fig.update_layout(
        title="Q-learning",
        title_x=0.5,
        xaxis_title="Timesteps",
        yaxis_title="Average Return",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(family="serif", size=12),
        width=600,
        height=500,
    )
    fig.write_image("QLearning.pdf", scale=2)


if __name__ == "__main__":
    test()
