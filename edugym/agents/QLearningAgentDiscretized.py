#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Q-learning Agent

"""

import numpy as np
import plotly.graph_objects as go
import gymnasium as gym
import random
from edugym.agents.Agent import Agent


class DiscretizingQLearningAgent(Agent):
    def __init__(self, env, env_name, gamma=1.0, learning_rate=0.1,):
        """
        This method initializes a Q-learning agent. 

        Parameters
        gamma (float, optional): The discount factor used in the Q-learning algorithm. The default value is 1.0.
        learning_rate (float, optional): The learning rate (alpha) used in the Q-learning algorithm. The default value is 0.1.
        """
        super(DiscretizingQLearningAgent, self).__init__()
        self.env = env
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.4 
        self.epsilon_annealing_factor = 0.001
        self.min_epsilon = 0.01
        self.buckets = (6, 6)
        self.is_catch = "Catch" in env_name
        self.is_minimal_catch = "minimal" in env_name
        self.is_onehot_catch = "onehot" in env_name
        self.is_color_catch = "color" in env_name
        if self.is_catch:
            try:
                multiply_size_by = int(env_name[6:7])
            except:
                multiply_size_by = 1
            rows = 10 * multiply_size_by
            cols = 5 * multiply_size_by
            position_possibilities = rows * cols
            # bucket 1 = ball position, bucket 2 = paddle position
            self.buckets = (position_possibilities, position_possibilities)
        
        self.is_continous = type(env.observation_space) is gym.spaces.Box
        if self.is_continous:
            self.Q_sa = np.zeros(self.buckets + (env.action_space.n,))
        else:
            self.Q_sa = np.zeros([env.observation_space.n, env.action_space.n])

    def discretize_state(self, state):
        if self.is_catch:
            if self.is_minimal_catch:
                return (
                    ((state[0] + 1) * (state[1] + 1)) - 1,
                    ((state[2] + 1) * (state[3] + 1)) - 1,
                )
            if self.is_color_catch:
                ball_pos = -1
                paddle_pos = -1
                ball_posi = np.argwhere(state == 128.0)
                paddle_posi = np.argwhere(state == 255.0)
                if len(ball_posi) > 0:
                    ball_pos = (ball_posi[0][0] + 1) * (ball_posi[0][1] + 1)
                if len(paddle_posi) > 0:
                    paddle_pos = (paddle_posi[0][0] + 1) * (paddle_posi[0][1] + 1)
                return (ball_pos - 1, paddle_pos - 1)
            else:
                item_positions = np.argwhere(state == 1.0)
                ball_pos = -1
                paddle_pos = -1
                if len(item_positions) > 0:
                    if self.is_onehot_catch:
                        for entry in item_positions:
                            extracted_pos = (entry[0] + 1) * (entry[1] + 1)
                            if entry[2] == 0:
                                ball_pos = extracted_pos
                            elif entry[2] == 1:
                                paddle_pos = extracted_pos
                    else:
                        ball_pos = (item_positions[0][0] + 1) * (
                            item_positions[0][1] + 1
                        )
                        if len(item_positions) > 1:
                            paddle_pos = (item_positions[1][0] + 1) * (
                                item_positions[1][1] + 1
                            )
                        else:
                            paddle_pos = ball_pos
                return (ball_pos - 1, paddle_pos - 1)
        upper_bounds = self.env.observation_space.high
        lower_bounds = self.env.observation_space.low

        width = [upper_bounds[i] - lower_bounds[i] for i in range(len(state))]
        ratios = [(state[i] - lower_bounds[i]) / width[i] for i in range(len(state))]
        bucket_indices = []
        for i in range(len(state)):
            val = ratios[i] * (self.buckets[i] - 1)
            if math.isinf(val):
                val = math.inf
            else:
                val = int(round(val))
            bucket_indices.append(val)
        bucket_indices = [
            max(0, min(bucket_indices[i], self.buckets[i] - 1))
            for i in range(len(state))
        ]
        return tuple(bucket_indices)
    def select_action(self, state, deterministic=False):
        """
        This method takes an action in the given state, using an epsilon-greedy policy. It returns the index of the selected action.

        Parameters
        state (np.array): The current state of the agent.
        epsilon (float, optional): The value of epsilon for the epsilon-greedy policy. The default value is 0.0.

        Returns
        action (int): The index of the selected action.
        """
        if not deterministic and random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.is_continous:
                return np.argmax(self.Q_sa[self.discretize_state(state)])
            else:
                return np.argmax(self.Q_sa[state])

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
        if self.is_continous:
            state = self.discretize_state(state)
            next_state = self.discretize_state(next_state)
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

        s = env.reset()
        for t in range(n_timesteps):
            # take action in environment
            a = self.select_action(s, epsilon)
            s_next, r, done, truncated, _ = env.step(a)

            # update Q table
            self.update(s, a, s_next, r)

            # Evaluate
            if (t % eval_interval) == 0:
                mean_eval_return = self.evaluate(eval_env, epsilon=0.0)
                timesteps.append(t)
                mean_eval_returns.append(mean_eval_return)

            # Set next state
            if done or truncated:
                s = env.reset()
            else:
                s = s_next

        return timesteps, mean_eval_returns

    def evaluate(self, eval_env, epsilon, n_episodes=50, max_horizon=100):
        """
        Evaluates the performance of the agent by running a number of episodes in the environment and computing the average return. The method returns the mean return.
        
        eval_env (gym.Env): The environment to evaluate the agent in.
        epsilon (float): The probability of selecting a random action during evaluation.
        n_episodes (int, optional): The number of episodes to run. Defaults to 50.
        max_horizon (int, optional): The maximum number of steps per episode. Defaults to 100.
        """

        returns = []  # list to store the reward per episode
        for i in range(n_episodes):  # run 50 evaluation episodes
            s = eval_env.reset()
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
    from edugym.envs.catch import Catch
    """ Basic Q-learning experiment """

    learning_rate = 0.1
    gamma = 1.0
    epsilon = 0.1

    n_timesteps = 100001
    n_repetitions = 5

    results = []
    for rep in range(n_repetitions):
        env = Catch()
        eval_env = Catch()
        Agent = DiscretizingQLearningAgent(env, "Catch-v0")
        time_steps, returns = Agent.train(env, eval_env, epsilon, n_timesteps)
        results.append(returns)
        print("Completed repetition {}".format(rep))
    average_learning_curve = np.mean(np.array(results), axis=0)

    # Generate figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=average_learning_curve, name="Q-learning"))

    # Customize layout
    fig.update_layout(
        title="Q-learning Discretized",
        title_x=0.5,
        xaxis_title="Timesteps",
        yaxis_title="Average Return",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(family="serif", size=12),
        width=600,
        height=500,
    )
    fig.write_image("QLearningDiscretized.pdf", scale=2)


if __name__ == "__main__":
    test()
