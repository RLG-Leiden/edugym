#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Dynamic Programming Agent

"""

import numpy as np
import plotly.graph_objects as go

from edugym.agents.Agent import Agent
from edugym.envs.supermarket import SupermarketEnv


class DynamicProgrammingAgent(Agent):
    def __init__(self, n_states, n_actions, gamma=1.0):
        """
        This method initializes a Dynamic Programming agent.

        Parameters: 
        n_states (int): The number of states in the environment.
        n_actions (int): The number of actions in the environment.
        gamma (float, optional): The discount factor used in the Bellman equation. Defaults to 1.0.
        """

        super(DynamicProgrammingAgent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state, epsilon=0.0):
        """
        Selects an action for a given state using an epsilon-greedy policy. The method returns the index of the selected action.
        
        state (np.array): The current state of the agent.
        epsilon (float, optional): The probability of selecting a random action. Defaults to 0.0.
         """
        if np.random.random() < epsilon:
            action = np.random.randint(
                self.n_actions
            )  # Explore: choose a random action
        else:
            action = np.argmax(
                self.Q_sa[state,]
            )  # Exploit: choose the action with the highest Q-value
        return action

    def update(self, s, a, p_sas, r_sas):
        """
        Updates the Q-value for a given state-action pair using the Bellman equation. The method does not return anything.
        
        s (int): The index of the current state.
        a (int): The index of the current action.
        p_sas (np.array): The transition probabilities for each possible next state.
        r_sas (np.array): The rewards for each possible next state-action pair.
        """

        self.Q_sa[s, a] = np.sum(
            p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1))
        )

    def train(self, descriptive_model, threshold=0.001, verbose=True):
        """
        Trains the agent by computing the optimal Q-values for all state-action pairs in the environment. The method does not return anything.
        
        descriptive_model (function): A function that takes the current state and action as input and returns the transition probabilities and rewards for each possible next state-action pair.
        threshold (float, optional): The maximum difference allowed between the previous and current Q-value for each state-action pair. Defaults to 0.001.
        verbose (bool, optional): Whether to print the progress of the training. Defaults to True.
        """

        max_error = np.inf
        i = 0

        while max_error > threshold:
            max_error = 0.0
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    # transition & reward
                    p_sas, r_sas, term_s = descriptive_model(s, a)

                    # update & log error
                    previous_estimate = np.copy(self.Q_sa[s, a])
                    self.update(s, a, p_sas, r_sas)
                    error = np.abs(self.Q_sa[s, a] - previous_estimate)
                    # if error > max_error:
                    #    print(s,a,np.where(p_sas>0),r_sas[np.where(p_sas>0)],self.Q_sa[s,a])

                    max_error = max(error, max_error)

            if verbose:
                print(
                    "Q-value iteration, iteration {}, max error {}".format(i, max_error)
                )
            i += 1

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
    """ Notebook experiments with Dynamic Programming """

    # 1. DP with perfect model
    step_timeout = 0.0
    gamma = 1.0
    threshold = 0.01

    env = SupermarketEnv(step_timeout=step_timeout)
    DPAgent = DynamicProgrammingAgent(
        env.observation_space.n, env.action_space.n, gamma
    )
    DPAgent.train(env.descriptive_model, threshold)

    done = False
    s = env.reset()
    while not done and not truncated:
        a = DPAgent.select_action(s)
        s_next, r, done, truncated,  _ = env.step(a)
        env.render()
        s = s_next
    env.close()

    # DP with noisy model
    gamma = 0.9
    noise_levels = [0, 0.5, 1, 2, 5]
    n_repetitions = 10
    performances = np.empty([len(noise_levels), n_repetitions])
    for i, noise in enumerate(noise_levels):
        for j in range(n_repetitions):
            env = SupermarketEnv(step_timeout=0.0, noise=noise)
            DPAgent = DynamicProgrammingAgent(
                env.observation_space.n, env.action_space.n, gamma
            )
            DPAgent.train(env.descriptive_model, threshold, verbose=False)
            performance = DPAgent.evaluate(env, 0.01)
            performances[i, j] = performance
        mean_performance_noise = np.mean(performances[i,])
        print(f"Noise level {noise}, mean return {mean_performance_noise}")
    mean_performance = np.mean(performances, axis=1)

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=noise_levels, y=np.array(mean_performance)))
    fig.update_layout(
        title="Dynamic Programming",
        title_x=0.5,
        xaxis_title="Reward model noise",
        yaxis_title="Average return at convergence",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(family="serif", size=12),
        width=600,
        height=500,
    )
    fig.write_image("DynamicProgramming.pdf", scale=2)


if __name__ == "__main__":
    test()
