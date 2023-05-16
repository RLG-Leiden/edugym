#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Prioritized Sweeping Agent

"""

import numpy as np
from queue import PriorityQueue
import plotly.graph_objects as go

from edugym.agents.Agent import Agent
from edugym.envs.supermarket import SupermarketEnv


class PrioritizedSweepingAgent(Agent):
    def __init__(
        self, n_states, n_actions, gamma=1.0, learning_rate=0.1, priority_cutoff=0.01
    ):
        """
        This method initializes a Prioritized Sweeping agent. 
        
        Parameters:
        n_states (int): The number of possible states. 
        n_actions (int): The number of possible actions.
        learning_rate (float, optional): The learning rate (alpha) used in the Q-learning algorithm. Defaults to 0.1.
        gamma (float, optional): The discount factor used in the Q-learning algorithm. Defaults to 1.0.
        """
        super(PrioritizedSweepingAgent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize Q(s,a) table
        self.Q_sa = np.zeros([self.n_states, self.n_actions])

        # Initialize model learning arrays
        self.n_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.r_sum_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.p_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.r_sas = np.zeros((self.n_states, self.n_actions, self.n_states))

        # Priority queue
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()

    def select_action(self, state, epsilon=0.0):
        """
        This method takes an action in the given state, using an epsilon-greedy policy. It returns the index of the selected action.

        Parameters:
        state (np.array): The current state of the agent.
        
        Returns:
        action (int): The index of the selected action.
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

    def update(self, state, action, next_state, reward, update_model=False):
        """
        This method updates the Q-values based on the observed state, action, next state, reward, and done flag.

        Parameters:
        state (np.array): The current state of the agent.
        action (int): The index of the selected action.
        next_state (np.array): The next state of the agent.
        reward (float): The reward received for taking the action in the current state.
        
        Returns:
        None.
        """

        # Update the Q-table
        td_error = (
            reward
            + self.gamma * np.max(self.Q_sa[next_state])
            - self.Q_sa[state, action]
        )
        self.Q_sa[state, action] += self.learning_rate * td_error

        # Update the model
        if update_model:
            self.n_sas[state, action, next_state] += 1
            self.r_sum_sas[state, action, next_state] += reward
            self.r_sas[state, action, next_state] = (
                self.r_sum_sas[state, action, next_state]
                / self.n_sas[state, action, next_state]
            )
            self.p_sas[state, action] = self.n_sas[state, action] / np.sum(
                self.n_sas[state, action,]
            )

    def learned_model(self, state, action):
        if np.sum(self.p_sas[state, action]) == 0:
            raise ValueError(
                "Cannot call learned_model on a state that has never been observed."
            )
        next_state = np.random.choice(self.n_states, p=self.p_sas[state, action])
        reward = self.r_sas[state, action, next_state]
        return next_state, reward

    def reverse_model(self, next_state):
        previous_state_actions = np.transpose(
            (self.n_sas[:, :, next_state] > 0).nonzero()
        )
        return previous_state_actions

    def plan(self, n_planning_updates=1):

        for i in range(n_planning_updates):
            if self.queue.qsize() == 0:
                break

            _, (s, a) = self.queue.get()

            # find the associated reward and next state
            s_next, r = self.learned_model(s, a)

            # perform an update
            self.update(s, a, s_next, r, update_model=False)

            # loop all precursors of s
            precursors = self.reverse_model(s)
            for precursor in precursors:
                s_prev, a_prev = precursor
                s_next, r = self.learned_model(s_prev, a_prev)
                priority = np.abs(
                    r + self.gamma * np.max(self.Q_sa[s]) - self.Q_sa[s_prev, a_prev]
                )
                if priority > self.priority_cutoff:
                    self.queue.put((-priority, (s_prev, a_prev)))

    def train(self, env, eval_env, epsilon, n_timesteps=10000, eval_interval=500):

        timesteps = []
        mean_eval_returns = []

        s = env.reset()
        for t in range(n_timesteps):
            # take action in environment
            a = self.select_action(s, epsilon)
            s_next, r, done, truncated, _ = env.step(a)

            # update Q table & model
            self.update(s, a, s_next, r, update_model=True)

            # Compute priority of new transition
            priority = np.abs(
                r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a]
            )
            if priority > self.priority_cutoff:
                self.queue.put((-priority, (s, a)))

            # background planning
            while not env.can_call_step():
                self.plan()

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
    """ Notebook experiments with Prioritized Sweeping """

    learning_rate = 0.1
    gamma = 1.0
    epsilon = 0.1
    priority_cutoff = 0.001

    planning_budget_seconds = [0.0, 0.001, 0.005]  # 0.01]
    n_timesteps = 7501
    n_repetitions = 10

    learning_curves_ps = []

    for planning_budget in planning_budget_seconds:
        results = []
        for rep in range(n_repetitions):
            env = SupermarketEnv(step_timeout=planning_budget)
            eval_env = SupermarketEnv(step_timeout=0.0)
            Agent = PrioritizedSweepingAgent(
                env.observation_space.n,
                env.action_space.n,
                gamma,
                learning_rate,
                priority_cutoff,
            )
            time_steps, returns = Agent.train(env, eval_env, epsilon, n_timesteps)
            results.append(returns)
        average_learning_curve = np.mean(np.array(results), axis=0)
        learning_curves_ps.append(average_learning_curve)
        print("Completed planning budget: {}".format(planning_budget))

    # Generate figure
    fig = go.Figure()
    for i, planning_budget in enumerate(planning_budget_seconds):
        name = (
            "Prioritized Sweeping (budget: {} sec)".format(planning_budget)
            if planning_budget > 0
            else "Q-learning"
        )
        fig.add_trace(go.Scatter(x=time_steps, y=learning_curves_ps[i], name=name,))

    # Customize layout
    fig.update_layout(
        title="Background planning: Prioritized Sweeping",
        title_x=0.5,
        xaxis_title="Timesteps",
        yaxis_title="Average Return",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(family="serif", size=12),
        width=600,
        height=500,
    )
    fig.write_image("PrioritizedSweeping.pdf", scale=2)


if __name__ == "__main__":
    test()
