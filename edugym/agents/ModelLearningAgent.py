#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Model Learning Agent

"""

import numpy as np

from edugym.agents.Agent import Agent


class ModelLearningAgent(Agent):
    def __init__(self, n_states, n_actions):
        """
        The ModelLearningAgent class implements an agent that learns a model of the environment based on observed transitions. 
        
        Parameters: 
        n_states (int): The number of states in the environment.
        n_actions (int): The number of actions in the environment.
        """
        super(ModelLearningAgent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.n_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.r_sum_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.p_sas = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.r_sas = np.zeros((self.n_states, self.n_actions, self.n_states))

    def select_action(self, state):
        """
        This method selects an action in the given state.
        
        Parameters:
        state (np.array): The current state of the agent.
        
        Returns:
        action (int): The index of the selected action.
        """
        action = np.random.randint(self.n_actions)
        return action

    def update(self, state, action, next_state, reward, done):
        """
        This method updates the transition model based on an observed state, action, next state, reward, and done flag.
        
        Parameters:
        state (np.array): The current state of the agent.
        action (int): The index of the selected action.
        next_state (np.array): The next state of the agent.
        reward (float): The reward received for taking the action in the current state.
        done (bool): Whether the episode has ended.
        
        Returns:
        None.
        """
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
        """
        This method returns the next state and reward according to the learned model of the environment.
        
        Parameters:
        state (np.array): The current state of the agent.
        action (int): The index of the selected action.
        
        Returns:
        next_state (int): The next state according to the learned model of the environment.
        reward (float): The reward received for transitioning to the next state.
        """
        if np.sum(self.p_sas[state, action]) == 0:
            raise Warning(
                "Calling learned_model() from a state that was never visited. Returning None."
            )
            return None, None
        else:
            next_state = np.random.choice(self.n_states, p=self.p_sas[state, action])
            reward = self.r_sas[state, action, next_state]
            return next_state, reward


def test():
    """ Notebook experiments with a Model Learning Agent """

    from edugym.envs.supermarket import SupermarketEnv
    # DP with perfect model
    step_timeout = 0.0
    env = SupermarketEnv(step_timeout=step_timeout)
    MLAgent = ModelLearningAgent(env.observation_space.n, env.action_space.n)

    n_steps = 1000
    s = env.reset()
    for t in range(n_steps):
        a = MLAgent.select_action(s)
        s_prime, r, done, truncated, _ = env.step(a)
        MLAgent.update(s, a, s_prime, r, done)
        if done or truncated:
            env.reset()
        else:
            s = s_prime

    # Compare what the true model and learned model say:
    state_vector = [0, 1, 0, 0, 0]
    state = env.vector_to_state(state_vector)
    action = 1
    next_state_model, reward_model, _ = env.generative_model(state, action)
    next_state_model_vector = env.state_to_vector(next_state_model)
    next_state_learned, reward_learned = MLAgent.learned_model(state, action)
    next_state_learned_vector = env.state_to_vector(next_state_learned)

    print(
        f"For state {state} (vector {state_vector}) and action {action},\nthe true model predicts next state {next_state_model} (vector {next_state_model_vector} and reward {reward_model},\nwhile the learned model predicts next state {next_state_learned} (vector {next_state_learned_vector} and reward {reward_learned}"
    )


if __name__ == "__main__":
    test()
