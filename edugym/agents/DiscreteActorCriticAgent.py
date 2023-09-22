#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Tabular Discrete Actor-Critic implementation

"""

import numpy as np
from edugym.agents.Agent import Agent

class DiscreteActorCriticAgent(Agent):
    def __init__(self, state_dim, action_dim, gamma=1.0, learning_rate=0.1):
        """
        Discrete Actor-Critic agent. 
        The agent maintains a tabular policy and value function. 
        The policy is updated on a policy gradient with advantage function. 
        The value function is updated on a standard tabular update rule (implicit MSE)

        Parameters
        state_dims (int or list): The number of options per state dimension.
        action_dim (int): The number of possible actions.
        gamma (float, optional): The discount factor of future rewards. The default value is 1.0.
        learning_rate (float, optional): The learning rate used in the update method. The default value is 0.1.
        """
        super(DiscreteActorCriticAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize policy and value table
        state_action_dim = np.append(state_dim, action_dim)
        self.logit_pi_sa = np.zeros(state_action_dim)
        self.Vs = np.zeros(state_dim)

    def select_action(self, state, epsilon):
        """
        This method takes an action in the given state, using an epsilon-greedy policy. It returns the index of the selected action.

        Parameters
        state (np.array): The current state of the agent.
        epsilon (float, optional): The value of epsilon for the epsilon-greedy policy.

        Returns
        action (int): The index of the selected action.
        """
        probs = stablesoftmax(self.logit_pi_sa[tuple(state)])
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def update(self, state, action, next_state, reward, done):
        """
        This method updates the policy and value tables based on the observed state, action, next state, reward, and done flag.

        Parameters
        state (np.array): The current state of the agent.
        action (int): The index of the selected action.
        next_state (np.array): The next state of the agent.
        reward (float): The reward received for taking the action in the current state.
        done (bool): Indicator whether the transition terminated the episode. 

        Returns
        None
        """

        # Update value table
        V_target = reward + self.gamma * self.Vs[tuple(next_state)]  # value estimate
        td_V = V_target - self.Vs[tuple(state)]
        self.Vs[tuple(state)] += self.learning_rate * td_V

        # Advantage function
        Q_sa_target = reward + self.gamma * self.Vs[tuple(next_state)]  # value estimate
        A_sa_target = Q_sa_target - self.Vs[tuple(state)]  # advantage estimate

        # Derivative: We want to update our logits vector, for which we will write x
        # For the policy gradient update, we then require d log pi(a_i) / d x for chosen action a_i
        # When you write this out (do this yourself!), you find:
        # d log pi(a_i) / d x_i = (1 - pi(a_i))
        # d log pi(a_i) / d x_j = - pi(a_j)
        pi_a_s = stablesoftmax(self.logit_pi_sa[tuple(state)])
        d_log_pi_ai_d_x = -pi_a_s  # vector of derivatives for each x_j
        d_log_pi_ai_d_x[action] = (
            1 - pi_a_s[action]
        )  # replace the derivative with respect to x_i with the right expression

        # Update policy with gradient *ascent*
        gradient = d_log_pi_ai_d_x * A_sa_target
        self.logit_pi_sa += self.learning_rate * gradient

    def train(self, env, eval_env, epsilon, n_timesteps=5001, eval_interval=500):
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
            s_next, r, done, truncated, info = env.step(a)

            # update Q table
            self.update(s, a, s_next, r, done)

            # Evaluate
            if (t % eval_interval) == 0:
                mean_eval_return = self.evaluate(eval_env, epsilon=0.0)
                timesteps.append(t)
                mean_eval_returns.append(mean_eval_return)
                print(mean_eval_return)

            # Set next state
            if done or truncated:
                s = env.reset()
            else:
                s = s_next

        return timesteps, mean_eval_returns

    def evaluate(self, eval_env, epsilon=0.0, n_episodes=20, max_horizon=100):
        """
        Evaluates the performance of the agent by running a number of episodes in the environment and computing the average return. The method returns the mean return.

        eval_env (gym.Env): The environment to evaluate the agent in.
        epsilon (float): The probability of selecting a random action during evaluation. Default to 0.0. 
        n_episodes (int, optional): The number of episodes to run. Defaults to 50.
        max_horizon (int, optional): The maximum number of steps per episode. Defaults to 100.
        """

        returns = []  # list to store the reward per episode
        for i in range(n_episodes):  # run 50 evaluation episodes
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_horizon):
                a = self.select_action(s, epsilon=epsilon)
                s_prime, r, done, truncated, info = eval_env.step(a)
                R_ep += r
                if done or truncated:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)  # Compute mean performance
        return mean_return


def argmax(x):
    index = np.random.choice(np.flatnonzero(np.isclose(x, x.max())))
    return index


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def get_state_dim(env):
    if hasattr(env.observation_space, "n"):
        state_dim = env.observation_space.n
    elif hasattr(env.observation_space, "nvec"):
        state_dim = env.observation_space.nvec
    else:
        raise ValueError(
            "Implementation only works with Discrete or MultiDiscrete state space, not {}".format(
                env.observation_space
            )
        )
    return state_dim


def test():
    """ Basic Q-learning experiment """

    # Pick an environment
    from edugym.envs.supermarket import SupermarketEnv

    env = SupermarketEnv()
    eval_env = SupermarketEnv()
    state_dim = get_state_dim(env)
    action_dim = env.action_space.n

    # Set your hyperparameters
    learning_rate = 0.1
    gamma = 1.0
    epsilon = 0.1
    n_timesteps = 10001
    n_repetitions = 5

    # Run the experiments
    results = []  # Vector to store evaluation performance
    for rep in range(n_repetitions):
        state_dim = get_state_dim(env)
        Agent = DiscreteActorCriticAgent(
            state_dim, action_dim, gamma=gamma, learning_rate=learning_rate
        )
        time_steps, mean_eval_returns = Agent.train(
            env, eval_env, epsilon=epsilon, n_timesteps=n_timesteps
        )
        results.append(mean_eval_returns)
        print("Completed repetition {}".format(rep))
    average_learning_curve = np.mean(
        np.array(results), axis=0
    )  # Average over repetitions

    # Plot performance
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_steps, y=average_learning_curve, name="Tabular Actor-Critic")
    )
    fig.update_layout(
        title="Tabular Actor-Critic",
        title_x=0.5,
        xaxis_title="Timesteps",
        yaxis_title="Average Return",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(family="serif", size=12),
        width=600,
        height=500,
    )
    fig.write_image("ActorCritic.pdf", scale=2)


if __name__ == "__main__":
    test()
