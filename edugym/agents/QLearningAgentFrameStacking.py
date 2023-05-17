import numpy as np
import random
from gymnasium import Env
from collections import deque


class QLearningAgent_Framestacking:
    def __init__(
        self,
        n_states,
        n_actions,
        gamma=1.0,
        learning_rate=0.1,
        framestack_size: int = 1,
    ):
        self.n_states = n_states + 1  # add one state for 'no observation yet', at the start of a rollout.
        self.n_actions = n_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.framestack_size = framestack_size

        # create a tuple of n_states * framestack_size + n_actions
        shape = (self.n_states,) * self.framestack_size + (self.n_actions,)

        self.Q_sa = np.zeros(shape)

        # Deque of observations. A deque is a special list of a fixed length.
        # If the maximum length is reached, the element furthest from the newly
        # inserted element is removed.
        self.observations = deque(maxlen=framestack_size)
        self.reset_observations()

    def select_action(self, state, epsilon=0.0):
        self.observations.append(state)
        state = tuple(self.observations)

        if random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.n_actions)  # Explore: choose a random action
        else:
            max_value = np.max(self.Q_sa[state])
            return int(random.choice([i for i, v in enumerate(self.Q_sa[state]) if v == max_value]))

        return action

    def update(self, state, action, next_state, reward):
        state = tuple(self.observations.copy())
        observation_with_next_state = self.observations.copy()
        observation_with_next_state.append(next_state)
        next_state = tuple(observation_with_next_state.copy())

        self.Q_sa[state][action] += self.learning_rate * (reward + self.gamma * np.max(self.Q_sa[next_state]) - self.Q_sa[state][action])

    def reset_observations(self):
        self.observations.clear()
        for _ in range(self.framestack_size):
            self.observations.append(self.n_states - 1)

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

        s, _ = env.reset()
        self.reset_observations()
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
                s, _ = env.reset()
                self.reset_observations()
            else:
                s = s_next

        return timesteps, mean_eval_returns

    def evaluate(self, eval_env, epsilon, n_episodes=50):
        """
        Evaluates the performance of the agent by running a number of episodes in the environment and computing the average return. The method returns the mean return.

        eval_env (gym.Env): The environment to evaluate the agent in.
        epsilon (float): The probability of selecting a random action during evaluation.
        n_episodes (int, optional): The number of episodes to run. Defaults to 50.
        max_horizon (int, optional): The maximum number of steps per episode. Defaults to 100.
        """

        returns = []  # list to store the reward per episode
        for i in range(n_episodes):  # run 50 evaluation episodes
            s, _ = eval_env.reset()
            self.reset_observations()
            R_ep = 0
            done = False
            while True:
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
    """Basic Q-learning experiment"""


    from edugym.envs.memorycorridor import MemoryCorridorEnv
    learning_rate = 0.1
    gamma = 0.99
    epsilon = 0.1

    n_timesteps = 10001
    n_repetitions = 5

    results = []
    for rep in range(n_repetitions):
        env = MemoryCorridorEnv(num_doors=3)
        eval_env = MemoryCorridorEnv(num_doors=3)
        Agent = QLearningAgent_Framestacking(env.observation_space.n, env.action_space.n, gamma, learning_rate, framestack_size=4)
        time_steps, returns = Agent.train(env, eval_env, epsilon, n_timesteps, eval_interval=50)
        results.append(returns)
        print("Completed repetition {}".format(rep))
    average_learning_curve = np.mean(np.array(results), axis=0)

    # Generate figure
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=average_learning_curve, name="Q-learning with framestacking"))

    # Customize layout
    fig.update_layout(
        title="Q-learning with framestacking",
        title_x=0.5,
        xaxis_title="Timesteps",
        yaxis_title="Average Return",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(family="serif", size=12),
        width=600,
        height=500,
    )
    fig.write_image("QLearning_with_framestacking.pdf", scale=2)


if __name__ == "__main__":
    test()
