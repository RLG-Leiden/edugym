import numpy as np
import random
import gym
import math


class DiscretizingQAgent:
    def __init__(self, env, env_name, buckets=(1, 1, 6, 3)):
        self.env = env
        self.is_catch = "Catch" in env_name
        self.is_minimal_catch = "minimal" in env_name
        self.is_onehot_catch = "onehot" in env_name
        self.is_color_catch = "color" in env_name
        self.buckets = buckets
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
        elif env_name == "CartPole-v1":
            self.buckets = (1, 1, 6, 3)
        elif env_name == "Acrobot-v1":
            self.buckets = (3, 3, 3, 3, 3, 3)
        elif env_name == "MountainCar-v0":
            self.buckets = (6, 6)
        elif env_name == "Pendulum-v0":
            self.buckets = (6, 6, 6)

        self.env_name = env_name
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.4  # TODO: add annealing
        self.epsilon_annealing_factor = 0.001
        self.min_epsilon = 0.01
        self.is_continous = type(env.observation_space) is gym.spaces.Box
        if self.is_continous:
            self.q_table = np.zeros(self.buckets + (env.action_space.n,))
        else:
            self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

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

        # cartpole only
        if self.env_name == "CartPole-v1":
            upper_bounds[1] = 0.5
            lower_bounds[1] = -0.5
            upper_bounds[3] = math.radians(50)
            lower_bounds[3] = math.radians(50)

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
        if not deterministic and random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.is_continous:
                return np.argmax(self.q_table[self.discretize_state(state)])
            else:
                return np.argmax(self.q_table[state])

    def update(self, state, action, next_state, reward):
        if self.is_continous:
            q_value = self.q_table[self.discretize_state(state)][action]
            max_value = np.max(self.q_table[self.discretize_state(next_state)])
            new_q_value = (1 - self.alpha) * q_value + self.alpha * (
                reward + self.gamma * max_value
            )
            self.q_table[self.discretize_state(state)][action] = new_q_value
        else:
            q_value = self.q_table[state, action]
            max_value = np.max(self.q_table[next_state])
            new_q_value = (1 - self.alpha) * q_value + self.alpha * (
                reward + self.gamma * max_value
            )
            self.q_table[state, action] = new_q_value
        self.epsilon = max(
            self.min_epsilon, self.epsilon - self.epsilon_annealing_factor
        )
    def train(self, env, timesteps, env_name, eval_env, inbetween_eval_steps, eval_after):
        state = env.reset()
        for timestep in range(0, timesteps):
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            self.update(state, action, next_state, reward)
            if done:
                state = env.reset()
            else:
                state = next_state
            if timestep % eval_after == 0:
                DiscretizingQAgent.evaluate(self, eval_env, inbetween_eval_steps)
        return self
    def evaluate(agent, env, timesteps):
        q = agent
        state = env.reset()
        for _ in range(0, timesteps):
            action = q.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            else:
                state = next_state

def train_q(env, timesteps, env_name, eval_env, inbetween_eval_steps, eval_after):
    agent = DiscretizingQAgent(env, env_name)
    agent.train(env, timesteps, env_name, eval_env, inbetween_eval_steps, eval_after)
    return agent
