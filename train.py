import gymnasium as gym
import numpy as np
from edugym.agents.qlearing import QLearningAgent


def evaluate_agent(agent, episodes=100):
    total_reward = 0
    for _ in range(episodes):
        state = agent.env.reset()[0]
        terminated = False

        while not terminated:
            action = np.argmax(agent.q_table[state[0], state[1]])
            next_state, reward, terminated, _, _ = agent.env.step(action)
            total_reward += reward
            state = next_state

    average_reward = total_reward / episodes
    print(f"Average reward over {episodes} evaluation episodes: {average_reward}")


if __name__ == "__main__":
    env = gym.make("edugym/Study-v0")
    agent = QLearningAgent(env, epsilon_decay=0.00005)
    print("training")
    agent.train(episodes=10000)
    print("evaluation")
    evaluate_agent(agent, episodes=2)

