# EduGym

This repository contains the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments of EduGym: a suite for reinforcement learning education.

See our [website](https://sites.google.com/view/edu-gym) for more information and [see the code in the Notebooks](https://sites.google.com/view/edu-gym/environments) to illustrate the specific challenge and possible solution approaches that the environments are supposed to teach.

# Repository Structure

We provide both Environments (`edugym/envs`) and Agents (`edugym/agents`) to train on.
Each can be executed as a main program. 
Executing an agent will train and evaluate it producing a learning curve, e.g.:
```shell
python3 -m edugym.agents.QLearningAgent
```
Executing an environment lets the user play an episode themselves. The key mapping will be output in the terminal:
```shell
python3 -m edugym.envs.supermarket
```
Below is a table of the available Agents / Environments paths one can execute

| Environments               | Agents                                    |
|----------------------------|-------------------------------------------|
| edugym.envs.boulder        | edugym.agents.DynaAgent                   |
| edugym.envs.catch          | edugym.agents.DynamicProgrammingAgent     |
| edugym.envs.golf           | edugym.agents.ModelLearningAgent          |
| edugym.envs.memorycorridor | edugym.agents.PrioritizedSweepingAgent     |
| edugym.envs.roadrunner     | edugym.agents.QLearningAgent              |
| edugym.envs.study          | edugym.agents.QLearningAgentDiscretized   |
| edugym.envs.supermarket    | edugym.agents.QLearningAgentFrameStacking |
| edugym.envs.tamagotchi     | edugym.agents.SarsaAgent                  |
| edugym.envs.trashbot       |                                           |
