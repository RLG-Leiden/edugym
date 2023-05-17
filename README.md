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
Executing an environment lets the user play ane pisode themselves by issuing actions via the command line:
```shell
python3 -m edugym.envs.supermarket
```
