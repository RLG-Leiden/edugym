#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edugym: Agent template

"""


class Agent:

    """ Template class for defining an agent in a reinforcement learning environment."""

    def select_action(self, s):
        """ Description: Method to select an action given the current state. """
        raise NotImplementedError

    def update(self):
        """ Method to update the agent's solution based on obtained data."""
        raise NotImplementedError

    def train(self):
        """ Description: Method to train the agent in the environment."""
        raise NotImplementedError

    def evaluate(self):
        """ Description: Method to evaluate the performance of the agent in the environment."""
        raise NotImplementedError
