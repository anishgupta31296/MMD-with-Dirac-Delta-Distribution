#!/usr/bin/env python

from .core import Agent


class Obstacle(Agent):

    def __init__(self, position, goal, noise_params, noise_samples=10000, radius=1, dt=1/10,name='Obstacle'):
        Agent.__init__(self, position=position, velocity=None, goal=goal, noise_params=noise_params, noise_samples=noise_samples, radius=radius, dt=dt, name=name)
