#!/usr/bin/env python

from .core import Agent


class Obstacle(Agent):

    def __init__(self, position, goal, noise_params, velocity=None, radius=1, noise_samples=10000, dt=1/10, name='Obstacle'):
        Agent.__init__(self, position=position, velocity=velocity, goal=goal, noise_params=noise_params, noise_samples=noise_samples, radius=radius, dt=dt, name=name)

    def set_velocity(self, velocity=None):
        super().set_velocity(velocity)
        self.update_noise()
        self.velocity_noise=self.get_noise_samples(self.noise_params['velocity'])
