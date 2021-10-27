#!/usr/bin/env python

import numpy as np
import math

class Agent:

    def __init__(self, position, velocity, goal, noise_params, noise_samples, radius=1, dt=1/10, name='Bot'):
        self.name = name
        self.radius = radius
        self.filter_radius=radius*3
        self.position = position
        if velocity is None:
            self.velocity=0.5*(goal-position)/np.linalg.norm(goal-position)
        else:
            self.velocity= velocity
        self.noise_samples=noise_samples
        self.position_noise=self.get_noise_samples(noise_params['position'])
        self.velocity_noise=self.get_noise_samples(noise_params['velocity'])
        self.noise_params=noise_params
        self.noise_samples=noise_samples
        self.goal = goal
        self.dt = dt
        self.path = []
        self.reduced_position_noise=None
        self.reduced_velocity_noise=None
        self.reduced_coeffs=None
        
    def __str__(self):
        return self.name

    def update_samples(self):
        print('update_samples')
        self.position_samples=self.position+self.position_noise-np.mean(self.position_noise)
        self.velocity_samples=self.velocity+self.velocity_noise

    def get_noise_samples(self, params, samples=None):
        if samples is None:
            samples=self.noise_samples
        weights=params['weights']
        means=params['means']
        stds=params['stds']
        if(means.ndim==1):
            return np.vstack((np.random.randn(np.int16(round(samples*weights[0])),1)*stds[0]+means[0], np.random.randn(samples-np.int16(round(samples*weights[0])),1)*stds[1]+means[1]))
        else:
            cols=means.shape[0]
            return np.vstack((np.random.randn(np.int16(round(samples*weights[0])),cols)*stds[:,0]+means[:,0], np.random.randn(samples-np.int16(round(samples*weights[0])),cols)*stds[:,1]+means[:,1]))

    def update_noise(self):
        print('update_noise_core')
        self.position_noise=self.position_noise+self.velocity_noise*self.dt
        dist_boolean_list=np.linalg.norm(self.position_noise, axis=1)>self.filter_radius
        new_samples_count=np.count_nonzero(dist_boolean_list)
        if(new_samples_count>0):
            self.position_noise[dist_boolean_list,:]= self.get_noise_samples(self.noise_params['position'], samples=new_samples_count)
        self.update_samples()       

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def set_position(self, position):
        self.position = position
        self.path.append(self.position)

    def get_goal(self):
        return self.goal

    def goal_reached(self):
        goal_reached = np.linalg.norm(self.get_position()-self.goal)<self.radius
        return goal_reached       

    def set_velocity(self, velocity=None):
        if velocity is None:
            self.set_position(self.get_position() + self.velocity * self.dt)
        else:
            if(self.goal_reached()):
               self.velocity=np.array([0,0])
            else:    
               self.velocity=velocity
               self.set_position(self.get_position() + self.velocity * self.dt)


    
    def set_velocity_noise(self, velocity_noise):
        self.velocity_noise=velocity_noise  

    def collision_cone(self, obstacle):
        r = self.get_position() - obstacle.get_position()
        v = self.get_velocity() - obstacle.get_velocity()
        cone = ((r @ v) / v.__pow__(2).sum()) - r.__pow__(2).sum() + (self.radius + obstacle.radius).__pow__(2)
        return cone

    def is_colliding(self, obstacle):
        if self.collision_cone(obstacle) > 0:
            return True
        return False
