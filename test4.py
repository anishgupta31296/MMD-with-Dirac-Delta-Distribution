import numpy as np
from seaborn import kdeplot
import matplotlib.pyplot as plt
from robot.bot import NonHolonomicBot
from robot.obstacle import Obstacle
from planner.core import *
import timeit
import random
import os

alpha=1
beta=0.05
save=1
dist=1
times=[]

agent_noise_params = {
    'position': {
        'weights': np.array([0.5, 0.5]),
        'means': np.array([[-0.1, 0.1],[0.075,-0.075]]),
        'stds': np.array([[0.2, 0.2],[0.1,0.1]])
    },
    'velocity': {
        'weights': np.array([0.3, 0.7]),
        'means': np.array([[-0.07, 0.03],[0.105,-0.045]]),
        'stds': np.array([[0.01, 0.06],[0.02,0.03]])
    },
    'controls': {
        'weights': np.array([0.4, 0.6]),
        'means': np.array([[-0.15, 0.1],[0.1,-0.667]]),
        'stds': np.array([[0.01, 0.06],[0.04,0.02]])
    },
    'head': {
        'weights': np.array([0.5, 0.5]),
        'means': np.array([0.0, 0.0]),
        'stds': np.array([0.0, 0.0])
    }        
}
obs_noise_params = {
    'position': {
        'weights': np.array([0.5, 0.5]),
        'means': np.array([[-0.2, 0.0],[-0.0,0.0]]),
        'stds': np.array([[0.0, 0.0],[0.0,0.0]])
    },
    'velocity': {
        'weights': np.array([0.3, 0.7]),
        'means': np.array([[-0.0, 0.0],[-0.0,0.0]]),
        'stds': np.array([[0.0, 0.0],[0.0,0.0]])
    }
}

bot=NonHolonomicBot(np.array([0,0]), np.array([20,20]), agent_noise_params, sensor_range=8)
samples = 100
plt.clf()
ax = plt.gcf().gca()
ax.set_xlim((-5, 20))
# ax.set_xlim((0, 12))
ax.set_ylim((-5,20))
# ax.set_ylim((0, 12)
ax.add_artist(plt.Circle(bot.get_position(), bot.radius, facecolor='#059efb', edgecolor='black', zorder=100))
obstacles = []
#obstacles.append(Obstacle(position=np.array([10,7]), goal=np.array([0,0]), noise_params=obs_noise_params))
#obstacles.append(Obstacle(position=np.array([7,10]), goal=np.array([0,0]), noise_params=obs_noise_params))
obstacles.append(Obstacle(position=np.array([7,7]), goal=np.array([0,0]), noise_params=obs_noise_params))
for i, obs in enumerate(obstacles):
    ax.add_artist(plt.Circle(obs.get_position(), obs.radius, facecolor='#ffa804', edgecolor='black', zorder=100))

itr=random.sample(range(10000),samples)
for i in range(samples):
    ax.add_artist(plt.Circle(bot.position_samples[itr[i],:], bot.radius, color='#059efb', zorder=3, alpha=0.1))

for j in range(len(obstacles)):
    for i in range(samples):
        ax.add_artist(plt.Circle(obstacles[j].position_samples[itr[i],:], obstacles[j].radius, color='#ffa804', zorder=2, alpha=0.1))

plt.show()