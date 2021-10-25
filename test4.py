import numpy as np
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
        'means': np.array([[-0.15, 0.15],[-0.12,0.12]]),
        'stds': np.array([[0.2, 0.2],[0.1,0.1]])
    },
    'velocity': {
        'weights': np.array([0.3, 0.7]),
        'means': np.array([[-0.07, 0.03],[-0.05,0.0214]]),
        'stds': np.array([[0.01, 0.06],[0.02,0.04]])
    }
}
from robot.bot import NonHolonomicBot
bot=NonHolonomicBot(np.array([0,0]), np.array([20,20]), agent_noise_params)
bot.sample_controls()
#from robot.obstacle import Obstacle
#Obstacle( np.array([7,7]), np.array([15,15]), obs_noise_params, name='obs-1')
