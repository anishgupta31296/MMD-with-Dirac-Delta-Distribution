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

def collision_cones(lin_ctrl, ang_ctrl,h, v, w,ap,op,ov ,R,dt,control_samples,samples):
    i=np.arange(samples)#random.sample(range(ap.shape[0]),samples)
    j=np.arange(samples)#random.sample(range(ap.shape[0]),samples)
    r1=ap[i].reshape(1,samples,1,2)
    vo1=ov[j].reshape(1,1,samples,2)
    ro1=op[j].reshape(1,1,samples,2)
    h1=h[i].reshape(1,samples,1,1)
    v_c=control_samples[i,0].reshape(1,samples,1,1)
    w_c=control_samples[i,1].reshape(1,samples,1,1)
    l_ctrl1=lin_ctrl.reshape(lin_ctrl.shape[0],1,1,1)
    a_ctrl1=ang_ctrl.reshape(ang_ctrl.shape[0],1,1,1)
    nh_v=(l_ctrl1+v+v_c)*np.concatenate((np.cos(h1+(a_ctrl1+w+w_c)*dt), np.sin(h1+(a_ctrl1+w+w_c)*dt)),axis=3)
    vr=nh_v-vo1
    rr=r1-ro1
    cones=np.square(np.sum(vr*rr, axis=3))+ np.sum(np.square(vr), axis=3)*((R)**2 - np.sum(np.square(rr), axis=3))
    cones=cones.reshape(lin_ctrl.shape[0],samples*samples)
    return cones,vr,rr

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
        'means': np.array([[0.4, -0.1],[-0.4,0.1]]),
        'stds': np.array([[0.15, 0.15],[0.2,0.2]])
    },
    'velocity': {
        'weights': np.array([0.5, 0.5]),
        'means': np.array([[0.0, 0.0],[0.0,0.0]]),
        'stds': np.array([[0.001, 0.001],[0.001,0.001]])
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

print(np.mean(obstacles[0].position_noise),np.std(obstacles[0].position_noise))
itr=random.sample(range(10000),samples)
for i in range(samples):
    ax.add_artist(plt.Circle(bot.position_samples[itr[i],:], bot.radius, color='#059efb', zorder=3, alpha=0.08))

for j in range(len(obstacles)):
    for i in range(samples):
        ax.add_artist(plt.Circle(obstacles[j].position_samples[itr[i],:], obstacles[j].radius, color='#ffa804', zorder=2, alpha=0.08))
bot.sample_controls()
#plt.show()

cones_final,vr1,rr1=collision_cones(bot.lin_ctrl, bot.ang_ctrl,bot.head_samples, bot.get_linear_velocity(), bot.get_angular_velocity(),  bot.position_samples, obstacles[0].position_samples, obstacles[0].velocity_samples,bot.radius+obstacles[0].radius,bot.dt,bot.controls_samples,100)
fig = plt.figure()
plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
cones,vr,rr=bot.collision_cones(obstacles[0],100)
#cones[cones<0]=0
ax = kdeplot(cones_final[-1,:], label='Final Cones', shade=True, color='#ffa804')
#ax.axvline(x=0)
ax.set_xlim((-25, 25))
ax.set_ylim((0, 0.3))


fig1= plt.figure()
plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
#cones[cones<0]=0
ax = kdeplot(cones, label='Final Cones', shade=True, color='#ffa804')
#ax.axvline(x=0)
ax.set_xlim((-25, 25))
ax.set_ylim((0, 0.3))
print(vr.flatten()-vr1[-1,:].flatten())
#print(bot.lin_ctrl, bot.ang_ctrl)
plt.show()