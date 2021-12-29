import numpy as np
from seaborn import kdeplot
import matplotlib.pyplot as plt
from robot.bot import NonHolonomicBot
from robot.obstacle import Obstacle
from planner.core import *
import timeit
import random
import os
import csv

def main():
    os.system("sudo rm ../MMD\ Python\ Outputs/*.png")
    sensor_range=10
    alpha=10
    beta=0.001
    gamma=0.1
    save=1
    dist=1
    samples_to_plot = 100

    times=[]

    agent_noise_params = {
        'position': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([[-0.0, 0.0],[0.0,-0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        },
        'velocity': {
            'weights': np.array([0.3, 0.7]),
            'means': np.array([[-0.0, 0.0],[0.0,-0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        },
        'controls': {
            'weights': np.array([0.4, 0.6]),
            'means': np.array([[-0.0, 0.0],[0.0,-0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
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
            'weights': np.array([0.3, 0.7]),
            'means': np.array([[-0.07, 0.03],[-0.05,0.0214]]),
            'stds': np.array([[0.01, 0.06],[0.02,0.04]])
        }
    }


    bot=NonHolonomicBot(np.array([0,0]), np.array([20,20]), agent_noise_params, sensor_range=8)
    obstacles = []
    #obstacles.append(Obstacle(position=np.array([10,7]), goal=np.array([0,0]), noise_params=obs_noise_params))
    #obstacles.append(Obstacle(position=np.array([7,10]), goal=np.array([0,0]), noise_params=obs_noise_params))
    obstacles.append(Obstacle(position=np.array([7,7]), goal=np.array([0,0]), noise_params=obs_noise_params))
    counter = 0

    #planner=Planner(param=0.1,samples_param=20,optimizer='MMD Dirac Delta',device='cuda:0')
    planner=Planner(param=1.0,samples_param=50,optimizer='PVO',device='cpu')
    while (bot.goal-bot.position).__pow__(2).sum() > 1:
        obstacles_in_range = []
        plt.clf()
        ax = plt.gcf().gca()
        ax.set_xlim((-5, 20))
        # ax.set_xlim((0, 12))
        ax.set_ylim((-5,20))
        # ax.set_ylim((0, 12))
        ax.add_artist(plt.Circle(bot.get_position(), bot.sensor_range, color='gray', alpha=0.1))
        for i, obs in enumerate(obstacles):
            if bot.in_sensor_range(obs):
                obstacles_in_range.append(obs)
            obs.set_velocity()
            ax.add_artist(plt.Circle(obs.get_position(), obs.radius, facecolor='#ffa804', edgecolor='black', zorder=100))
            plt.plot(np.array(obs.path)[:,0],np.array(obs.path)[:,1], '#ffa804', zorder=1)

        counter=counter+1        
        start = timeit.default_timer()
        planner.get_controls(bot,obstacles_in_range, alpha, beta)
        #print('velocity:',bot.get_velocity(),'     controls',planner.optimal_control)
        print(bot.get_velocity())
        bot.set_controls(planner.optimal_control)
        print(timeit.default_timer() - start)
        times.append(timeit.default_timer() - start)
        '''
        if(len(obstacles_in_range)):
            print(planner.optimal_control)
            header = ['V(Linear Velocity)', 'W(Angular Velocity)', 'goal_reaching_cost', 'MMD cost', 'Avoided Samples']
            with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                table=np.round(np.vstack((bot.lin_ctrl,bot.ang_ctrl,planner.goal_reaching_cost,planner.coll_avoidance_cost,planner.optimizer.avoided_samples)).T,4)
                writer.writerows(table)
            break    
        '''
        #print(np.array(times).mean())
        #bot.set_linear_acceleration(control)
        #.append(bot.get_linear_velocity())
        ax.add_artist(plt.Circle(bot.get_position(), bot.radius, facecolor='#059efb', edgecolor='black', zorder=100))
        plt.plot(np.array(bot.path)[:,0], np.array(bot.path)[:,1], '#059efb', zorder=2)
        plt.arrow(bot.get_position()[0], bot.get_position()[1], 1.5*bot.get_velocity()[0], 1.5*bot.get_velocity()[1],length_includes_head=True, head_width=0.3, head_length=0.2)
        itr=random.sample(range(10000),samples_to_plot)
        for i in range(samples_to_plot):
            ax.add_artist(plt.Circle(bot.position_samples[itr[i],:], bot.radius, color='#059efb', zorder=3, alpha=0.08))
        
        for j in range(len(obstacles)):
            for i in range(samples_to_plot):
                ax.add_artist(plt.Circle(obstacles[j].position_samples[itr[i],:], obstacles[j].radius, color='#ffa804', zorder=2, alpha=0.08))
        plt.draw()
        plt.pause(0.001)
        if(save==1):
            plt.gcf().savefig('../MMD Python Outputs/{}.png'.format( str(int(counter)).zfill(4)), dpi=300)
        if len(obstacles_in_range) > 0 and dist==1:
            for i in range(len(obstacles_in_range)):
                fig = plt.figure()
                plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
                cones=bot.collision_cones(obstacles[i],100)
                #cones[cones<0]=0
                ax = kdeplot(cones, label='Final Cones', shade=True, color='#ffa804')
                #ax.axvline(x=0)
                ax.set_xlim((-25, 25))
                ax.set_ylim((0, 0.3))
                # ax.axvline(x=0)
                fig.savefig('../MMD Python Outputs/dist-{}.png'.format( str(int(counter)).zfill(4)), dpi=300)
                # plt.show()
                plt.close(fig)

if __name__ == '__main__':
    main()