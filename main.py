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
    alpha=1
    beta=0.1
    gamma=0.1
    delta=0.1
    save=1
    dist=0
    samples_to_plot = 250
    steps=5
    times=[]
    obstacles = []
    counter = 0
    radius=0.65
    control_costs=[]
    dist=[]
    agent_noise_params = {
        'position': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([[-0.0, 0.0],[0.0,-0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        },
        'velocity': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([[-0.0, 0.0],[0.0,-0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        },
        'controls': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([[-0.0, 0.0],[0.0,-0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        },
        'head': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([0.0, 0.0]),
            'stds': np.array([0.001, 0.001])
        }        
    }

    obs_noise_params1 = {
        'position': {
            'weights': np.array([0.4, 0.6]),
            'means': np.array([[0.3, -0.2],[0.03,-0.04]]),
            'stds': np.array([[0.25, 0.05],[0.02,0.02]])
        },
        'velocity': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([[-0.0, 0.0],[-0.0,0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        }
    } 
    
    obs_noise_params2 = {
        'position': {
            'weights': np.array([0.4, 0.6]),
            'means': np.array([[-0.3, 0.2],[0.03,-0.04]]),
            'stds': np.array([[0.25, 0.05],[0.02,0.02]])
        },
        'velocity': {
            'weights': np.array([0.5, 0.5]),
            'means': np.array([[-0.0, 0.0],[-0.0,0.0]]),
            'stds': np.array([[0.01, 0.01],[0.01,0.01]])
        }
    }
    bot=NonHolonomicBot(np.array([0,0]), np.array([0, 25]), agent_noise_params, sensor_range=sensor_range)
    obstacles.append(Obstacle(position=np.array([-3.1,12]), goal=np.array([0,0]), noise_params=obs_noise_params2))
    obstacles.append(Obstacle(position=np.array([3.1,12]), goal=np.array([0,0]), noise_params=obs_noise_params1))
    obstacles.append(Obstacle(position=np.array([0,22]), goal=np.array([0,0]), noise_params=obs_noise_params2))
    obstacles.append(Obstacle(position=np.array([6.2,20]), goal=np.array([0,0]), noise_params=obs_noise_params1))
    obstacles.append(Obstacle(position=np.array([-6.2,20]), goal=np.array([0,0]), noise_params=obs_noise_params1))
    planner=Planner(param=1,samples_param=25,optimizer='KLD',device='cuda:0',gaussian_approximation=False)
    #planner=Planner(param=1.5,samples_param=25,optimizer='KLD',device='cpu',gaussian_approximation=False)
    while (bot.goal-bot.position).__pow__(2).sum() > 1:
        obstacles_in_range = []
        plt.clf()
        ax = plt.gcf().gca()
        ax.set_ylim((-5,25))
        ax.set_xlim((-15, 15))

        ax.add_artist(plt.Circle(bot.get_position(), bot.sensor_range-bot.radius, color='gray', alpha=0.1))
        for i, obs in enumerate(obstacles):
            if bot.in_sensor_range(obs):
                obstacles_in_range.append(obs)
            obs.set_velocity()
            ax.add_artist(plt.Circle(obs.get_position(), obs.radius, facecolor='#ffa804', edgecolor='black', zorder=3))
            plt.plot(np.array(obs.path)[:,0],np.array(obs.path)[:,1], '#ffa804', zorder=1)
            plt.text(obs.get_position()[0]-obs.radius/3, obs.get_position()[1]-obs.radius/3,str(i+1), fontsize = 12, zorder=4)

        counter=counter+1        
        start = timeit.default_timer()
        planner.get_controls(bot,obstacles_in_range, alpha, beta, delta)
        control_costs.append(planner.optimal_control[0]**2+planner.optimal_control[1]**2)
        
        for i in range(len(obstacles)):
            dist.append(np.linalg.norm(bot.position-obstacles[i].position))

    
        bot.set_controls(planner.optimal_control)

        #print(planner.optimal_control)
        print(bot.get_linear_velocity(),bot.get_angular_velocity())
        times.append(timeit.default_timer() - start)


        '''
        if(len(obstacles_in_range)):
            steps=steps-1
            header = ['V(Linear Velocity)', 'W(Angular Velocity)', 'goal_reaching_cost','MU','SIGMA' ,'PVO cost']
            with open(str(counter)+'.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                table=np.round(np.vstack((bot.lin_ctrl,bot.ang_ctrl,planner.goal_reaching_cost,planner.optimizer.mu,planner.optimizer.sigma,planner.coll_avoidance_cost)).T,4)
                writer.writerows(table)
            if(steps==0):
                break    
        
        if(len(obstacles_in_range)):
            print(bot.position)
            print(obstacles[0].position)
            print(obstacles[1].position)
            print(obstacles[2].position)
        '''
        ax.add_artist(plt.Circle(bot.get_position(), bot.radius, facecolor='#059efb', edgecolor='black', zorder=3))
        plt.plot(np.array(bot.path)[:,0], np.array(bot.path)[:,1], '#059efb', zorder=1)
        itr=random.sample(range(10000),samples_to_plot)
        for i in range(samples_to_plot):
            ax.add_artist(plt.Circle(bot.position_samples[itr[i],:], bot.radius, color='#059efb', zorder=2, alpha=0.08))
        
        for j in range(len(obstacles)):
            for i in range(samples_to_plot):
                ax.add_artist(plt.Circle(obstacles[j].position_samples[itr[i],:], obstacles[j].radius, color='#ffa804', zorder=2, alpha=0.08))

        plt.arrow(bot.get_position()[0], bot.get_position()[1], 1.5*bot.get_velocity()[0], 1.5*bot.get_velocity()[1],length_includes_head=True, head_width=0.3, head_length=0.2, zorder=4)
        plt.draw()
        plt.pause(0.001)
        if(save==1):
            plt.gcf().savefig('../MMD Python Outputs/{}.png'.format( str(int(counter)).zfill(4)), dpi=300)
        '''
        if(len(obstacles_in_range)==3):
            for i in range(len(obstacles_in_range)):
                np.save('test'+str(i)+'.npy',planner.final_cones[i])
                controls=np.vstack((bot.lin_ctrl,bot.ang_ctrl)).T
            np.save('controls.npy',controls)
            break
        '''
        if len(obstacles_in_range) > 0 and dist==1:
            for i in range(len(obstacles_in_range)):
                fig = plt.figure()
                plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
                cones=planner.final_cones
                #cones=bot.collision_cones(obstacles[i],100)
                #cones[cones<0]=0
                ax = kdeplot(cones, label='Final Cones', shade=True, color='#ffa804')
                #ax.axvline(x=0)
                ax.set_xlim((-25, 25))
                ax.set_ylim((0, 0.3))
                # ax.axvline(x=0)
                fig.savefig('../MMD Python Outputs/dist-{}.png'.format( str(int(counter)).zfill(4)), dpi=300)
                # plt.show()
                plt.close(fig)
    print(sum(control_costs),min(dist),counter)
if __name__ == '__main__':
    main()