#!/usr/bin/env python

import numpy as np
import seaborn as sns
from random import sample
from planner.optimizer import MMD



class Planner:

    def __init__(self,gamma=0.1,reduced_samples=20, device='cpu'):
        self.noise_samples = None
        self.optimal_control = None
        self.reduced_samples=reduced_samples
        self.optimizer = MMD(gamma,reduced_samples, device)
        i,j=np.meshgrid(range(self.reduced_samples),range(self.reduced_samples))
        self.i=i.flatten()
        self.j=j.flatten()


        
    def get_coll_avoidance_cost(self,Agent,Obstacles):
        self.noise_samples=Agent.noise_samples
        v_noise=Agent.linear_velocity_noise+Agent.linear_velocity_control_noise
        w_noise=Agent.angular_velocity_noise+Agent.angular_velocity_control_noise
        control_samples,control_coeff=self.reduced_sets_method(np.hstack((v_noise.reshape(self.noise_samples,1), w_noise.reshape(self.noise_samples,1), Agent.position_samples, Agent.head_samples.reshape(self.noise_samples,1))),self.reduced_samples)
        agent_p_samples=control_samples[:,2:4,:]
        head_samples=control_samples[:,4,:]
        control_samples=control_samples[:,0:2,:]
        Agent.reduced_position_noise=agent_p_samples[self.j,:,:]
        Agent.reduced_controls_samples=control_samples[self.j,:,:]
        Agent.reduced_head_samples=head_samples[self.j]
        Agent.control_coeff=control_coeff[self.j]
        MMD_cost=[]
        for x in range(len(Obstacles)):
            MMD_cost.append(np.zeros(control_samples.shape[0]))
            obs_para, obs_coeff=self.reduced_sets_method(np.hstack((Obstacles[x].position_samples, Obstacles[x].velocity_samples)) ,self.reduced_samples) 
            obs_velocity=obs_para[:,2:4,:]
            obs_position=obs_para[:,0:2,:]
            Obstacles[x].reduced_position_noise=obs_position[self.i,:,:]
            Obstacles[x].reduced_velocity_noise=obs_velocity[self.i,:,:]
            Obstacles[x].reduced_coeffs=obs_coeff[self.i]
            R=Agent.radius+Obstacles[x].radius
            MMD_cost[x]=self.optimizer.get_cost(Agent,Obstacles[x])
        return MMD_cost    

    def get_controls(self,Agent,Obstacles, alpha, beta):
        Agent.sample_controls()
        self.goal_reaching_cost=Agent.get_desired_velocity_cost()
        self.coll_avoidance_cost=0
        if(len(Obstacles)>0):
            coll_avoidance_cost_list=self.get_coll_avoidance_cost(Agent,Obstacles)
            self.coll_avoidance_cost=np.sum(coll_avoidance_cost_list,axis=0).reshape(len(coll_avoidance_cost_list[0]))
        cost=alpha*self.coll_avoidance_cost+beta*self.goal_reaching_cost

        indcs=np.argmin(cost)
        #print(indcs)
        #if(isinstance(indcs,list)):
        #    indcs=sample(indcs,1)
        self.optimal_control=[Agent.lin_ctrl[indcs],Agent.ang_ctrl[indcs]]

    def reduced_sets_method(self, dist,target_size):
        dist_shape=dist.shape
        big_sample_size_row=dist_shape[0]
        if(len(dist_shape)==2):
            big_sample_size_col=dist_shape[1]
            dist=dist.reshape(big_sample_size_row,big_sample_size_col,1)
            no_o=1
        elif(len(dist_shape)==1):
            dist=dist.reshape(big_sample_size_row,1,1)
            big_sample_size_col=1
            no_o=1  
        else:
            big_sample_size_col=dist_shape[1]
            no_o=dist_shape[2]
        idx = np.arange(big_sample_size_row)
        np.random.shuffle(idx)
        reduced_dist=dist[idx[:target_size],:,:]
        reduced_dist_coeff=[]
        for x in range(no_o):
            kz=self.generate_kernel_matrix(reduced_dist[:,:,x],reduced_dist[:,:,x])
            kzx=self.generate_kernel_matrix(reduced_dist[:,:,x], dist[:,:,x])
            alpha=np.ones(big_sample_size_row)
            coeff=(np.linalg.inv(kz)@kzx@alpha)/big_sample_size_row
            reduced_dist_coeff=np.append(reduced_dist_coeff,coeff)
        return reduced_dist,reduced_dist_coeff

    def generate_kernel_matrix(self, dist, dist1):
        r1=dist.shape[0]
        r2=dist1.shape[0]
        x,y=np.meshgrid(range(r1), range(r2))

        x=x.T.flatten()
        y=y.T.flatten()
        kernel_matrix=self.kernelRBF(dist[x,:], dist1[y,:]).reshape(r1, r2)
        return kernel_matrix
    
    def kernelRBF(self, dist, dist1):
        g=0.1
        k=np.exp(-g*(np.sum((dist-dist1)**2, axis=1)))  
        return k        