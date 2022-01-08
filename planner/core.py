#!/usr/bin/env python

import numpy as np
import seaborn as sns
from random import sample

class Planner:

    def __init__(self,param=0.1,samples_param=20,optimizer='Gaussian Approximation',device='cpu'):
        self.noise_samples = None
        self.optimal_control = None
        self.reduced_samples=0
        self.constraint=0
        if(optimizer=='MMD Dirac Delta'):
            from planner.optimizer import MMD_Dirac_Delta
            self.optimizer = MMD_Dirac_Delta(param,samples_param, device)
            self.reduced_samples=samples_param
        elif(optimizer=='MMD'):
            from planner.optimizer import MMD
            self.optimizer = MMD(param,samples_param, device)
            self.reduced_samples=samples_param
        elif(optimizer=='KLD'):
            from planner.optimizer import KLD
            self.optimizer = KLD(param,samples_param, device)

        else:
            from planner.optimizer import PVO
            self.constraint=1
            self.optimizer = PVO(param,samples_param, device)

        i,j=np.meshgrid(range(self.reduced_samples),range(self.reduced_samples))
        self.i=i.flatten()
        self.j=j.flatten()


        
    def get_coll_avoidance_cost(self,Agent,Obstacles):
        cost=[]
        if(self.reduced_samples>0):
            self.noise_samples=Agent.noise_samples
            control_samples,control_coeff=self.reduced_sets_method(np.hstack((Agent.controls_samples, Agent.position_samples, Agent.head_samples.reshape(self.noise_samples,1))),self.reduced_samples)
            agent_p_samples=control_samples[:,2:4,:]
            head_samples=control_samples[:,4,:]
            control_samples=control_samples[:,0:2,:]
            Agent.reduced_position_noise=agent_p_samples
            Agent.reduced_controls_samples=control_samples
            Agent.reduced_head_samples=head_samples
            Agent.control_coeff=control_coeff[self.i]
            for x in range(len(Obstacles)):
                cost.append(np.zeros(control_samples.shape[0]))
                obs_para, obs_coeff=self.reduced_sets_method(np.hstack((Obstacles[x].position_samples, Obstacles[x].velocity_samples)) ,self.reduced_samples) 
                obs_velocity=obs_para[:,2:4,:]
                obs_position=obs_para[:,0:2,:]
                Obstacles[x].reduced_position_noise=obs_position
                Obstacles[x].reduced_velocity_noise=obs_velocity
                Obstacles[x].reduced_coeffs=obs_coeff[self.j]
                R=Agent.radius+Obstacles[x].radius
                cost[x]=self.optimizer.get_cost(Agent,Obstacles[x])
        else:
            for x in range(len(Obstacles)):
                cost.append(np.zeros(Agent.controls_samples.shape[0]))
                cost[x]=self.optimizer.get_cost(Agent,Obstacles[x])
            
        return cost    

    def get_controls(self,Agent,Obstacles, alpha, beta, delta):
        Agent.sample_controls()
        self.goal_reaching_cost=Agent.get_desired_velocity_cost()
        self.goal_reaching_cost=(self.goal_reaching_cost-np.amin(self.goal_reaching_cost))/(np.amax(self.goal_reaching_cost)-np.amin(self.goal_reaching_cost))
        self.coll_avoidance_cost=0
        if(len(Obstacles)>0):
            coll_avoidance_cost_list=self.get_coll_avoidance_cost(Agent,Obstacles)
            self.coll_avoidance_cost=np.sum(coll_avoidance_cost_list,axis=0).reshape(len(coll_avoidance_cost_list[0]))
            if(self.constraint==0):
                print(2)
                cost=alpha*self.coll_avoidance_cost+beta*self.goal_reaching_cost
                indcs=np.argmin(cost)
            else:
                print(3)
                cons=self.coll_avoidance_cost 
                cost=beta*self.goal_reaching_cost+delta*(Agent.ang_ctrl**2)
                if(np.any(cons<0)):
                    min_cost=np.min(cost_list[cons<0])
                    indcs=np.where(cost_list==min_cost)
                    if(len(indcs)>1):
                        indcs=indc[np.random.choice(len(indcs))]
                    print(cons[indcs])
                else:
                    cost=beta*cons+delta*(Agent.ang_ctrl**2)
                    indcs=np.argmin(cons)
                    print(4,self.optimizer.mu[indcs],self.optimizer.sigma[indcs],cons[indcs])
        else:
            print(1)
            cost=alpha*self.coll_avoidance_cost+beta*self.goal_reaching_cost
            indcs=np.argmin(cost)
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