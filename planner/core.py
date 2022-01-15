#!/usr/bin/env python

import numpy as np
import seaborn as sns
from random import sample
import copy

class Planner:

    def __init__(self,param=0.1,samples_param=20,optimizer='Gaussian Approximation',device='cpu',gaussian_approximation=False):
        self.noise_samples = None
        self.optimal_control = None
        self.reduced_samples=0
        self.constraint=0
        self.obs_priority=0
        self.gaussian_approximation=gaussian_approximation
        self.final_cones=[]
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
            self.obs_priority=1
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
                self.final_cones.append(self.optimizer.cones)
                cost[x]=self.optimizer.get_cost(Agent,Obstacles[x])
        else:
            for x in range(len(Obstacles)):
                cost.append(np.zeros(Agent.controls_samples.shape[0]))
                cost[x]=self.optimizer.get_cost(Agent,Obstacles[x])
                self.final_cones.append(self.optimizer.cones)

            
        return cost    

    @staticmethod
    def gaussian_approx(dist):
        means=np.mean(dist,axis=0)
        stds=np.std(dist,axis=0)
        return np.random.normal(means,stds,dist.shape)

    def get_controls(self,Bot,Obs, alpha, beta, delta):
        Agent=copy.copy(Bot)
        Obstacles=copy.copy(Obs)
        if(self.gaussian_approximation):
            Agent.position_samples=self.gaussian_approx(Agent.position_samples)
            Agent.controls_samples=self.gaussian_approx(Agent.controls_samples)
            Agent.head_samples=self.gaussian_approx(Agent.head_samples)
            for Obstacle in Obstacles:
                Obstacle.position_samples=self.gaussian_approx(Obstacle.position_samples)
                print('OBS:',np.mean(Obstacle.position_samples,axis=0)-Obstacle.position)
                Obstacle.velocity_samples=self.gaussian_approx(Obstacle.velocity_samples)
        Agent.sample_controls()
        self.goal_reaching_cost=Agent.get_desired_velocity_cost()
        self.goal_reaching_cost=(self.goal_reaching_cost-np.amin(self.goal_reaching_cost))/(np.amax(self.goal_reaching_cost)-np.amin(self.goal_reaching_cost))
        self.coll_avoidance_cost=0
        if(len(Obstacles)>0):
            coll_avoidance_cost_list=self.get_coll_avoidance_cost(Agent,Obstacles)
            if(self.constraint==0):
                if(self.obs_priority==1 and len(Obstacles)>1):
                    avoided_samples=np.zeros((len(Obstacles),len(self.final_cones[0])))
                    for i in range(len(Obstacles)>1):
                        avoided_samples[i,:]=np.sum(self.final_cones[0]<0,axis=1)/len(self.final_cones[0])
                    avoided_samples_bool=avoided_samples>0.95
                    avoided_samples_bool_sum=np.sum(avoided_samples_bool,axis=0)
                    for i in range(len(self.final_cones[0])):
                        if(avoided_samples_bool_sum[i]==len(Obstacles)):
                            min_ind=np.argmin(avoided_samples[i,:])
                            avoided_samples_bool[min_ind]=False
                    coll_avoidance_cost_list=np.array(coll_avoidance_cost_list)
                    coll_avoidance_cost_list[avoided_samples_bool]=coll_avoidance_cost_list[avoided_samples_bool]*0.000001       
                
                self.coll_avoidance_cost=np.sum(coll_avoidance_cost_list,axis=0).reshape(len(coll_avoidance_cost_list[0]))
                cost=alpha*self.coll_avoidance_cost+beta*self.goal_reaching_cost
                indcs=np.argmin(cost)
            else:
                self.coll_avoidance_cost=coll_avoidance_cost_list
                cons=np.array(coll_avoidance_cost_list).T
                cost=beta*self.goal_reaching_cost+delta*(Agent.ang_ctrl**2)

                if(np.any((cons<0).all(axis=1))):
                    min_cost=np.min(cost[(cons<0).all(axis=1)])
                    indcs=np.where(cost==min_cost)[0]
                    if(len(indcs)>1):
                        indcs=indc[np.random.choice(len(indcs))]
                    else:
                        indcs=indcs[0]
                else:
                    max_cons=np.max(cons,axis=1) + delta*(Agent.ang_ctrl**2 + Agent.lin_ctrl**2)
                    indcs=np.argmin(max_cons)
                    #print(max_cons[indcs])
                
        else:
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