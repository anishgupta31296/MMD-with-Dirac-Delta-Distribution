#!/usr/bin/env python

import numpy as np
import math
from .core import Agent

class NonHolonomicBot(Agent):

    def __init__(self, position, goal, noise_params, sensor_range=7, dt=1/10, noise_samples=10000, radius=1,   name='Bot'):
        self.min_linear_velocity = 0.4
        self.max_linear_velocity = 1.5
        self.max_angular_velocity = 0.8
        self.sensor_range = sensor_range
        self.head=math.atan2(goal[1]-position[1],goal[0]-position[0])
        self.linear_velocity = self.min_linear_velocity
        self.angular_velocity = 0
        velocity=self.linear_velocity*np.array([np.cos(self.head),np.sin(self.head)])
        Agent.__init__(self, position, velocity, goal, noise_params, noise_samples, radius, dt, name)
        self.noise_params['linear_velocity']={'weights':noise_params['velocity']['weights'], 'means':noise_params['velocity']['means'][0,:] , 'stds':noise_params['velocity']['stds'][0,:] }
        self.noise_params['angular_velocity']={'weights':noise_params['velocity']['weights'], 'means':noise_params['velocity']['means'][1,:] , 'stds':noise_params['velocity']['stds'][1,:] }
        self.noise_params['linear_velocity_controls']={'weights':noise_params['controls']['weights'], 'means':noise_params['controls']['means'][0,:] , 'stds':noise_params['controls']['stds'][0,:] }
        self.noise_params['angular_velocity_controls']={'weights':noise_params['controls']['weights'], 'means':noise_params['controls']['means'][1,:] , 'stds':noise_params['controls']['stds'][1,:] }
        self.linear_velocity_control_bounds = np.array([-0.2,0.2])
        self.angular_velocity_control_bounds = np.array([-0.2,0.2])
        velocity_noise=self.get_noise_samples(noise_params['velocity'])
        self.linear_velocity_noise = velocity_noise[:,0]
        self.angular_velocity_noise = velocity_noise[:,1]
        controls_noise=self.get_noise_samples(noise_params['controls'])
        self.linear_velocity_control_noise = controls_noise[:,0]
        self.angular_velocity_control_noise = controls_noise[:,1]
        self.head_noise=self.get_noise_samples(noise_params['head'])
        self.reduced_controls_samples=None
        self.reduced_head_noise=None
        self.linear_velocity_samples=self.linear_velocity+self.linear_velocity_noise
        self.angular_velocity_samples=self.angular_velocity+self.angular_velocity_noise
        self.head_noise=self.head_noise.reshape(self.head_noise.shape[0])
        self.head_samples=self.head+self.head_noise
        self.update_samples()

    def update_noise(self):
        self.linear_velocity_noise=self.linear_velocity_noise+self.linear_velocity_control_noise*self.dt
        self.angular_velocity_noise=self.angular_velocity_noise+self.angular_velocity_control_noise*self.dt
        controls=np.vstack((self.linear_velocity_noise,self.angular_velocity_noise)).T
        dist_boolean_list=np.linalg.norm(controls, axis=1)>4.5*np.linalg.norm(self.noise_params['velocity']['stds'])
        new_samples_count=np.sum(dist_boolean_list)
        if(new_samples_count>10):
            controls[dist_boolean_list,:]= self.get_noise_samples(self.noise_params['velocity'], samples=new_samples_count)
        self.linear_velocity_noise=controls[:,0]
        self.angular_velocity_noise=controls[:,1]   
        controls_noise=self.get_noise_samples(self.noise_params['controls'])
        self.linear_velocity_control_noise = controls_noise[:,0]
        self.angular_velocity_control_noise = controls_noise[:,1]
        self.velocity_noise=self.linear_velocity_samples[...,np.newaxis]*np.array([np.cos(self.head_samples+self.angular_velocity_samples*self.dt),np.sin(self.head_samples+self.angular_velocity_samples*self.dt)]).T-self.velocity
        self.update_samples()
        super().update_noise()

    def update_samples(self):
        self.linear_velocity_samples=self.linear_velocity+self.linear_velocity_noise        
        self.angular_velocity_samples=self.angular_velocity+self.angular_velocity_noise
        self.head_samples=self.head+self.head_noise

    def in_sensor_range(self, obstacle):
        if (self.get_position() - obstacle.get_position()).__pow__(2).sum() < self.sensor_range**2:
            r = self.get_position() - obstacle.get_position()
            v = self.get_velocity() - obstacle.get_velocity()
            if r @ v < 0:
                return True
        return False

    def get_linear_velocity(self):
        return self.linear_velocity

    def get_angular_velocity(self):
        return self.angular_velocity
        
    def get_bounds(self):
        return [self.max_linear_velocity, self.max_angular_velocity]

    def get_min_linear_velocity(self):
        return self.min_linear_velocity

    def get_linear_velocity_control_bounds(self):
        return self.linear_velocity_control_bounds

    def get_angular_velocity_control_counds(self):
        return self.angular_velocity_control_bounds
        
    def get_desired_velocity_cost(self):
        bounds=self.get_bounds()
        '''        
        desired_velocity=bounds[0]*(self.get_goal()-self.position_noise).T/np.linalg.norm(self.get_goal()-self.position_noise, axis=1)
        linear_velocity=self.linear_velocity+self.linear_velocity_noise[np.newaxis,...]+self.lin_ctrl[..., np.newaxis]+self.linear_velocity_noise[np.newaxis,...]
        start= time.time()
        print(time.time()-start)
        angular_velocity=self.angular_velocity+self.angular_velocity_noise[np.newaxis,...]+self.ang_ctrl[..., np.newaxis]+self.angular_velocity_noise[np.newaxis,...]
        head=self.head+angular_velocity*self.dt
        velocity=linear_velocity*np.cos(head)
        velocity=velocity[:,np.newaxis,:]
        velocity1=linear_velocity*np.sin(head)
        velocity1=velocity1[:,np.newaxis,:]
        velocity=np.append(velocity,velocity1, axis=1)
        cost=np.linalg.norm(velocity-desired_velocity,axis=1)
        '''
        pos_noise=self.position
        #pos_noise=np.mean(self.position_noise,axis=0)
        desired_velocity=bounds[0]*(self.get_goal()-pos_noise).T/np.linalg.norm(self.get_goal()-pos_noise)
        linear_velocity=self.linear_velocity+np.mean(self.linear_velocity_noise)+self.lin_ctrl+np.mean(self.linear_velocity_noise)
        angular_velocity=self.angular_velocity+np.mean(self.angular_velocity_noise)+self.ang_ctrl+np.mean(self.angular_velocity_noise)
        head=self.head+angular_velocity*self.dt
        cost=np.linalg.norm((linear_velocity*np.vstack((np.cos(head),np.sin(head)))).T-desired_velocity,axis=1)
        return cost        

    def set_controls(self, controls):
        bounds=self.get_bounds()
        controls[0]=controls[0]+self.get_linear_velocity()
        controls[0] = max(self.get_min_linear_velocity(), controls[0])
        controls[0] = min(bounds[0],controls[0])
        self.linear_velocity = controls[0]
        controls[1]=controls[1]+self.get_angular_velocity()
        controls[1] = max(-bounds[1], controls[1])
        controls[1] = min(bounds[1], controls[1])
        self.angular_velocity = controls[1]
        self.head=self.head+controls[1]*self.dt
        velocity=self.get_linear_velocity()*np.array([np.cos(self.head),np.sin(self.head)])
        self.set_velocity(velocity)
        self.update_noise()
  
    def sample_controls(self,samples=20):
        agent_v=np.array([self.get_linear_velocity(),self.get_angular_velocity()])
        bounds=self.get_bounds()
        v_ctr_bounds=self.get_linear_velocity_control_bounds()
        w_ctr_bounds=self.get_angular_velocity_control_counds()
        vel_cap_max=bounds[0]
        vel_cap_min=self.get_min_linear_velocity()
        w_cap=bounds[1]
        lb=[v_ctr_bounds[0],w_ctr_bounds[0]];
        ub=[v_ctr_bounds[1],w_ctr_bounds[1]];
        if(agent_v[0]>(vel_cap_max-ub[0])):
            ub[0]=vel_cap_max-agent_v[0]
        elif(agent_v[0]<(vel_cap_min-lb[0])):    
            lb[0]= -(agent_v[0]-vel_cap_min)
        if(agent_v[1]>(w_cap-ub[1])):
            ub[1]=w_cap-agent_v[1]
        elif(agent_v[1]<(-w_cap-lb[1])):    
            lb[1]= -(agent_v[1]+w_cap)   
        v_list=np.linspace(lb[0],ub[0],samples)
        w_list=[];
        for k in range(len(v_list)):
             if((agent_v[0]+v_list[k])<w_cap):
              max_w=agent_v[0]+v_list[k]
             else:
              max_w=w_cap
             if(abs(agent_v[1]+lb[1])>max_w):
              l_bound=-max_w-agent_v[1]
             else:
              l_bound=lb[1]

             if(abs(agent_v[1]+ub[1])>max_w):
              u_bound= max_w-agent_v[1]
             else:
              u_bound=ub[1]
             w_list.append(np.linspace(l_bound,u_bound,samples))
        xv, yv = np.meshgrid(range(samples), range(samples))
        yv=yv.flatten()
        v_list=v_list[yv]
        v_list=np.append(v_list,vel_cap_min)
        w_list=np.append(w_list,0)
        self.lin_ctrl=v_list
        self.ang_ctrl=w_list
        #self.controls = np.vstack((v_list,w_list)).T
        
class HolonomicBot(Agent):
    def __init__(self, position, goal, noise_params, noise_samples=10000, radius=1, dt=1/10, sensor_range=7, name='Bot'):
        self.sensor_range = sensor_range
        Agent.__init__(self, position, goal, noise_params, noise_samples, radius, dt, name)
        self.x_velocity_control_bounds = np.array([-1.5,1.5])
        self.y_velocity_control_bounds = np.array([-1.5,1.5])
        self.x_velocity_bounds = np.array([-0.2,0.2])
        self.y_velocity_bounds = np.array([-0.2,0.2])

    def in_sensor_range(self, obstacle):
        if (self.get_position() - obstacle.get_position()).__pow__(2).sum() < self.sensor_range**2:
            r = self.get_position() - obstacle.get_position()
            v = self.get_velocity() - obstacle.get_velocity()
            if r @ v < 0:
                return True
        return False

        
    def get_x_velocity_bounds(self):
        return self.x_velocity_bounds

    def get_y_velocity_bounds(self):
        return self.y_velocity_bounds

    def get_x_velocity_control_bounds(self):
        return self.x_velocity_control_bounds

    def get_y_velocity_control_counds(self):
        return self.y_velocity_control_bounds
        
    def set_controls(self, controls):
        vx_bnds=self.get_x_velocity_bounds()
        vy_bnds=self.get_y_velocity_bounds()
        controls=controls+self.get_velocity()
        controls[0] = max(vx_bnds[0], controls[0])
        controls[0] = min(vx_bnds[1], controls[0])
        controls[1] = max(vy_bnds[0], controls[1])
        controls[1] = min(vx_bnds[1], controls[1])
        self.set_velocity(controls)
  
    def sample_controls(self,samples=20):
        agent_v=np.array([self.get_linear_velocity(),self.get_angular_velocity()])
        vx_bounds=self.get_x_velocity_control_bounds()
        vy_bounds=self.get_y_velocity_control_bounds()
        x,y=np.meshgrid(range(samples),range(samples))
        x=x.flatten()
        y=y.flatten()
        vx_list=np.linspace(vx_bounds[0],vx_bounds[1],samples)
        vy_list=np.linspace(vy_bounds[0],vy_bounds[1],samples)
        vx_list=vx_list[x]
        vy_list=vy_list[y]
        controls = np.vstack((vx_list,vy_list)).T
        return self.controls
        print('55')
