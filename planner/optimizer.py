#!/usr/bin/env
import numpy as np
import torch

class MMD:
    def __init__(self, gamma, reduced_samples, device):
        self.reduced_samples=reduced_samples
        self.reduced_samples_grid=reduced_samples*reduced_samples
        self.dirac_delta_distribution=np.random.randn(self.reduced_samples_grid,1)*0.00000000001
        self.gamma=gamma
        self.device = device
        self.ones_mat=np.ones((1,self.dirac_delta_distribution.shape[0]))
        self.desired_mat=self.dirac_delta_distribution@self.ones_mat
        self.desired_coeffs=self.ones_mat/self.dirac_delta_distribution.shape[0]
        self.kernel_yy = np.exp(-self.gamma*np.power(self.desired_mat-self.desired_mat.T,2))
        self.mmd_term3 = self.desired_coeffs @ self.kernel_yy @ self.desired_coeffs.T
        #print(torch.cuda.memory_allocated())
        self.mmd_term3 = torch.tensor(self.mmd_term3, device=self.device)
        self.ones_mat = torch.tensor(self.ones_mat, device=self.device)
        self.desired_mat=torch.tensor(self.desired_mat, device=self.device)
        #print(torch.cuda.memory_allocated())
        
    def get_cost(self,Agent,Obstacles):
        cones= self.collision_cones(Agent.lin_ctrl, Agent.ang_ctrl,Agent.reduced_head_samples, Agent.get_linear_velocity(), Agent.get_angular_velocity(),  Agent.reduced_position_noise, Obstacles.reduced_position_noise, Obstacles.reduced_velocity_noise,Agent.radius+Obstacles.radius,Agent.dt,Agent.reduced_controls_samples)
        cones[cones<0]=0
        cones=cones[..., np.newaxis]
        coeffs=Agent.control_coeff*Obstacles.reduced_coeffs
        coeffs=coeffs[np.newaxis,...]
        #coeffs=np.tile(coeffs.reshape(1,self.reduced_samples_grid),(Agent.lin_ctrl.shape[0],1)).reshape(Agent.lin_ctrl.shape[0],self.reduced_samples_grid,1)
        a = torch.tensor(cones, device=self.device)
        a_coeffs= torch.tensor(coeffs, device=self.device).float()
        c=self.MMD_dirac_delta_cost(a,a_coeffs)
        return c    
        
    def collision_cones(self, lin_ctrl, ang_ctrl,h, v, w,ap,op,ov ,R,dt,control_samples):
        r1=ap.reshape(1,self.reduced_samples,1,2)
        vo1=ov.reshape(1,1,self.reduced_samples,2)
        ro1=op.reshape(1,1,self.reduced_samples,2)
        h1=h.reshape(1,self.reduced_samples,1,1)
        v_c=control_samples[:,0].reshape(1,self.reduced_samples,1,1)
        w_c=control_samples[:,1].reshape(1,self.reduced_samples,1,1)
        l_ctrl1=lin_ctrl.reshape(lin_ctrl.shape[0],1,1,1)
        a_ctrl1=ang_ctrl.reshape(ang_ctrl.shape[0],1,1,1)
        nh_v=(l_ctrl1+v+v_c)*np.concatenate((np.cos(h1+(a_ctrl1+w+w_c)*dt), np.sin(h1+(a_ctrl1+w+w_c)*dt)),axis=3)
        vr=nh_v-vo1
        rr=r1-ro1
        cones=np.square(np.sum(vr*rr, axis=3))+ np.sum(np.square(vr), axis=3)*((R)**2 - np.sum(np.square(rr), axis=3))
        cones=cones.reshape(lin_ctrl.shape[0],self.reduced_samples_grid)
        print(cones.shape)
        return cones

    def MMD_dirac_delta_cost(self,a,a_coeffs):
        kernel_xx = torch.exp(-self.gamma*torch.pow(a@self.ones_mat-(a@self.ones_mat).transpose(2,1),2)) 
        kernel_xy = torch.exp(-self.gamma*torch.pow(a@self.ones_mat-self.desired_mat.T,2))
        mmd_term1 = a_coeffs @ kernel_xx.float() @ a_coeffs.T
        mmd_term2 = a_coeffs @ kernel_xy.float() @ (self.ones_mat/self.dirac_delta_distribution.shape[0]).T.float()
        mmd = mmd_term1 - 2*mmd_term2 + self.mmd_term3 
        return mmd.cpu().numpy()