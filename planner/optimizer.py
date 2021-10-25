#!/usr/bin/env
import numpy as np
import torch

class MMD:
    def __init__(self, gamma, reduced_samples, device):
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
        
    def collision_cones(self, lin_ctrl, ang_ctrl,head_samples, v, w,ap,op,ov ,R,dt,control_samples):
        lin_ctrl=lin_ctrl.reshape(lin_ctrl.shape[0],1)
        ang_ctrl=ang_ctrl.reshape(ang_ctrl.shape[0],1)        
        lin_ctrl=np.tile(lin_ctrl,(1,self.reduced_samples_grid))
        ang_ctrl=np.tile(ang_ctrl,(1,self.reduced_samples_grid))
        ax=ap[:,0].reshape(1,self.reduced_samples_grid)
        ay=ap[:,1].reshape(1,self.reduced_samples_grid)
        ox=op[:,0].reshape(1,self.reduced_samples_grid)
        oy=op[:,1].reshape(1,self.reduced_samples_grid)
        ovx=ov[:,0].reshape(1,self.reduced_samples_grid)
        ovy=ov[:,1].reshape(1,self.reduced_samples_grid)
        head_samples=head_samples.reshape(1,self.reduced_samples_grid)
        v_samples=control_samples[:,0].reshape(1,self.reduced_samples_grid)
        w_samples=control_samples[:,1].reshape(1,self.reduced_samples_grid)
        #print(lin_ctrl.shape,head_samples.shape,ang_ctrl.shape,control_samples.shape,ovx.shape)
        vx=(v+lin_ctrl+v_samples)*np.cos(head_samples+(w+ang_ctrl+w_samples)*dt)-ovx
        vy=(v+lin_ctrl+v_samples)*np.sin(head_samples+(w+ang_ctrl+w_samples)*dt)-ovy
        rx=ax-ox
        ry=ay-oy
        return (vx*rx + vy*ry)**2 + (vx**2 + vy**2)*((R)**2 - (rx**2 + ry**2))

    def MMD_dirac_delta_cost(self,a,a_coeffs):
        kernel_xx = torch.exp(-self.gamma*torch.pow(a@self.ones_mat-(a@self.ones_mat).transpose(2,1),2)) 
        kernel_xy = torch.exp(-self.gamma*torch.pow(a@self.ones_mat-self.desired_mat.T,2))
        mmd_term1 = a_coeffs @ kernel_xx.float() @ a_coeffs.T
        mmd_term2 = a_coeffs @ kernel_xy.float() @ (self.ones_mat/self.dirac_delta_distribution.shape[0]).T.float()
        mmd = mmd_term1 - 2*mmd_term2 + self.mmd_term3 
        return mmd.cpu().numpy()