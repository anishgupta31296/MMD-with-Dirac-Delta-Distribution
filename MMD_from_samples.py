import numpy as np
import reduced_sets_method
import coll_cone
import timeit
import MMD_cost
import torch

#Your statements here


def MMD_from_samples(agent,obs_p, obs_v, v_noise, w_noise, pos_noise, head_noise, reduced_samples,r1,r2, d):

     start = timeit.default_timer()
     obs_para, obs_coeff=reduced_sets_method.reduced_sets_method(np.hstack((obs_p[:,0,:], obs_p[:,1,:], obs_v[:,0,:], obs_v[:,1,:])) ,reduced_samples) 
     control_samples,control_coeff=reduced_sets_method.reduced_sets_method(np.hstack((v_noise, w_noise, pos_noise, head_noise)),reduced_samples)
     obs_velocity=obs_para[:,2:4,:]
     obs_position=obs_para[:,0:2,:]
     agent_p_samples=control_samples[:,2:4,:]
     head_samples=control_samples[:,4,:]
     control_samples=control_samples[:,0:2,:]
     i,j=np.meshgrid(range(obs_velocity.shape[0]),range(control_samples.shape[0]))
     i=i.flatten()
     j=j.flatten()
     obs_position=obs_position[i,:,:]
     obs_velocity=obs_velocity[i,:,:]
     obs_coeff=obs_coeff[i]
     control_samples=control_samples[j,:,:]
     agent_p_samples=agent_p_samples[j,:,:]
     head_samples=head_samples[j]
     control_coeff=control_coeff[j]
     R=r1+r2
     u=[0.1,0.1]
     dt=0.1

     cones= coll_cone.coll_cones(u,head_samples, agent.get_linear_velocity(), agent.get_angular_velocity(),  agent_p_samples[:,0], agent_p_samples[:,1], obs_position[:,0], obs_position[:,1], obs_velocity[:,0], obs_velocity[:,1],R,dt,control_samples)
     stop = timeit.default_timer()
     print('Time: ', stop - start)       
     print('Time for reduced sets+coll_cone')
start = timeit.default_timer()
a=np.random.randn(100,625)
b=np.random.randn(1,625)
x,y=np.meshgrid(range(a.shape[1]),range(a.shape[1]))
i,j=np.meshgrid(range(a.shape[1]),range(b.shape[1]))
e,f=np.meshgrid(range(b.shape[1]),range(b.shape[1]))
a = torch.tensor(np.array(a), device="cuda:0")
b = torch.tensor(np.array(b), device="cuda:0")
c=MMD_cost.MMD_cost(a,b,x,y,i,j,e,f)     
stop = timeit.default_timer()
print('Time: ', stop - start)    
start = timeit.default_timer()
a=np.random.randn(100,625)
b=np.random.randn(1,625)
a = torch.tensor(np.array(a), device="cuda:0")
b = torch.tensor(np.array(b), device="cuda:0")
c=MMD_cost.MMD_cost(a,b)     
stop = timeit.default_timer()
print('Time: ', stop - start) 
b=np.random.randn(625)
start = timeit.default_timer()
b=b.reshape(625,1)
b=b.reshape(625,1)
b=b.reshape(625,1)
b=b.reshape(625,1)
b=b.reshape(625,1)
b=b.reshape(625,1)
b=b.reshape(625,1)
b=b.reshape(625,1)
stop = timeit.default_timer()
print('Time: ', stop - start)