import numpy as np
import torch
import timeit

def MMD_cost(a,b,a_coeff):
    '''
    kernel_xx = torch.exp(-0.1*torch.pow(a[:,x]-a[:,y],2)) 
    kernel_xy = torch.exp(-0.1*torch.pow(a[:,i]-b[:,j],2))
    kernel_yy = torch.exp(-0.1*torch.pow(b[:,e]-b[:,f],2))
    '''
    z=torch.ones((1,625), device="cuda:0").double()
    s=b.reshape(625,1)@z

    start = timeit.default_timer()
    kernel_xx = torch.exp(-0.1*torch.pow(a.reshape(100,625,1)@z-(a.reshape(100,625,1)@z).transpose(2,1),2)) 
    kernel_xy = torch.exp(-0.1*torch.pow(a.reshape(100,625,1)@z-s.T,2))
    mmd_term1 = a_coeff @ kernel_xx.float() @ a_coeff.T
    mmd_term2 = a_coeff @ kernel_xy.float() @ a_coeff.T
    stop = timeit.default_timer()

    kernel_yy = torch.exp(-0.1*torch.pow(b.reshape(625,1)@z-(b.reshape(625,1)@z).T,2))
    mmd_term3 = (torch.ones((1, a.shape[1]), device="cuda:0") @ kernel_yy.float() @ torch.ones((a.shape[1], 1), device="cuda:0"))

    mmd = mmd_term1 - 2*mmd_term2 + mmd_term3

    print('Time: ', stop - start)       
    print('Time for MMD')    
    return mmd
