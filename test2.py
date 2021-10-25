import numpy as np
import torch
import timeit
import MMD_cost

a=np.random.randn(100,625)
b=np.random.randn(1,625)
a = torch.tensor(np.array(a), device="cuda:0")
b = torch.tensor(np.array(b), device="cuda:0")
a_coeff=torch.ones((1, a.shape[1]), device="cuda:0")
c=MMD_cost.MMD_cost(a,b,a_coeff)

a=np.random.randn(100,625)
b=np.random.randn(1,625)
a = torch.tensor(np.array(a), device="cuda:0")
b = torch.tensor(np.array(b), device="cuda:0")
c=MMD_cost.MMD_cost(a,b,a_coeff)
