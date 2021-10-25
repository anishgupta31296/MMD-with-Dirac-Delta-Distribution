import numpy as np
import kernelRBF 
def generate_kernel_matrix(dist,dist1):
	r1=dist.shape[0]
	r2=dist1.shape[0]
	x,y=np.meshgrid(range(r1), range(r2))

	x=x.T.flatten()
	y=y.T.flatten()
	kernel_matrix=kernelRBF.kernelRBF(dist[x,:], dist1[y,:]).reshape(r1, r2)
	return kernel_matrix
