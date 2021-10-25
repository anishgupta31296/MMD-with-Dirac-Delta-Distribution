import numpy as np
def kernelRBF(dist, dist1):
	g=0.1
	k=np.exp(-g*(np.sum((dist-dist1)**2, axis=1)))	
	return k