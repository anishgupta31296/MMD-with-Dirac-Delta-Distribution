import numpy as np
import generate_kernel_matrix
def reduced_sets_method(dist ,target_size):
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
	reduced_dist_coeff=np.zeros((target_size,no_o))
	for x in range(no_o):
		kz=generate_kernel_matrix.generate_kernel_matrix(reduced_dist[:,:,x],reduced_dist[:,:,x])
		kzx=generate_kernel_matrix.generate_kernel_matrix(reduced_dist[:,:,x], dist[:,:,x])
		alpha=np.ones(big_sample_size_row)
		coeff=(np.linalg.pinv(kz)@kzx@alpha)/big_sample_size_row
		reduced_dist_coeff[:,x]=coeff
	return reduced_dist,reduced_dist_coeff
