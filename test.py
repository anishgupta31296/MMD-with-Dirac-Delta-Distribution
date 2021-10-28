import numpy as np
import timeit
samples=1000
R=2
v=np.random.randn(samples,2).reshape(samples,1,2)
r=np.random.randn(samples,2).reshape(samples,1,2)
vo=np.random.randn(samples,2).reshape(1,samples,2)
ro=np.random.randn(samples,2).reshape(1,samples,2)
start = timeit.default_timer()
vr=v-vo
rr=r-ro
cones=np.square(np.sum(vr*rr, axis=2))+ np.sum(np.square(vr), axis=2)*((R)**2 - np.sum(np.square(rr), axis=2))
cones=cones.flatten()
print(cones.shape,timeit.default_timer() - start)
