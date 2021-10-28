import numpy as np
import timeit
import random
samples=1000
s=100
R=2
start = timeit.default_timer()
i=random.sample(range(1000),s)
j=random.sample(range(1000),s)
v=np.random.randn(samples,2)[i].reshape(s,1,2)
r=np.random.randn(samples,2)[i].reshape(s,1,2)
vo=np.random.randn(samples,2)[j].reshape(1,s,2)
ro=np.random.randn(samples,2)[j].reshape(1,s,2)
vr=v-vo
rr=r-ro
vr1=vr.reshape(s*s,2)
rr1=rr.reshape(s*s,2)
vx=vr1[:,0]
vy=vr1[:,1]
rx=rr1[:,0]
ry=rr1[:,1]
cones1=(vx*rx + vy*ry)**2 + (vx**2 + vy**2)*((R)**2 - (rx**2 + ry**2))
print('1D',cones1.shape,timeit.default_timer() - start)

start = timeit.default_timer()
i=random.sample(range(1000),s)
j=random.sample(range(1000),s)
v=np.random.randn(samples,2)[i].reshape(s,1,2)
r=np.random.randn(samples,2)[i].reshape(s,1,2)
vo=np.random.randn(samples,2)[j].reshape(1,s,2)
ro=np.random.randn(samples,2)[j].reshape(1,s,2)
vr=v-vo
rr=r-ro
cones=np.square(np.sum(vr*rr, axis=2))+ np.sum(np.square(vr), axis=2)*((R)**2 - np.sum(np.square(rr), axis=2))
cones=cones.flatten()
print('2D',cones.shape,timeit.default_timer() - start)



print(np.sum(cones1-cones))




