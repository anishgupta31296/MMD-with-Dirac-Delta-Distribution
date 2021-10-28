import numpy as np
import timeit

samples=25
R=2
reduced_samples_grid=samples*samples
dt=0.1


v=np.random.randn(samples,1)
w=np.random.randn(samples,1)
h=np.random.randn(samples,1)
r=np.random.randn(samples,2)
vo=np.random.randn(samples,2)
ro=np.random.randn(samples,2)
l_ctrl=np.random.randn(400,1)
a_ctrl=np.random.randn(400,1)
start = timeit.default_timer()
i,j=np.meshgrid(range(samples), range(samples))
i=i.flatten()
j=j.flatten()
ap=r[i]
op=ro[j]
ov=vo[j]
head_samples=h[i]
v_samples=v[i]
w_samples=w[i]
lin_ctrl=l_ctrl.reshape(l_ctrl.shape[0],1)
ang_ctrl=a_ctrl.reshape(a_ctrl.shape[0],1)        
lin_ctrl=np.tile(lin_ctrl,(1,reduced_samples_grid))
ang_ctrl=np.tile(ang_ctrl,(1,reduced_samples_grid))
ax=ap[:,0].reshape(1,reduced_samples_grid)
ay=ap[:,1].reshape(1,reduced_samples_grid)
ox=op[:,0].reshape(1,reduced_samples_grid)
oy=op[:,1].reshape(1,reduced_samples_grid)
ovx=ov[:,0].reshape(1,reduced_samples_grid)
ovy=ov[:,1].reshape(1,reduced_samples_grid)
head_samples=head_samples.reshape(1,reduced_samples_grid)
v_samples=v_samples.reshape(1,reduced_samples_grid)
w_samples=w_samples.reshape(1,reduced_samples_grid)
vx=(lin_ctrl+v_samples)*np.cos(head_samples+(ang_ctrl+w_samples)*dt)-ovx
vy=(lin_ctrl+v_samples)*np.sin(head_samples+(ang_ctrl+w_samples)*dt)-ovy
rx=ax-ox
ry=ay-oy
cones1=(vx*rx + vy*ry)**2 + (vx**2 + vy**2)*((R)**2 - (rx**2 + ry**2))
print('1D',timeit.default_timer() - start)


v=np.random.randn(samples,1)
w=np.random.randn(samples,1)
h=np.random.randn(samples,1)
r=np.random.randn(samples,2)
vo=np.random.randn(samples,2)
ro=np.random.randn(samples,2)
l_ctrl=np.random.randn(400,1)
a_ctrl=np.random.randn(400,1)
start = timeit.default_timer()
r1=r.reshape(1,samples,1,2)
vo1=vo.reshape(1,1,samples,2)
ro1=ro.reshape(1,1,samples,2)
v1=v.reshape(1,samples,1,1)
w1=w.reshape(1,samples,1,1)
h1=h.reshape(1,samples,1,1)
l_ctrl1=l_ctrl.reshape(400,1,1,1)
a_ctrl1=a_ctrl.reshape(400,1,1,1)
nh_v=(l_ctrl1+v1)*np.concatenate((np.cos(h1+(a_ctrl1+w1)*dt), np.sin(h1+(a_ctrl1+w1)*dt)),axis=3)
vr=nh_v-vo1
rr=r1-ro1
cones=np.square(np.sum(vr*rr, axis=3))+ np.sum(np.square(vr), axis=3)*((R)**2 - np.sum(np.square(rr), axis=3))
cones=cones.reshape(400,reduced_samples_grid)
print('2D',timeit.default_timer() - start)



print(np.sum(cones1-cones))