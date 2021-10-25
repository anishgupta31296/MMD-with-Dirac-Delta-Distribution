import numpy as np
import numpy.matlib
import timeit
'''
def collision_cones(lin_ctrl, ang_ctrl,head_samples, v, w,ap,op,ov ,R,dt,control_samples):
    lin_ctrl=lin_ctrl.reshape(lin_ctrl.shape[0],1)
    ang_ctrl=ang_ctrl.reshape(lin_ctrl.shape[0],1)        
    lin_ctrl=np.tile(lin_ctrl,(1,ap.shape[0]))
    ang_ctrl=np.tile(ang_ctrl,(1,ap.shape[0]))
    ax=ap[:,0].reshape(1,ap.shape[0])
    ay=ap[:,1].reshape(1,ap.shape[0])
    ox=op[:,0].reshape(1,op.shape[0])
    oy=op[:,1].reshape(1,op.shape[0])
    ovx=ov[:,0].reshape(1,ov.shape[0])
    ovy=ov[:,1].reshape(1,ov.shape[0])
    vx=(v+lin_ctrl+control_samples[:,0])*np.cos(head_samples+(w+ang_ctrl+control_samples[:,1])*dt)-ovx
    vy=(v+lin_ctrl+control_samples[:,0])*np.sin(head_samples+(w+ang_ctrl+control_samples[:,1])*dt)-ovy
    rx=ax-ox
    ry=ay-oy
    return (vx*rx + vy*ry)**2 + (vx**2 + vy**2)*((R)**2 - (rx**2 + ry**2))
'''
def collision_cones(lin_ctrl, ang_ctrl,head_samples, v, w,ap,op,ov ,R,dt,control_samples):
    start = timeit.default_timer()
    lin_ctrl=lin_ctrl.reshape(100,1)
    ang_ctrl=ang_ctrl.reshape(100,1)        
    lin_ctrl=np.tile(lin_ctrl,(1,625))
    ang_ctrl=np.tile(ang_ctrl,(1,625))
    ax=ap[:,0].reshape(1,625)
    ay=ap[:,1].reshape(1,625)
    ox=op[:,0].reshape(1,625)
    oy=op[:,1].reshape(1,625)
    ovx=ov[:,0].reshape(1,625)
    ovy=ov[:,1].reshape(1,625)
    print(timeit.default_timer() - start)
    vx=(v+lin_ctrl+control_samples[:,0])*np.cos(head_samples+(w+ang_ctrl+control_samples[:,1])*dt)-ovx
    vy=(v+lin_ctrl+control_samples[:,0])*np.sin(head_samples+(w+ang_ctrl+control_samples[:,1])*dt)-ovy
    rx=ax-ox
    ry=ay-oy
    return (vx*rx + vy*ry)**2 + (vx**2 + vy**2)*((R)**2 - (rx**2 + ry**2))

a=np.random.randn(100)
b=np.random.randn(100)


head_samples=np.random.randn(625)
v=2
w=0.2
ap=np.random.randn(625,2)
op=np.random.randn(625,2)
ov=np.random.randn(625,2)
R=1
dt=0.2
control_samples=np.random.randn(625,2)
c=collision_cones(a, b,head_samples, v, w,ap,op,ov ,R,dt,control_samples)

