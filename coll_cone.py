import numpy as np
def coll_cones(u,head_samples, v, w,  ax, ay, ox, oy, ovx, ovy,R,dt,control_samples):
    vx=(v+u[0]+control_samples[:,0])*np.cos(head_samples+(w+u[1]+control_samples[:,1])*dt)-ovx
    vy=(v+u[0]+control_samples[:,0])*np.sin(head_samples+(w+u[1]+control_samples[:,1])*dt)-ovy
    rx=ax-ox
    ry=ay-oy
    return (vx*rx + vy*ry)**2 + (vx**2 + vy**2)*((R)**2 - (rx**2 + ry**2))
