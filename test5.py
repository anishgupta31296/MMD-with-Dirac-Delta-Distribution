import numpy as np
from seaborn import kdeplot
import matplotlib.pyplot as plt

a=np.load('test.npy')
b=np.load('controls.npy')
b=np.round(b,2)
print(np.round(b,2))
c=np.where((b==(-0.2,0.18)).all(axis=1))[0][0]
cones=a[c]
fig = plt.figure()
plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
ax = kdeplot(cones, label='Final Cones', shade=True, color='#ffa804')
plt.show()