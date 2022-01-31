import numpy as np
from seaborn import kdeplot
import matplotlib.pyplot as plt




'''
a0=np.load('test0.npy')
a1=np.load('test1.npy')
a2=np.load('test2.npy')

b=np.load('controls.npy')
b=np.round(b,2)
print(b)
v_list=np.unique(b[:,0])
v_list=np.sort(v_list)
w_list=np.unique(b[:,1])
w_list=np.sort(w_list)

c=np.where((b==(v_list[0],v_list[0])).all(axis=1))[0][0]
cones0=a0[c]
cones1=a1[c]
cones2=a2[c]

fig = plt.figure()

plt.subplot(1, 3, 1)
plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
ax = kdeplot(cones0, label='Final Cones', shade=True, color='#ffa804')
ax.set_xlim((-20, 15))
ax.set_title('Obstacle 1')

plt.subplot(1, 3, 2)
plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
ax = kdeplot(cones1, label='Final Cones', shade=True, color='#ffa804')
ax.set_xlim((-20, 15))
ax.set_title('Obstacle 2')

plt.subplot(1, 3, 3)
plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
ax = kdeplot(cones2, label='Final Cones', shade=True, color='#ffa804')
ax.set_xlim((-20, 15))
ax.set_title('Obstacle 3')
#plt.gcf().savefig('../MMD Python Outputs/{}.png'.format( str(int(counter)).zfill(4)), dpi=300)
plt.show()
'''
v=2
w=2
for x in range(v):
	for y in range(w):
		print(x,y)
		fig, big_axes = plt.subplots( figsize=(25.0, 15.0) , nrows=2, ncols=1, sharey=True) 
		title=['Correct Side','Incorrect Side']
		for row, big_ax in enumerate(big_axes, start=1):
		    big_ax.set_title("%s \n" % title[row-1], fontsize=16,fontweight="bold")
		    # Turn off axis lines and ticks of the big subplot 
		    # obs alpha is 0 in RGBA string!
		    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
		    # removes the white frame
		    big_ax._frameon = False
		a0=np.load('test0.npy')
		a1=np.load('test1.npy')
		a2=np.load('test2.npy')
		b=np.load('controls.npy')
		b=np.round(b,3)
		v_list=np.unique(b[:,0])
		v_list=np.sort(v_list)
		w_list=np.unique(b[:,1])
		w_list=np.sort(w_list)
		cones=[]
		c=np.where((b==(v_list[x],0.179)).all(axis=1))[0][0]
		cones.append(a0[c])
		cones.append(a1[c])
		cones.append(a2[c])
		c=np.where((b==(0,w_list[y])).all(axis=1))[0][0]
		cones.append(a0[c])
		cones.append(a1[c])
		cones.append(a2[c])

		for i in range(1,7):
			ax = fig.add_subplot(2,3,i)
			plt.arrow(0, 0, 0, 0.15,width=0.2,length_includes_head=True, head_width=0.7, head_length=0.005,color='black')
			avoiding=100*np.sum(cones[i-1]<0)/len(cones[i-1])
			if(avoiding<99.99):
				kdeplot(cones[i-1], label='Final Cones', shade=True, color='#ffa804')
			else:
				kdeplot(cones[i-1]-np.amax(cones[i-1])-5, label='Final Cones', shade=True, color='#ffa804')
			ax.set_xlim((-20, 15))
			ax.set_title('Obstacle ' + str((i-1)%3+1))
		fig.set_facecolor('w')
		plt.tight_layout()
		plt.gcf().savefig('../MMD Python Outputs/{}.png'.format('v='+format(v_list[x],'0.3f')+' w='+format(w_list[-1-y],'0.3f'), dpi=300))
		plt.close(fig)
