import numpy as np
import reduced_sets_method
dist=np.random.randn(10000,2,5)*1 + 0
[x,y]=reduced_sets_method.reduced_sets_method(dist,25)
print(np.sum(y))