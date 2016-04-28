from Least_Trace_local import Least_Trace_local
from Least_Trace import Least_Trace

import scipy.io as sio
import numpy as np
import copy
import time

mat = sio.loadmat('school.mat')
num_tasks = mat['X'][0].size
X=[]
Y=[]
for i in range(0,num_tasks):
    X.append( np.matrix(mat['X'][0][i], dtype = np.double) )
    Y.append( np.matrix(mat['Y'][0][i], dtype = np.double) )


X2=copy.copy(X)
t0 = time.time()
W2, funcVal2 = Least_Trace_local(X2, Y, 1)
#W, funcVal = Least_Trace(sc, X2, Y, 1)
print time.time() - t0, "seconds wall time"
