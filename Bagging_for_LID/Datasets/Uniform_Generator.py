import numpy as np
#################################################################################################################

def Simple_LID_data(n=2500, lid=1, dim=2):
    res = np.empty((n, dim))
    for i in range(n):
        if lid > 1:
            U = np.random.uniform(size=lid-1)
            u = np.random.uniform(size=1)
            res[i][:lid-1] = U[0:lid-1]
            res[i][lid-1:] = u
        else:
            u = np.random.uniform(size=1)
            res[i] = u
    return res

