import numpy as np
from skdim._commonfuncs import LocalEstimator

class MADA(LocalEstimator):

    _N_NEIGHBORS = 20

    def __init__(self, DM=False):
        self.DM = DM

    def _fit(self, X, dists=None, knnidx=None, n_jobs=1):
        self.dimension_pw_ = self._mada(dists)

    def _mada(self, dists_list):
        n = len(dists_list)
        ests = np.zeros(n)
        for i in range(n):
            dists1 = np.sort(np.array(dists_list[i]))
            k_q = len(dists1)
            if k_q < 2:
                print('Less than 2 knn distances were found, estimation is not possible. This should only occur in the OOB, when using adjustment, otherwise it is a true error. In the OOB case, the inf estimate value is separately handled.')
                ests[i] = np.inf
            else:
                k_half = int(np.floor(k_q / 2))
                rk = dists1[k_q - 1]
                rk2 = dists1[k_half - 1]
                if rk2 == 0 or rk == rk2:
                    raise ValueError('Less than 2 different, non-zero knn distances were found, estimation is not possible.')
                else:
                    ests[i] = np.log(2) / np.log(rk / rk2)
        return ests