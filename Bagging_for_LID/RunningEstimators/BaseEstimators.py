from tqdm import tqdm
import sys
import skdim
import numpy as np
###################################################OWN IMPORT###################################################
from Bagging_for_LID.RunningEstimators.RewrittenRawEstimators.MADA import *
from Bagging_for_LID.RunningEstimators.BaggingSmoothing.Smoothing import *
###############################################################################################################################BASE ESTIMATORS###############################################################################################################################
def sk_MLE(X, dists, knnidx, k=10, correct=True, w=None, return_ks=False, use_w='direct', smooth=False, geo=None,
           smooth_style='code2', bag_indices=None):
    if w is None:
        mle = skdim.id.MLE()
        mle.fit(X, n_neighbors=k, comb='mean', precomputed_knn_arrays=(dists, knnidx))
        if correct:
            lid_estimates = k / (k - 1) * mle.dimension_pw_
        else:
            lid_estimates = mle.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo,
                                                      smooth_style=smooth_style)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo,
                                                      smooth_style=smooth_style, bag_indices=bag_indices)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        for q in range(n):
            if use_w == 'direct':
                lid_estimates[q] = - len(dists[q]) / np.sum(np.log(dists[q] / w[q]))
                ks[q] = len(dists[q])
            elif use_w == 'indirect':
                if dists[q][-1] != np.max(dists[q]):
                    Warning('Distances are not ordered. Check failed.')
                lid_estimates[q] = - len(dists[q]) / np.sum(np.log(dists[q] / dists[q][-1]))
                ks[q] = len(dists[q])
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)

def sk_TLE(X, dists, knnidx, k=10, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None,
           smooth_style="code2", bag_indices=None):
    if w is None:
        tle = skdim.id.TLE()
        tle._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = tle.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo,
                                                      smooth_style=smooth_style)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo,
                                                      smooth_style=smooth_style, bag_indices=bag_indices)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        #n = X.shape[0]
        #lid_estimates = np.empty(n)
        #ks = np.empty(n)
        #tle = TLE()
        #tle._fit(X, dists_list=dists, knnidx_list=knnidx)
        #lid_estimates = tle.dimension_pw_
        #if return_ks:
        #    return lid_estimates, np.mean(lid_estimates), ks
        #else:
        #    return lid_estimates, np.mean(lid_estimates)
        NotImplemented(f"Not implemented use_w: {use_w}")
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def sk_MADA(X, dists, knnidx, k=10, correct=True, w=None, return_ks=False, use_w='indirect', smooth=False, geo=None,
            smooth_style='code2', bag_indices=None):
    if w is None:
        mada = MADA()
        mada._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = mada.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo,
                                                      smooth_style=smooth_style)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo,
                                                      smooth_style=smooth_style, bag_indices=bag_indices)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        mada = MADA()
        mada._fit(X, dists=dists, knnidx=knnidx)
        lid_estimates = mada.dimension_pw_
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def sk_MLE_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mle = skdim.id.MLE()
    mle.fit(X, n_neighbors=k, comb='mean', precomputed_knn_arrays=(dists, knnidx))
    if correct:
        lid_estimates = k / (k - 1) * mle.dimension_pw_
    else:
        lid_estimates = mle.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_TLE_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    tle = skdim.id.TLE()
    tle._fit(X, dists=dists, knnidx=knnidx)
    lid_estimates = tle.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_MADA_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mada = MADA()
    mada._fit(X, dists=dists, knnidx=knnidx)
    lid_estimates = mada.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_MOM(X, dists, knnidx, k = 10, w=None, return_ks = False, use_w = 'direct', smooth=False, geo=None):
    if w is None:
        mom = skdim.id.MOM()
        lid_estimates = mom._mom(dists)
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    else:
        n = X.shape[0]
        lid_estimates = np.empty(n)
        ks = np.empty(n)
        for q in range(n):
            if use_w == 'direct':
                mu_hat = np.mean(dists[q])
                lid_estimates[q] = - mu_hat/(mu_hat-w[q])
                ks[q] = len(dists[q])
            elif use_w == 'indirect':
                if dists[q][-1] != np.max(dists[q]):
                    Warning('Distances are not ordered. Check failed.')
                mu_hat = np.mean(dists[q])
                lid_estimates[q] = - mu_hat / (mu_hat - np.max(dists[q]))
                ks[q] = len(dists[q])
        if return_ks:
            return lid_estimates, np.mean(lid_estimates), ks
        else:
            return lid_estimates, np.mean(lid_estimates)

def sk_2NN(X, dists, knnidx, k = 10, correct = True, w = None, return_ks = False, use_w = 'indirect', smooth=False, geo=None):
    if w is None:
        twonn = skdim.id.TwoNN()
        twonn.fit_pw(X, precomputed_knn=knnidx, smooth=False, n_neighbors=k, n_jobs=1)
        lid_estimates = twonn.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        #n = X.shape[0]
        #lid_estimates = np.empty(n)
        #ks = np.empty(n)
        #twonn = skdim.id.TwoNN()
        #twonn = fit_pw_with_list(twonn, X, knnidx, smooth=False)
        #lid_estimates = twonn.dimension_pw_
        #if return_ks:
        #    return lid_estimates, np.mean(lid_estimates), ks
        #else:
        #    return lid_estimates, np.mean(lid_estimates)
        NotImplemented(f"Not implemented use_w: {use_w}")
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")


def sk_ESS(X, dists, knnidx, k = 10, correct = True, w=None, return_ks = False, use_w = 'indirect', smooth=False, geo=None):
    if w is None:
        est_ess = skdim.id.ESS()
        est_ess._fit(X, dists, knnidx)
        lid_estimates = est_ess.dimension_pw_
        if return_ks:
            ks = np.repeat(k, X.shape[0])
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates), ks
            else:
                return lid_estimates, np.mean(lid_estimates), ks
        else:
            if smooth:
                lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=dists, knnidx=knnidx, geo=geo)
                return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
            else:
                return lid_estimates, np.mean(lid_estimates)
    elif use_w == 'indirect':
        #n = X.shape[0]
        #lid_estimates = np.empty(n)
        #ks = np.empty(n)
        #est_ess = ESS()
        #est_ess._fit(X, dists=dists, knnidx=knnidx)
        #lid_estimates = est_ess.dimension_pw_
        #if return_ks:
        #    return lid_estimates, np.mean(lid_estimates), ks
        #else:
        #    return lid_estimates, np.mean(lid_estimates)
        NotImplemented(f"Not implemented use_w: {use_w}")
    else:
        NotImplemented(f"Not implemented use_w: {use_w}")

def sk_MOM_full(X, k = 10, correct = True, dists=None, knnidx=None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    mom = skdim.id.MOM()
    lid_estimates = mom._mom(dists)
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_2NN_full(X, k = 10, correct = False, dists=None, knnidx=None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    twonn = skdim.id.TwoNN()
    twonn.fit_pw(X, precomputed_knn=knnidx, smooth=False, n_neighbors=k, n_jobs=1)
    lid_estimates = twonn.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

def sk_ESS_full(X, k = 10, correct = True, dists = None, knnidx= None, w=None, smooth=False, geo=None):
    if (dists is None) or (knnidx is None):
        dists, knnidx = skdim._commonfuncs.get_nn(X, k=k, n_jobs=1)
    est_ess = skdim.id.ESS()
    est_ess._fit(X, dists, knnidx)
    lid_estimates = est_ess.dimension_pw_
    if smooth:
        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
    else:
        return lid_estimates, np.mean(lid_estimates)

#def LIDL_full(X, k = 10, correct = True, dists = None, knnidx= None, model_type="gm", w=None, smooth=False, geo=None):
#    gm = dim_estimators.LIDL(model_type=model_type)
#    # the more deltas, the more accurate the estimate
#    deltas = [0.025, 0.02835781, 0.03216662, 0.036487, 0.04138766,0.04694655,\
#              0.05325205, 0.06040447, 0.06851755, 0.07772031, 0.08815913, 0.1]
#    result = gm(deltas, X, X)
#    lid_estimates = np.array(result)
#    if smooth:
#        lid_smoothed_estimates, _ = smoothing(X, lid_estimates, k=k, dists=None, knnidx=None, geo=geo)
#        return lid_smoothed_estimates, np.mean(lid_smoothed_estimates)
#    else:
#        return lid_estimates, np.mean(lid_estimates)