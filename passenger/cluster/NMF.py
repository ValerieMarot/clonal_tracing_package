from scipy.stats import binom
import scipy.optimize as op
import numpy as np
from joblib import Parallel, delayed
from itertools import permutations


def get_weights(cov, X, X_high_conf=False, mode=""):
    # set weight matrix
    weights = np.zeros(cov.shape)
    if mode == "VAF":
        # Set S to probability of missing ALT when randomly sampling cov times
        # note: this is probably not adapted to genetic variants
        b = 0  # prob of monoallelic expression
        for i in np.unique(cov):
            p = 1 - (b + (1 - b) * binom.pmf(k=0, n=i, p=.5))
            weights[np.where(cov == i)] = p
        if X_high_conf:  # observing a variant set to higher confidence
            weights[X > 0] = 1
    else:
        weights[np.where(cov >= 2)] = 0.5
        weights[X == 0.5] = 1
    return weights


def get_varcall(REF, ALT, mode):
    # set X matrix
    cov = REF + ALT
    X = np.zeros(REF.shape)
    whr = np.where(cov >= 2)
    if mode == "VAF":  # M contains exact VAF
        X[whr] = (ALT[whr] / cov[whr])
    else:  # M contains VAF=.5 if we see ref and alt, VAF=1 if we see only ALT, VAF=0 else
        X[whr] = ((ALT[whr] >= 1) * (REF[whr] >= 1) * 0.5) + \
                 ((ALT[whr] >= 1) * (REF[whr] == 0) * 1)
    return X


def NMF_weighted(X, weights, k=2, max_cycles=1000,
                 parallel=True, n_cores=None):
    """
    Function to calculate weighted NMF given variant call matrix X and weight matrix (of equal shape as X).
    We try to learn C (cells*k) and V (k*variants) that minimise the cost function:
        ((X-CV)**2)oW     with o the hadamart product
    There is no closed solution of this equation, instead we alternatively minimize C and V.
    In this equation the rows of C and columns of V are independent. Because of this, we can mimize them in parallel
    for speedup.

    Parameters
    ----------
    X: `float` np.array [n_vars * n_cells]  # todo make it be n_cells * n_vars
        Variant call matrix. Contains VAF of variant for that cell / position.
    weights: `float` np.array [n_cells * n_vars]
        Weight matrix. Should reflect the confidence of our variant calls.
    k: `int` (default: 2)
        Number of clusters
    max_cycles: `int` (default: 1000)
        Number of cycles to fit C and V before break.
    parallel: `bool` (default: True)
        Whether to fit rows of C and columns of V in parallel for runtime speed-up.
    n_cores: `int` (default: None)
        How many cores to run on in parallel. Per default we set it to the number of available CPUs.

    Returns
    -------
    C: `float` np.array [n_cells * k]
        Final cell weights for weighted NMF.
    V: `float` np.array [k * n_vars]
        Final variant weights for weighted NMF.
    """
    # init
    bounds = np.repeat([[0, 1]], k, 0)  # bounds of C and V matrices (values in these matrices should be btw 0 and 1)
    v_entries, c_entries = X.shape[0], X.shape[1]
    # initialize C and V with random values btw 0 and 1:
    V = np.random.rand(v_entries, k)
    C = np.random.rand(k, c_entries)
    cost = []  # to check wether we can break iteration before 1000 runs

    # for speedup:
    if parallel:
        if n_cores is None:
            import os
            n_cores = os.cpu_count()
        # set size of C and V blocks to send to each core. Note that too small blocks here will slow down the overall
        # runtime because of send & fetch delays.
        n_parts_V = np.max((int(v_entries / n_cores), 10))
        n_parts_C = np.max((int(c_entries / n_cores), 10))
    for i in range(max_cycles):
        if i % 100 == 0:
            print(i)

        #########
        # fit V #
        #########
        def fun_v(v, h):  # helper function for minimize call
            return np.sum(((np.dot(v, C) - X[h]) ** 2) * weights[h])

        if parallel:
            V = parallel_run(V, v_entries, n_parts_V, fun_v, bounds, n_jobs=n_cores)
        else:
            for j in np.arange(0, X.shape[0]).tolist():
                V[j] = op.minimize(fun_v, V[j], bounds=bounds, args=j).x

        #########
        # fit C #
        #########
        def fun_c(c, h):  # helper function for minimize call
            return np.sum(((np.dot(V, c) - X[:, h]) ** 2) * weights[:, h])

        if parallel:
            C = parallel_run(C, c_entries, n_parts_C, fun_c, bounds, axis=1, n_jobs=n_cores).T
        else:
            for j in np.arange(0, X.shape[0]).tolist():
                C[:, j] = op.minimize(fun_c, C[:, j], bounds=bounds, args=j).x

        ###############
        # break check #
        ###############
        cost.append(np.sum(((np.clip((np.dot(V, C)), 0, 1) - X) ** 2) * weights))
        if (i % 10 == 0) & (i > 0):  # can break run if the cost function is not getting better anymore
            if (np.mean(cost[i - 10:i - 5]) - np.mean(cost[i - 4:i])) < 0:
                print("breaking at iteration " + str(i))
                break
    return C, V


def parallel_run(in_arr, n_entries, n_parts, fun, bounds, n_jobs, axis=0):
    """
    Helper function to run minimize in parallel. This is needed to send and fetch the running block to the CPU.
    """
    # todo make this easier to read?
    # prepare input
    pars = []
    for j in range(n_entries):
        pars.append((in_arr[j] if axis == 0 else in_arr[:, j], j))

    # run fitting
    def fit_helper(pars_):
        out = []
        for p in pars_:
            out.append(op.minimize(fun, p[0], bounds=bounds, args=p[1]).x)
        return out

    n_ = int(np.ceil(n_entries / n_parts))
    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_helper)(pars[i:i + n_]) for i in range(0, n_entries, n_))

    # format result
    r = []
    for i in result:
        r.extend(i)
    return np.array(r)


def bootstrap_wNMF(REF, ALT, k=2, n_bootstrap=10, bootstrap_percent=.9,
                   max_cycles=1000, parallel=True, n_cores=None, mode="VAF_discrete"):
    """
    Bootstrap function for the wNMF. We are not guaranteed to find a global minimum for the wNMF and the output
    will change over multiple iteration. To get an estimation of the robustnes we run the wNMF multiple times using
    a subset of the variants as input. We then get n_bootstrap C and V matrices, align these and return the mean of
    the runs as cell weights. We also return the std of the runs to get an estimation of the robustness of these
    assignments.

    Parameters
    ----------
    REF: `float` np.array [n_vars * n_cells]  # todo make it be n_cells * n_vars
        Count matrix for reference reads.
    ALT: `float` np.array [n_vars * n_cells]  # todo make it be n_cells * n_vars
        Count matrix for reference reads.
    k: `int` (default: 2)
        Number of clusters
    n_bootstrap: `int` (default: 10)
        Number of boostrap runs.
    bootstrap_percent: `float` (default: 0.9)
        Percentage of variants kept in each of the bootstrap iterations.
    max_cycles: `int` (default: 1000)
        Number of cycles to fit C and V before break.
    parallel: `bool` (default: True)
        Whether to fit rows of C and columns of V in parallel for runtime speed-up.
    n_cores: `int` (default: None)
        How many cores to run on in parallel. Per default we set it to the number of available CPUs.
    mode: `str` in ['VAF_discrete', 'VAF'] (default: 'VAF_discrete')
        Whether to use the exact VAF as observation or the discretised ones assuming diploid model (0 - 0.5 - 1).
   Returns
    -------

    """
    n_vars = ALT.shape[0]
    C_all, V_all = [], []

    ######################
    # run bootstrap wNMF #
    ######################
    for i in range(n_bootstrap):
        # subset to bootstrap_percent% of variants
        idx = np.random.choice(np.arange(n_vars), int(n_vars * bootstrap_percent), replace=False)
        REF_sub, ALT_sub = REF[idx], ALT[idx]
        cov = REF_sub + ALT_sub
        # get variant call matrix and weight matrix
        X = get_varcall(REF_sub, ALT_sub, mode=mode)
        S = get_weights(cov, X, mode=mode)  # S probability matrix of seing mutation given the coverage
        # run wNMF
        C, V = NMF_weighted(X, weights=S, k=k, max_cycles=max_cycles, parallel=parallel, n_cores=n_cores)
        V_ = np.zeros(shape=(n_vars, k)) * np.nan  # weights of excluded variants set to nan
        V_[idx] = V
        C_all.append(C), V_all.append(V_)
    C_all, V_all = np.array(C_all), np.array(V_all)

    #######################
    # align the wNMF runs #
    #######################
    # We align all runs to maximize simlarity to the first one. # todo maybe compare all to all?
    # Note: this could cause issues if the first run is the only one that didn't work. However, assuming the wNMF works
    # reasonably well for that patient it should be ok. And if the results are untrustworthy we assume the method fails.
    clus1 = np.argmax(C_all[0], axis=0)
    perm = list(permutations(np.arange(0, k)))
    for i in np.arange(0, n_bootstrap):
        d = np.zeros((k, k))
        clus2 = np.argmax(C_all[i], axis=0)
        for j in range(k):
            for h in range(k):
                d[j, h] = (np.sum((clus1 == j) & (clus2 == h)))
        p = [perm[np.argmax(np.sum(d[np.arange(0, k), perm], axis=1))]]
        C_all[i] = C_all[i][p][0]
        V_all[i] = V_all[i].T[p][0].T  # this part does not work give the same output in older versions of numpy
        # that's why we have numpy~=1.25.0 in the requirements

    ######################
    #     get output     #
    ######################
    C, C_std = np.mean(np.array(C_all), axis=0), np.nanstd(C_all, axis=0)
    V, V_std = np.nanmean(V_all, axis=0), np.nanstd(V_all, axis=0)

    return C, C_std, V, V_std
