from scipy.stats import binom
import scipy.optimize as op
import numpy as np
from joblib import Parallel, delayed
import random
from itertools import permutations


def get_state(cov, M, M_high_conf=True):
    # set state matrix
    S = np.zeros(cov.shape)
    b = 0  # prob of monoallelic expression
    for i in np.unique(cov):
        p = 1 - (b + (1 - b) * binom.pmf(k=0, n=i, p=.5))
        S[np.where(cov == i)] = p
    if M_high_conf:
        S[M > 0] = 1
    return S


def NMF_weighted(X, weights, k=2, max_cycles=25, force_cell_assignment=False,
                 break_iteration=True,
                 parallel=True, n_cores=None, reg=0):
    """

    Parameters
    ----------
    X: mutation by cells matrix
    weights:
    k: # of clusters
    max_cycles: # of cycles before break
    break_iteration:
    force_cell_assignment: whether to force cells to be assigned to exactly and only 1 cluster. If True this sets
                                  an additional constraint on the minimize function with sum(row C) == 1.
    parallel:
    n_cores:
    reg:

    Returns
    -------

    """
    # init
    bounds = np.repeat([[0, 1]], k, 0)
    v_entries, c_entries = X.shape[0], X.shape[1]
    V = np.random.rand(v_entries, k)
    C = np.random.rand(k, c_entries)
    # optional constraint
    if force_cell_assignment:
        def con(c):
            return np.sum(c) - 1

        cons = {'type': 'eq', 'fun': con}  # note: eq means that minimize forces con==1

    if parallel:
        if n_cores is None:
            import os
            n_cores = os.cpu_count()
        n_parts_V = int(v_entries / n_cores)
        n_parts_C = int(c_entries / n_cores)

    for i in range(max_cycles):

        V_ = V.copy()
        C_ = C.copy()

        #########
        # fit V #
        #########
        def fun_v(v, h):
            # note: in the function we use np.clip because np.dot(v, C) can be > 1 (e.g. if a variant is associated
            # with multiple clones, and a cell with multiple clones), but x cannot be > 1
            # -> if we have np.dot > 1 and x==1 this should not be penalised
            reg_ = reg * np.sum(v ** 2)
            return np.sum((np.clip((np.dot(v, C) * weights[h]), 0, 1) - X[h]) ** 2) + reg_

        if parallel:
            V_ = parallel_run(V_, v_entries, n_parts_V, fun_v, bounds, constraints=None)
        else:
            for j in np.arange(0, X.shape[0]).tolist():
                V_[j] = op.minimize(fun_v, V[j], bounds=bounds, args=j).x

        #########
        # fit C #
        #########
        def fun_c(c, h):
            # same as above with np.clip
            return np.sum((np.clip((np.dot(V_, c) * weights[:, h]), 0, 1) - X[:, h]) ** 2)

        if parallel:
            C_ = parallel_run(C_, c_entries, n_parts_C, fun_c, bounds,
                              constraints=cons if force_cell_assignment else None, axis=1).T
        else:
            for j in np.arange(0, X.shape[0]).tolist():
                C_[:, j] = op.minimize(fun_c, C[:, j], bounds=bounds,
                                       constraints=cons if force_cell_assignment else None,
                                       args=j).x
        ###############
        # break check #
        ###############
        it_v, it_c = np.sum((V_ - V) ** 2), np.sum((C_ - C) ** 2)
        V, C = V_, C_

        if (it_c / C.shape[1] < .001) & (it_v / V.shape[0] < .001) & break_iteration:
            print("breaking at iteration " + str(i))
            print("cost " + str(np.sum((np.clip((np.dot(V, C) * weights), 0, 1) - X) ** 2)))
            print(np.sum(V ** 2))
            break

    return C, V


def parallel_run(in_arr, n_entries, n_parts, fun, bounds, constraints, axis=0):
    # prepare input
    pars = []
    for j in range(n_entries):
        pars.append((in_arr[j] if axis == 0 else in_arr[:, j], j))
    pars = np.array(pars)

    # partition
    to_split = range(0, n_entries)
    n_ = int(np.ceil(n_entries / n_parts))
    output = [to_split[i:i + n_] for i in range(0, n_entries, n_)]

    # run fitting
    def fit_helper(pars_):
        out = []
        for p in pars_:
            out.append(op.minimize(fun, p[0], bounds=bounds,
                                   constraints=constraints,
                                   args=p[1]).x)
        return out

    result = Parallel(n_jobs=8, backend="loky")(delayed(fit_helper)(pars[i]) for i in output)

    # format result
    r = []
    for i in result:
        r.extend(i)
    return np.array(r)


def bootstrap_wNMF(REF, ALT, k=2, VAF_thres=.2, n_bootstrap=10, bootstrap_percent=.9,
                   max_cycles=25, force_cell_assignment=False,
                   break_iteration=True,
                   parallel=True, n_cores=None, reg=0):

    # run wNMF
    C_all, V_all = [], []
    print("running bootstrap")
    for i in range(n_bootstrap):
        print()
        REF_sub = subsample(REF, bootstrap_percent)
        ALT_sub = subsample(ALT, bootstrap_percent)
        cov = REF_sub + ALT_sub
        M = ((ALT / cov) > VAF_thres) & (ALT >= 2)

        S = get_state(cov, M)  # S probability matrix of seing mutation given the coverage
        C, V = NMF_weighted(M,
                            weights=S,
                            k=k,  # number of clusters we are looking for
                            max_cycles=max_cycles, force_cell_assignment=force_cell_assignment,
                            break_iteration=break_iteration,
                            parallel=parallel, n_cores=n_cores, reg=reg
                            )
        C_all.append(C), V_all.append(V)

    # align the NMF runs
    clus1 = np.argmax(C_all[0], axis=0)
    perm = list(permutations(np.arange(0, k)))
    for i in np.arange(0, n_bootstrap):
        d = np.zeros((k, k))
        clus2 = np.argmax(C_all[i], axis=0)
        for j in range(k):
            for h in range(k):
                d[j, h] = (np.sum((clus1 == j) & (clus2 == h)))
        p = [perm[np.argmax(np.sum(d[np.arange(0, k), perm], axis=1))]]
        C_all[i] = C_all[i][p]
        V_all[i] = V_all[i].T[p].T

    aggr = np.sum(np.argmax(C_all, axis=1), axis=0)
    conf = 1 - np.min((binom.cdf(aggr, n_bootstrap, 1 / k),
                       binom.cdf(n_bootstrap - aggr, n_bootstrap, 1 / k)), axis=0)
    C = np.mean(np.array(C_all), axis=0)

    V = np.nanmean(V_all, axis=0)
    V_std = np.nanstd(V_all, axis=0)

    return C, conf, V, V_std


def subsample(mat, bootstrap_percent):
    """
    https://stackoverflow.com/questions/11818215/subsample-a-matrix-python
    """
    keys, counts = zip(*[
        ((i, j), mat[i, j])
        for i in range(mat.shape[0])
        for j in range(mat.shape[1])
        if mat[i, j] > 0
    ])
    # Make the cumulative counts array
    counts = np.array(counts, dtype=np.int64)
    sum_counts = np.cumsum(counts)

    # Decide how many counts to include in the sample
    count_select = int(sum_counts[-1] * bootstrap_percent)

    # Choose unique counts
    ind_select = sorted(random.sample(range(sum_counts[-1]), count_select))

    # A vector to hold the new counts
    out_counts = np.zeros(counts.shape, dtype=np.int64)

    # Perform basically the merge step of merge-sort, finding where
    # the counts land in the cumulative array
    i, j = 0, 0
    while i < len(sum_counts) and j < len(ind_select):
        if ind_select[j] < sum_counts[i]:
            j += 1
            out_counts[i] += 1
        else:
            i += 1

    # Rebuild the matrix using the `keys` list from before
    out_mat = np.zeros(mat.shape, dtype=np.int64)
    for i in range(len(out_counts)):
        out_mat[keys[i]] = out_counts[i]

    return out_mat