import numpy as np
import time
import sys


def _init_gibbs(REF, ALT, K, n_gibbs, random_state=0):
    """
    Initialize t=0 state for Gibbs sampling.
    Replace initial word-topic assignment
    ndarray (M, N, N_GIBBS) in-place.

    """
    # initialize
    n_vars, n_cells = REF.shape
    # V: vocab size = n of alleles = 2* n of positions
    np.random.seed(random_state)
    alpha = 1  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
    eta = 200  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V
    print(f"α: {alpha}\nη: {eta}")
    n_di = np.zeros((n_cells, K), dtype=int)

    # word-topic assignment: matrix Z
    assign = np.zeros((n_cells, n_vars, n_gibbs + 1), dtype=int)
    print(f"assign: dim {assign.shape}")
    assign[:, :, 0] = np.random.randint(K, size=(n_cells, n_vars))

    REF_assign_sum = np.zeros((K, n_vars), dtype=int)
    ALT_assign_sum = np.zeros((K, n_vars), dtype=int)

    for d in range(n_cells):
        for k in range(K):
            idx = np.where(assign[d, :, 0].T == k)
            REF_assign_sum[k, idx] += np.sum(REF[idx, d], axis=0)
            ALT_assign_sum[k, idx] += np.sum(ALT[idx, d], axis=0)

            n_di[d, k] = np.sum(assign[d, :, 0] == k, axis=0)

    REF_assign_sum += eta - 1
    ALT_assign_sum += eta - 1

    return alpha, eta, n_di, assign, REF_assign_sum, ALT_assign_sum


def get_phi_ML(var_idx, REF_assign_sum, ALT_assign_sum, sample_phi=True):
    if sample_phi:
        return np.random.beta(a=ALT_assign_sum[:, var_idx], b=REF_assign_sum[:, var_idx])
    else:
        return ALT_assign_sum[:, var_idx] / (ALT_assign_sum[:, var_idx] + REF_assign_sum[:, var_idx])


def _conditional_prob(var_idx, cell_idx, alpha, n_di, c2, REF, ALT, REF_assign_sum, ALT_assign_sum, sample_phi=True):
    """
    P(z_{dn}^i=1 | z_{(-dn)}, w)
    """
    # P(w_dn | z_i)
    phi_ML = get_phi_ML(var_idx, REF_assign_sum, ALT_assign_sum, sample_phi=sample_phi)
    # print(np.round(phi_ML,2))
    _1 = (phi_ML ** ALT[var_idx, cell_idx]) * ((1 - phi_ML) ** REF[var_idx, cell_idx])
    # _1 += 0.01
    # print(np.round(_1/np.sum(_1, axis=0),2))
    # P(z_i | d)
    _2 = ((n_di[cell_idx, :] + alpha) / (n_di[cell_idx, :].sum() + c2))

    # print(_1, _2)
    prob = (_1.T * _2) + sys.float_info.min  # float_info.min to deal with floating point errors
    # print(np.any(np.sum(prob, axis=1) == 0))
    # print(prob.shape)
    # print(np.sum(prob, axis=1))
    # print(np.round(prob, 5))
    return (prob.T / np.sum(prob, axis=1)).T


def run_Gibbs(REF, ALT, K, n_gibbs, blc=100, sample_phi=True, init=None):
    """
        Run collapsed Gibbs sampling
        """
    prev_t = time.time()
    # initialize required variables
    n_vars, n_cells = REF.shape
    if init is None:
        alpha, eta, n_di, assign, REF_assign_sum, ALT_assign_sum = _init_gibbs(REF, ALT, K, n_gibbs)
    else:
        alpha, eta, n_di, assign, REF_assign_sum, ALT_assign_sum = init
    c2 = K * alpha

    print("\n", "=" * 10, "START SAMPLER", "=" * 10)

    # run the sampler
    rand_arr = np.arange(0, n_vars)
    np.random.shuffle(rand_arr)
    dist = []
    dist_final = []
    for t in range(n_gibbs):
        print(t)
        for d in range(n_cells):
            # print(d)

            # np.random.shuffle(rand_arr)
            cycles = int(n_vars / blc) + 1 if (n_vars % blc) > 0 else int(n_vars / blc)
            for rng in range(cycles):
                n = rand_arr[blc * rng: np.min((blc * (rng + 1), n_vars))]
                # decrement counter
                topics = assign[d, n, t]  # previous assignments
                for k in range(K):
                    n_di[d, k] -= np.sum(topics == k)
                    REF_assign_sum[k, n] -= REF[n, d] * (topics == k)
                    ALT_assign_sum[k, n] -= ALT[n, d] * (topics == k)
                # print(np.any(REF_assign_sum == 0))
                # print(np.any(ALT_assign_sum == 0))
                # print(np.any(n_di < 0))
                # assign new topics
                prob = _conditional_prob(n, d, alpha, n_di, c2, REF, ALT, REF_assign_sum, ALT_assign_sum,
                                         sample_phi=sample_phi)
                # print(prob)

                topics = np.argmax([np.random.multinomial(1, prob[i]) for i in range(prob.shape[0])], axis=1)
                for k in range(K):
                    n_di[d, k] += np.sum(topics == k)
                    REF_assign_sum[k, n] += REF[n, d] * (topics == k)
                    ALT_assign_sum[k, n] += ALT[n, d] * (topics == k)
                assign[d, n, t + 1] = topics
        dist.append(np.sum((assign[:, :, t + 1] != assign[:, :, t])))
        dist_final.append(np.sum((assign[:, :, t + 1] != assign[:, :, t])))
        # print out status
        if (t + 1) % 5 == 0:
            print(f"Sampled {t + 1}/{n_gibbs}")
            print("time: ", np.round(divmod(time.time() - prev_t, 60), 0))  # , " (m, s)")
            prev_t = time.time()
            print(dist)
            dist = []

    sigma = np.empty(n_di.shape)
    phi_ML = get_phi_ML(np.arange(0, n_vars), REF_assign_sum, ALT_assign_sum)

    for d in range(n_cells):
        for i in range(K):
            sigma[d, i] = (n_di[d, i] + alpha) / (n_di[d, :].sum() + K * alpha)
    return assign, sigma, phi_ML, dist_final, REF_assign_sum, ALT_assign_sum  # , n_di, n_iw, eta, alpha
