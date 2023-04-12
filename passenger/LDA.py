import numpy as np
import time


def init_lda(n_cells, V, K, random_state=0):
    # initialize alpha, eta
    np.random.seed(random_state)
    alpha = 1  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
    eta = 1  # np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V
    print(f"α: {alpha}\nη: {eta}")

    n_iw = np.zeros((K, V), dtype=int)  # n of times word w has been assigned to topic k
    n_di = np.zeros((n_cells, K), dtype=int)  # n of times words in cells have been assigned to topic k

    return alpha, eta, n_iw, n_di


def _init_gibbs(Muts, cov, K, n_gibbs=2000):
    """
    Initialize t=0 state for Gibbs sampling.
    Replace initial word-topic assignment
    ndarray (M, N, N_GIBBS) in-place.

    """
    # initialize
    n_vars, n_cells = Muts.shape
    # V: vocab size = n of alleles = 2* n of positions
    # in our case the number of words per document (N) == vocab size
    V = n_vars  # *2
    alpha, eta, n_iw, n_di = init_lda(n_cells, V, K=K)

    # word-topic assignment: matrix Z
    assign = np.zeros((n_cells, n_vars, n_gibbs + 1), dtype=int)
    print(f"assign: dim {assign.shape}")

    # word matrix -- this represents the hidden X matrix, init based on X'
    hidden_muts = Muts.copy()
    see_mut_prob = np.ones(hidden_muts.shape)

    # initial assignment
    for d in range(n_cells):
        for n in range(n_vars):
            w_dn = Muts[n, d]
            c_dn = cov[n, d]
            # set hidden_muts based on muts and cov:
            if not w_dn:
                p = .5 ** (c_dn + 1)
                see_mut_prob[n, d] = p
                hidden_muts[n, d] = np.random.binomial(1, see_mut_prob[n, d])

            # randomly assign topic to word w_{dn}
            h_w_dn = hidden_muts[n, d]
            assign[d, n, 0] = np.random.randint(K)

            # increment counters
            if h_w_dn:
                i = assign[d, n, 0]
                n_iw[i, n] += 1
                n_di[d, i] += 1

    return alpha, eta, n_iw, n_di, assign, hidden_muts, see_mut_prob, V


def _conditional_prob(n, d, eta, alpha, n_iw, n_di, c1, c2):
    """
    P(z_{dn}^i=1 | z_{(-dn)}, w)
    """

    # P(w_dn | z_i)
    _1 = (n_iw[:, n] + eta).T / (n_iw.sum(axis=1) + c1)
    # P(z_i | d)
    _2 = (n_di[d, :] + alpha) / (n_di[d, :].sum() + c2)
    prob = _1 * _2

    return (prob.T / prob.sum(axis=1)).T


def run_Gibbs(Muts, cov, K, n_gibbs, blc=150):
    """
        Run collapsed Gibbs sampling
        """
    prev_t = time.time()
    # initialize required variables
    n_vars, n_cells = Muts.shape
    alpha, eta, n_iw, n_di, assign, hidden_muts, see_mut_prob, V = _init_gibbs(Muts, cov, K, n_gibbs)
    c1 = V * eta
    c2 = K * alpha

    print("\n", "=" * 10, "START SAMPLER", "=" * 10)

    # run the sampler
    rand_arr = np.arange(0, n_vars)
    np.random.shuffle(rand_arr)
    for t in range(n_gibbs):

        for d in range(n_cells):
            # np.random.shuffle(rand_arr)
            for rng in range(int(n_vars / blc)):
                n = rand_arr[blc * rng: blc * (rng + 1)]

                # decrement counter
                i_t = assign[d, n, t]  # previous assignments
                h_w_d = hidden_muts[n, d]  # limit ourselves to existing mutations
                for k in range(K):
                    idx = n[h_w_d & (i_t == k)]
                    n_iw[k, idx] -= 1
                    n_di[d, k] -= np.sum(i_t[h_w_d] == k)

                # reset hidden_muts based on muts and cov:
                idx = n[~Muts[n, d]]
                hidden_muts[idx, d] = np.random.binomial(1, see_mut_prob[idx, d])

                # assign new topics
                prob = _conditional_prob(n, d, eta, alpha, n_iw, n_di, c1, c2)
                i_tp1 = np.argmax([np.random.multinomial(1, prob[i]) for i in range(blc)], axis=1)
                h_w_d = hidden_muts[n, d]  # limit ourselves to existing mutations
                for k in range(K):
                    idx = n[h_w_d & (i_tp1 == k)]
                    n_iw[k, idx] += 1
                    n_di[d, k] += np.sum(i_tp1[h_w_d] == k)
                # increment counter with new assignment

                assign[d, n, t + 1] = i_tp1
        # print out status
        if (t + 1) % 5 == 0:
            print(f"Sampled {t + 1}/{n_gibbs}")
            print("time: ", np.round(divmod(time.time() - prev_t, 60), 0))  # , " (m, s)")
            prev_t = time.time()

    beta = np.empty(n_iw.shape)
    sigma = np.empty(n_di.shape)

    for j in range(V):
        for i in range(K):
            beta[i, j] = (n_iw[i, j] + eta) / (n_iw[i, :].sum() + V * eta)

    for d in range(n_cells):
        for i in range(K):
            sigma[d, i] = (n_di[d, i] + alpha) / (n_di[d, :].sum() + K * alpha)
    return beta, sigma  # , n_di, n_iw, eta, alpha
